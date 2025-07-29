from typing import Dict, List, Optional, Union, Any
import os
import torch
from loguru import logger
from prometheus_client import Histogram, Counter, Gauge
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from core.config import settings
from models.reranker.base import BaseReranker, RerankerConfig

class TransformerReranker(BaseReranker):
    """通用Transformer重排器基类，适用于HuggingFace模型。"""

    # Prometheus metrics
    LATENCY_HIST = Histogram(
        'local_reranker_latency_seconds',
        'Latency for local reranker requests',
        ['model_name']
    )
    
    ERROR_COUNTER = Counter(
        'local_reranker_errors_total',
        'Count of local reranker errors',
        ['model_name', 'error_type']
    )
    
    MEMORY_GAUGE = Gauge(
        'local_reranker_memory_usage_mb',
        'GPU memory usage in MB',
        ['device_id']
    )

    def __init__(self, config: RerankerConfig):
        """
        初始化通用Transformer重排器。
        
        Args:
            config: 模型配置
            default_model_name: 默认模型名称
            metrics_prefix: Prometheus指标前缀
        """
        # Model components
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.predownload = False  # 是否为本地预下载模型
        self.cache_path = os.path.join("./models/local_model_cache", self.model_name)  # 模型缓存路径
        self.name_or_path = ""
        self.device = config.device
        self.device_map = config.device_map
        self.local_files_only = getattr(config, 'local_files_only', False)
        self.trust_remote_code = getattr(config, 'trust_remote_code', False)
        self.max_tokens = getattr(config, 'max_tokens', settings['reranker']['max_tokens'])
        self.compile_model = getattr(config, 'compile_model', True)
        self.use_fp16 = getattr(config, 'use_fp16', False)
        self.max_memory = getattr(config, 'max_memory', None)
        

        # Runtime state
        self._is_initialized = False
            
        logger.info(f"Initializing {self.__class__.__name__} with model: {self.model_name}")
    
    def _validate_reranker_model(self, model_path: str) -> bool:
        """Check if the model directory contains necessary files. //检查模型目录是否包含必要文件"""
        must_have = ["config.json", "tokenizer_config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        vocab_files = ["vocab.txt", "vocab.json", "tokenizer.json"]
        
        # 检查必备文件
        for f in must_have:
            if not os.path.exists(os.path.join(model_path, f)):
                return False
        
        # 检查模型权重文件(至少存在一种)
        if not any(os.path.exists(os.path.join(model_path, f)) for f in model_files):
            return False
        
        # 检查词汇表文件(至少存在一种)
        if not any(os.path.exists(os.path.join(model_path, f)) for f in vocab_files):
            return False
        
        return True
        
    def _cache_model(self):
        """Save model to cache directory."""
        self.model.save_pretrained(self.cache_path) # type: ignore
        self.tokenizer.save_pretrained(self.cache_path) # type: ignore
    
    async def startup(self) -> None:
        """Initialize the reranker model."""
        if self._is_initialized:
            return

        if self.model_path is None and self.local_files_only:
            raise ValueError("Local model path not specified.")
        
        if self.model_path is not None:
            valid_path = self._validate_reranker_model(self.model_path)
            if valid_path:
                self.predownload = True
            else:
                valid_cache = self._validate_reranker_model(self.cache_path)
                if valid_cache:
                    logger.info(f"Using cached reranker model: {self.cache_path}")
                    self.model_path = os.path.abspath(self.cache_path)
                    self.predownload = True
                else:
                    self.predownload = False

        if self.predownload:
            self.name_or_path = self.model_path
        else:
            self.name_or_path = self.model_name

        logger.debug(f"Reranker model name: {self.model_name}, path: {self.model_path}")
        logger.debug(f"Reranker model name or path: {self.name_or_path}")
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = self.name_or_path,
            trust_remote_code = self.trust_remote_code,
            use_fast = True,
            model_max_length = self.max_tokens,
            padding_side = 'right',
            local_files_only = self.local_files_only
        )
            
        # Load model with optimized settings
        load_kwargs = {
            "pretrained_model_name_or_path": self.name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
        }
            
        # Device configuration
        if torch.cuda.is_available():
            if self.device_map is not None:  # Multi-GPU
                load_kwargs.update({
                    "device_map": self.device_map,
                    "max_memory": self.max_memory,
                })
            else:  # Single GPU
                # 移除device参数，稍后使用.to()方法
                target_device = self.device or "cuda:0"
                
            # Precision control
            load_kwargs["torch_dtype"] = torch.float16 if self.use_fp16 else torch.float32
        else:  # CPU fallback
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
            target_device = "cpu"

        self.model = AutoModelForSequenceClassification.from_pretrained(**load_kwargs)
        
        # 如果是首次使用从HuggingFace下载的模型，则将模型从默认缓存路径保存到本地
        if self.predownload is not True:
            try:
                self._cache_model()
                logger.debug(f"Reranker model {self.model_name} downloaded to local cache: {self.cache_path}")
            except Exception as e:
                logger.error(f"Error saving reranker model to local cache: {e}")

        # 如果没有使用device_map，则使用.to()方法将模型移动到指定设备
        if self.device_map is None:
            self.model = self.model.to(target_device) # type: ignore
            # 确保self.device与实际使用的设备一致
            self.device = target_device
            logger.debug(f"Reranker model loaded to device: {target_device}")
        else:
            logger.debug(f"Reranker model loaded with device_map: {self.device_map}")
        
        # 记录模型参数所在的设备
        sample_param = next(self.model.parameters()) # type: ignore
        logger.debug(f"Reranker model parameters located on device: {sample_param.device}")
        
        self.model.eval() # type: ignore

        # Model compilation (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile( # type: ignore
                self.model,
                mode='max-autotune' if torch.cuda.is_available() else None
            )

        self._is_initialized = True
        logger.info(f"Reranker model {self.model_name} initialized successfully.")
    
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None for all)
            return_scores: Whether to return scores with indices
            
        Returns:
            List of dicts with 'index' and 'score' keys
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call startup() first.")
        
        if not documents:
            return []
        
        # Set top_k to number of documents if not specified
        if top_k is None:
            top_k = len(documents)
        else:
            top_k = min(top_k, len(documents))
        
        try:
            # Prepare pairs for reranking
            pairs = [(query, doc) for doc in documents]
            
            # Tokenize pairs
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_tokens
                )
                
                # 根据模型配置方式处理设备分配
                if self.device_map is None:
                    # 单设备模式：将输入移动到模型所在的设备
                    inputs = inputs.to(self.device)
                else:
                    # 多设备模式：让模型处理设备分配
                    # 不要手动移动输入张量，让模型的forward方法处理
                    pass
                
                # Get scores
                scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()
            
            # Create list of (index, score) tuples
            scored_results = [(i, score) for i, score in enumerate(scores)]
            
            # Sort by score in descending order
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Limit to top_k results
            scored_results = scored_results[:top_k]
            
            # Return results in requested format
            return [{"index": idx, "score": float(score)} for idx, score in scored_results]
        
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.model:
            # Move model to CPU to free GPU memory
            if self.device != "cpu" and not self.device_map:
                self.model = self.model.to("cpu")
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Delete model and tokenizer
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            logger.info(f"{self.__class__.__name__} model resources released")