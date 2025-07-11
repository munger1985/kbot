from typing import List, Optional, Any
import numpy as np
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from prometheus_client import Histogram, Counter, Gauge
from models.embedding.base import BaseEmbedding, LocalEmbeddingConfig

class LocalEmbedding(BaseEmbedding):
    """
    Production-grade local embedding model with explicit configuration management.
    """

    # Prometheus metrics
    LATENCY_HIST = Histogram(
        'local_embedding_latency_seconds',
        'Latency for local embedding requests',
        ['model_name']
    )
    
    ERROR_COUNTER = Counter(
        'local_embedding_errors_total',
        'Count of local embedding errors',
        ['model_name', 'error_type']
    )
    
    MEMORY_GAUGE = Gauge(
        'local_embedding_memory_usage_mb',
        'GPU memory usage in MB',
        ['device_id']
    )

    def __init__(self, config: LocalEmbeddingConfig):
        """
        Initialize local embedding model with explicit parameter unpacking.
        
        Args:
            config: Configuration containing:
                - model_name: HuggingFace model identifier (e.g. "BAAI/bge-small-en-v1.5")
                - model_path: Optional local path to model files
                - device: Target device (e.g., "cuda:0", "cpu")
                - device_map: For multi-GPU setups (e.g., "auto", "balanced")
                - max_tokens: Maximum input sequence length (default: 512)
                - compile_model: Whether to compile model with torch.compile() (PyTorch 2.0+)
                - use_fp16: Use half-precision inference (recommended for GPU)
                - local_files_only: Only use local model files (no internet download)
                - trust_remote_code: Trust custom model code from HuggingFace
                - max_memory: Dict of GPU memory limits (e.g. {"0": "24GB", "1": "24GB"})
        """
        # Model components
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[Any] = None
        
        # Configuration parameters
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.device = config.device
        self.device_map = config.device_map
        self.max_tokens = getattr(config, 'max_tokens', 512)
        self.compile_model = getattr(config, 'compile_model', False)
        self.use_fp16 = getattr(config, 'use_fp16', torch.cuda.is_available())
        self.local_files_only = getattr(config, 'local_files_only', False)
        self.trust_remote_code = getattr(config, 'trust_remote_code', False)
        self.max_memory = getattr(config, 'max_memory', None)
        
        # Runtime state
        self._batch_size = self._auto_detect_batch_size()
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize model with hardware-aware settings."""
        if self._is_initialized:
            return

        if self.model_path is None and self.local_files_only:
            raise ValueError("Local model path not specified.")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path or self.model_name,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
            model_max_length=self.max_tokens,
            padding_side='right',
            local_files_only=self.local_files_only
        )

        # Load model with optimized settings
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_path or self.model_name,
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

        self.model = AutoModel.from_pretrained(**load_kwargs)
        
        # 如果没有使用device_map，则使用.to()方法将模型移动到指定设备
        if self.device_map is None:
            self.model = self.model.to(target_device)
            logger.debug(f"模型已加载到设备: {target_device}")
        else:
            logger.debug(f"模型已加载到多设备: {self.device_map}")
            
        # 记录模型参数所在的设备
        sample_param = next(self.model.parameters())
        logger.debug(f"模型参数设备检查: {sample_param.device}")
            
        self.model.eval() # type: ignore

        # Model compilation (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile( # type: ignore
                self.model,
                mode='max-autotune' if torch.cuda.is_available() else None
            )

        self._is_initialized = True

    def _auto_detect_batch_size(self) -> int:
        """Dynamically determine safe batch size based on hardware."""
        if not torch.cuda.is_available():
            return 32  # Conservative CPU batch size
        
        # Memory-based calculation
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            available_mem = total_mem * 0.8  # Reserve 20% overhead
            
            # Heuristic memory estimation (GB per batch)
            mem_per_batch = {
                "small": 0.5,
                "base": 1.3,
                "large": 3.8
            }.get(self.model_name.split("-")[-1], 1.0)  # Default 1GB
            
            return min(
                int(available_mem / mem_per_batch),
                128  # Upper bound
            )
        except Exception:
            return 32  # Fallback value

    async def embed(
    self,
    texts: List[str],
    batch_size: Optional[int] = None,
    normalize: bool = True,
    raise_on_error: bool = True
) -> np.ndarray:
        """
        Generate embeddings with automatic batch processing.
        
        Args:
            texts: List of input texts
            batch_size: Override auto-detected batch size
            normalize: L2-normalize output embeddings
            raise_on_error: Whether to raise exceptions
            
        Returns:
            np.ndarray: Embedding matrix of shape (num_texts, embedding_dim)
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call startup() first.")
            
        # 使用推荐批次大小（除非显式指定）
        effective_batch_size = batch_size if batch_size is not None else self._batch_size
        all_embeddings = []
        
        try:
            with self.LATENCY_HIST.labels(model_name=self.model_name).time():
                # 分批处理文本
                for i in range(0, len(texts), effective_batch_size):
                    batch = texts[i:i + effective_batch_size]
                    
                    # Tokenize
                    encoded_input = self.tokenizer( # type: ignore
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_tokens,
                        return_tensors="pt"
                    )
                    
                    # 设备处理（兼容单设备/多设备）
                    if not hasattr(self.model, 'device_map'):  # 单设备情况
                        device = self.model.device  # type: ignore
                        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                        logger.info(f"批次 {i//effective_batch_size + 1}: 使用设备 {device}")
                        
                        # 如果是GPU，记录内存使用情况
                        if device.type == 'cuda':
                            gpu_id = device.index if device.index is not None else 0
                            gpu_name = torch.cuda.get_device_name(gpu_id)
                            current_mem = torch.cuda.memory_allocated(device) / (1024**3)
                            max_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
                            logger.debug(f"GPU: {gpu_name}, 内存使用: {current_mem:.2f}GB / {max_mem:.2f}GB (当前/峰值)")
                    else:
                        # 多设备情况
                        logger.info(f"批次 {i//effective_batch_size + 1}: 使用多设备配置 {self.model.device_map}")  # type: ignore
                    
                    # 推理
                    with torch.no_grad():
                        outputs = self.model(**encoded_input) # type: ignore
                    
                    # 池化
                    embeddings = self._mean_pooling(
                        outputs.last_hidden_state,
                        encoded_input['attention_mask']
                    )
                    
                    if normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        
                    all_embeddings.append(embeddings.cpu())
                    
                return np.vstack([e.numpy() for e in all_embeddings])
                
        except torch.cuda.OutOfMemoryError:
            # 动态调整批次大小
            new_batch_size = max(1, effective_batch_size // 2)
            self._batch_size = new_batch_size  # 持久化新大小
            
            self.ERROR_COUNTER.labels(
                model_name=self.model_name,
                error_type="out_of_memory"
            ).inc()
            
            if raise_on_error:
                raise RuntimeError(
                    f"CUDA OOM. Batch size auto-adjusted from {effective_batch_size}→{new_batch_size}\n"
                    f"Suggest: embed(texts, batch_size={new_batch_size})"
                )
            return np.array([])
            
        except Exception as e:
            self.ERROR_COUNTER.labels(
                model_name=self.model_name,
                error_type=str(type(e).__name__)
            ).inc()
            if raise_on_error:
                raise
            return np.array([])

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute mean token embeddings with attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    async def shutdown(self) -> None:
        """Clean up resources safely."""
        if self.model is not None:
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        torch.cuda.empty_cache()
        self._is_initialized = False

    @property
    def embedding_dim(self) -> int:
        """Get the output dimension of embeddings."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model.config.hidden_size # type: ignore

    @property
    def recommended_batch_size(self) -> int:
        """Get the currently recommended batch size."""
        return self._batch_size