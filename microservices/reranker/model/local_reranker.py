import os
from typing import Any
from pydantic import Field
from loguru import logger

# 优雅降级导入
try:
    import torch
    TORCH_AVAILABLE = True
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("警告: PyTorch 不可用，将使用备用方案")

from .base import BaseReranker, RerankerConfig

class LocalRerankerConfig(RerankerConfig):
    """本地 Reranker 模型配置"""
    model_name: str = Field(..., description="Reranker 模型名称")
    model_path: str | None = Field(None, description="模型文件的本地路径（可选）")
    device: str | None = Field("cuda:0", description="目标设备（如 'cuda:0', 'cpu'）")
    device_map: str | None = Field(None, description="多 GPU 设置（如 'auto', 'balanced'）")
    max_tokens: int | None = Field(512, description="最大输入序列长度")
    compile_model: bool = Field(True, description="是否使用 torch.compile() 编译模型（PyTorch 2.0+）")
    use_fp16: bool = Field(False, description="使用半精度推理（推荐用于 GPU）")
    local_files_only: bool = Field(False, description="仅使用本地模型文件（不下载）")
    cache_dir: str = Field("./cached_models", description="模型文件的本地缓存目录")
    trust_remote_code: bool = Field(False, description="信任来自 HuggingFace 的自定义模型代码")
    max_memory: dict[str, str] | None = Field(None, description="GPU 内存限制字典（如 {'0': '24GB', '1': '24GB'}）")
    batch_size: int = Field(16, description="批处理大小以避免内存溢出")

class LocalReranker(BaseReranker):
    """通用 Reranker 重排器基类，支持优雅降级"""

    def __init__(self, config: LocalRerankerConfig):
        # 模型组件
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.predownload = False
        self.cache_dir = config.cache_dir
        self.cache_path = os.path.join(config.cache_dir, self.model_name.replace('/', '_'))
        self.name_or_path = ""
        self.device = config.device
        self.device_map = config.device_map
        self.local_files_only = config.local_files_only
        self.trust_remote_code = config.trust_remote_code
        self.max_tokens = config.max_tokens
        self.compile_model = config.compile_model and TORCH_AVAILABLE  # 只有 PyTorch 可用时才编译
        self.use_fp16 = config.use_fp16 and TORCH_AVAILABLE  # 只有 PyTorch 可用时才使用 FP16
        self.max_memory = config.max_memory
        self.batch_size = config.batch_size

        # 运行时状态
        self._is_initialized = False
        self._fallback_mode = not TORCH_AVAILABLE  # 降级模式标志
            
        logger.info(f"正在初始化 {self.__class__.__name__}，模型: {self.model_name}")
    
    def _validate_model_files(self, model_path: str) -> bool:
        """验证模型目录包含必需的文件"""
        try:
            # 使用集合操作提高可读性和性能
            required_files = {"config.json", "tokenizer_config.json"}
            model_files = {"pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"}
            vocab_files = {"vocab.txt", "vocab.json", "tokenizer.json"}
            
            existing_files = set(os.listdir(model_path))
            
            config_valid = required_files.issubset(existing_files)
            found_model_file = bool(model_files & existing_files)
            found_vocab_file = bool(vocab_files & existing_files)
            
            return config_valid and found_model_file and found_vocab_file
            
        except Exception as e:
            logger.warning(f"模型验证检查失败: {str(e)}")
            return False

    def _determine_model_source(self) -> None:
        """确定模型来源：本地路径、缓存或下载"""
        # 优先检查显式指定的模型路径
        if self.model_path and os.path.exists(self.model_path):
            if self._validate_model_files(self.model_path):
                self.predownload = True
                logger.info(f"使用预下载的模型: {self.model_path}")
                return
        
        # 检查缓存
        if os.path.exists(self.cache_path) and self._validate_model_files(self.cache_path):
            self.model_path = self.cache_path
            self.predownload = True
            logger.info(f"使用缓存的模型: {self.cache_path}")
            return
        
        # 需要下载
        self.predownload = False
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"将从 hub 下载模型: {self.model_name}, 并缓存到: {self.cache_dir}")
    
    def _setup_device_config(self) -> tuple[str, dict]:
        """设置设备配置并返回目标设备和加载参数"""
        if not TORCH_AVAILABLE:
            return "cpu", {}
            
        target_device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        load_kwargs = {
            "pretrained_model_name_or_path": self.name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
            "cache_dir": self.cache_dir
        }
        
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision('high')
            except Exception as e:
                logger.warning(f"设置矩阵乘法精度失败: {e}")
            
            if self.device_map:
                load_kwargs.update({
                    "device_map": self.device_map,
                    "max_memory": self.max_memory,
                })
            else:
                # 验证设备字符串
                if not target_device.startswith(('cuda:', 'cpu')):
                    target_device = "cuda:0"
            
            # 精度控制
            load_kwargs["torch_dtype"] = torch.float16 if self.use_fp16 else torch.float32
        else:
            load_kwargs.update({
                "device_map": "cpu",
                "torch_dtype": torch.float32
            })
            target_device = "cpu"
            
        return target_device, load_kwargs
    
    async def startup(self) -> None:
        """初始化 reranker 模型"""
        if self._is_initialized:
            return

        # 如果 PyTorch 不可用，进入降级模式
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch 不可用，LocalReranker 进入降级模式")
            self._is_initialized = True
            self._fallback_mode = True
            return

        if self.model_path is None and self.local_files_only:
            raise ValueError("未指定本地模型路径")
        
        self._determine_model_source()
        self.name_or_path = self.model_path if self.predownload else self.model_name

        logger.debug(f"Reranker 模型名称: {self.model_name}")
            
        try:
            # 加载分词器
            self.tokenizer = self._load_tokenizer()
                
            # 设置设备配置
            target_device, load_kwargs = self._setup_device_config()

            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(**load_kwargs)
            except Exception as e:
                logger.warning(f"首次加载失败: {str(e)}，尝试不使用低CPU内存模式")
                load_kwargs["low_cpu_mem_usage"] = False
                self.model = AutoModelForSequenceClassification.from_pretrained(**load_kwargs)

            # 单设备模式：手动移动模型
            if self.device_map is None:
                self.model = self.model.to(target_device)
                self.device = target_device
                logger.debug(f"Reranker 模型已加载到设备: {target_device}")
            else:
                logger.debug(f"Reranker 模型已使用 device_map 加载: {self.device_map}")
            
            # 记录模型参数设备
            sample_param = next(self.model.parameters())
            logger.debug(f"Reranker 模型参数位于设备: {sample_param.device}")
            
            self.model.eval()

            # 模型编译
            if self.compile_model and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(
                        self.model,
                        mode='max-autotune' if torch.cuda.is_available() else None,
                        fullgraph=False  # 允许部分图编译以提高兼容性
                    )
                    logger.info("模型编译成功")
                except Exception as e:
                    logger.warning(f"模型编译失败: {str(e)}，将继续使用未编译的模型")

            self._is_initialized = True
            self._fallback_mode = False
            logger.info(f"Reranker 模型 {self.model_name} 初始化成功")
            
        except Exception as e:
            logger.error(f"Reranker 模型初始化失败: {e}")
            # 初始化失败时进入降级模式
            self._is_initialized = True
            self._fallback_mode = True
            logger.warning("LocalReranker 进入降级模式")
    
    def _load_tokenizer(self) -> Any:
        """使用全面配置加载分词器"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用，无法加载分词器")
            
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.name_or_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
                model_max_length=self.max_tokens,
                padding_side='right',
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir
            )
            
            # 确保分词器有填充token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
                if tokenizer.pad_token == '[PAD]':
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    
            logger.debug("分词器加载成功")
            return tokenizer
        except Exception as e:
            logger.error(f"加载分词器失败: {str(e)}")
            raise

    def _compute_scores_fallback(self, query: str, documents: list[str]) -> list[float]:
        """降级模式下的分数计算"""
        scores = []
        query_words = set(query.lower().split())
        
        for doc in documents:
            doc_words = set(doc.lower().split())
            
            # 计算 Jaccard 相似度
            if len(query_words) == 0 or len(doc_words) == 0:
                scores.append(0.0)
                continue
                
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # 添加基于长度的权重
            length_penalty = min(len(doc) / 1000, 1.0)
            
            score = jaccard_similarity * 0.7 + length_penalty * 0.3
            scores.append(score)
        
        return scores

    async def _process_batch(self, query: str, batch_documents: list[str]) -> list[float]:
        """处理一个批次的文档，返回分数列表"""
        # 如果处于降级模式，使用备用方案
        if self._fallback_mode or not TORCH_AVAILABLE:
            return self._compute_scores_fallback(query, batch_documents)
            
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未初始化，请先调用 startup() 方法")
        
        pairs = [(query, doc) for doc in batch_documents]
        
        # 使用 inference_mode 替代 no_grad，性能更好
        inference_mode = torch.inference_mode if TORCH_AVAILABLE else (lambda: lambda f: f)()
        
        with inference_mode():
            try:
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_tokens,
                    return_attention_mask=True
                )
                
                # 设备转移
                if self.device_map is None:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 确保attention mask存在
                if 'attention_mask' not in inputs:
                    inputs['attention_mask'] = (inputs['input_ids'] != self.tokenizer.pad_token_id).long()
                
                # 获取分数
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 统一分数处理逻辑
                if logits.dim() == 2:
                    if logits.size(1) == 1:  # 二元分类或回归分数
                        scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
                    else:  # 多分类
                        scores = torch.softmax(logits, dim=-1)[:, -1].cpu().tolist()
                else:
                    scores = torch.sigmoid(logits).cpu().tolist()

            except RuntimeError as e:
                if "attention mask" in str(e).lower():
                    logger.warning(f"Attention mask 错误: {e}，尝试手动创建attention mask")
                    return await self._process_batch_with_manual_mask(query, batch_documents)
                else:
                    logger.error(f"处理批次时出错: {e}，使用降级模式")
                    return self._compute_scores_fallback(query, batch_documents)

        return scores if isinstance(scores, list) else [scores]
    
    async def _process_batch_with_manual_mask(self, query: str, batch_documents: list[str]) -> list[float]:
        """手动创建attention mask的处理批次方法"""
        if not TORCH_AVAILABLE:
            return self._compute_scores_fallback(query, batch_documents)
            
        pairs = [(query, doc) for doc in batch_documents]
        
        with torch.inference_mode():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_tokens,
                return_attention_mask=False
            )
            
            if self.device_map is None:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 手动创建attention mask
            inputs['attention_mask'] = (inputs['input_ids'] != self.tokenizer.pad_token_id).long()
            
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().tolist()
            
        return scores if isinstance(scores, list) else [scores]
    
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        根据与查询的相关性对文档进行重排序
        """
        if not self._is_initialized:
            raise RuntimeError("模型未初始化，请先调用 startup() 方法")
        
        if not documents:
            return []
        
        top_k = len(documents) if top_k is None else min(top_k, len(documents))
        
        try:
            all_scores = []
            
            # 如果处于降级模式，直接计算所有分数
            if self._fallback_mode:
                logger.warning(f"使用降级模式对 {len(documents)} 个文档进行重排序")
                all_scores = self._compute_scores_fallback(query, documents)
            else:
                # 分批处理文档
                for i in range(0, len(documents), self.batch_size):
                    batch_docs = documents[i:i + self.batch_size]
                    batch_scores = await self._process_batch(query, batch_docs)
                    all_scores.extend(batch_scores)
                    
                    # 可选：记录内存使用情况
                    if logger.level("DEBUG").no <= logger._core.min_level and TORCH_AVAILABLE and torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                        logger.debug(f"批次 {i//self.batch_size + 1}: GPU 内存: {allocated:.2f}MB")
            
            # 使用 enumerate 和 zip 避免重复索引操作
            scored_results = list(enumerate(all_scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            mode_info = "降级模式" if self._fallback_mode else "正常模式"
            logger.info(f"重排序完成({mode_info})，返回前 {top_k} 个结果")
            
            return [{"index": idx, "score": float(score)} for idx, score in scored_results[:top_k]]
        
        except Exception as e:
            logger.exception(f"重排序过程中发生错误: {str(e)}")
            # 在错误时返回基于索引的默认排序
            return [{"index": i, "score": 1.0 - (i * 0.01)} for i in range(min(top_k, len(documents)))]

    async def shutdown(self) -> None:
        """清理资源"""
        if not self._is_initialized:
            return
            
        if TORCH_AVAILABLE and self.model is not None:
            # 单设备模式：移动到 CPU
            if self.device != "cpu" and not self.device_map:
                self.model = self.model.to("cpu")
            
            # 清除 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 显式删除
            del self.model
            self.model = None
            
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            
        self._is_initialized = False
        logger.info(f"{self.__class__.__name__} 模型资源已释放")

    @property
    def is_fallback_mode(self) -> bool:
        """检查是否处于降级模式"""
        return self._fallback_mode or not TORCH_AVAILABLE

    @property
    def supports_torch_compile(self) -> bool:
        """检查是否支持 Torch 编译"""
        return self.compile_model and TORCH_AVAILABLE and hasattr(torch, 'compile')