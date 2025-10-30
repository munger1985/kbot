import os
import gc
import torch
from typing import Any
from pydantic import Field
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from prometheus_client import Histogram, Counter, Gauge
from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem

def auto_set_matmul_precision():
    """
    根据硬件环境自动设置 torch 的矩阵乘法精度。
    """
    # 检查是否有可用的 CUDA GPU
    if not torch.cuda.is_available():
        logger.warning("未检测到 CUDA GPU，保持默认的矩阵乘法精度设置。")
        return
    
    # 获取当前 GPU 的设备属性
    current_device = torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(current_device)
    gpu_name = gpu_props.name
    major, minor = gpu_props.major, gpu_props.minor  # 计算能力主次版本号

    # 判断是否支持 TF32（通常为 Ampere 架构及以上，计算能力 >= 8.0）
    # 计算能力版本参考: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    if major >= 8: 
        # Ampere (e.g., A100, RTX 30xx) 或更新架构
        try:
            torch.set_float32_matmul_precision('high') # 启用 TF32
            logger.info(f"检测到支持 TF32 的 GPU: {gpu_name} (计算能力 {major}.{minor})，已设置 float32_matmul_precision 为 'high'。")
        except Exception as e:
            logger.warning(f"检测到支持 TF32 的 GPU: {gpu_name}，但设置 'high' 时出现异常: {e}，使用默认设置。")
    else:
        # 不支持 TF32 的 GPU（如 Turing, Volta, Pascal 等）
        logger.warning(f"检测到 GPU: {gpu_name} (计算能力 {major}.{minor}) 不支持 TF32，保持默认的矩阵乘法精度设置。")

# 在代码开头调用函数
auto_set_matmul_precision()

class LocalEmbeddingConfig(EmbeddingConfig):
    model_path: str | None = Field(None, description="本地模型路径")
    device: str | None = Field(None, description="模型设备")
    device_map: str | None = Field(None, description="模型设备映射")
    max_memory: dict | None = Field(None, description="最大内存")
    trust_remote_code: bool = Field(False, description="信任远程代码")
    use_fp16: bool = Field(False, description="使用 FP16 精度")
    local_files_only: bool = Field(False, description="仅使用本地文件")
    compile_model: bool = Field(True, description="编译模型") # 当使用 PyTorch 2.0+ 时为 True，否则为 False
    cache_dir: str = Field("./cached_models", description="模型缓存目录") # 从 nacos manager 中读取 cache_dir 配置，用于缓存模型


class LocalEmbedding(BaseEmbedding):
    """
    生产级本地嵌入模型，具有增强的配置管理、错误处理和资源优化功能。
    """

    # Prometheus 指标
    LATENCY_HIST = Histogram(
        'local_embedding_latency_seconds',
        '本地嵌入请求的延迟时间',
        ['model_name']
    )
    
    ERROR_COUNTER = Counter(
        'local_embedding_errors_total',
        '本地嵌入错误计数',
        ['model_name', 'error_type']
    )
    
    MEMORY_GAUGE = Gauge(
        'local_embedding_memory_usage_mb',
        'GPU 内存使用量（MB）',
        ['device_id']
    )
    
    REQUEST_SIZE_GAUGE = Gauge(
        'local_embedding_request_size',
        '嵌入请求的字符大小',
        ['model_name']
    )

    def __init__(self, config: LocalEmbeddingConfig):
        """
        通过健壮的配置验证初始化本地嵌入模型。
        
        参数:
            config: 包含模型和设备设置的配置对象
        
        异常:
            TypeError: 如果 config 不是 LocalEmbeddingConfig 类型
            ValueError: 如果必需的设置缺失或无效
        """
        # 验证配置类型
        if not isinstance(config, LocalEmbeddingConfig):
            raise TypeError("config 必须是 LocalEmbeddingConfig 的实例")

        # 模型组件
        self.model: torch.nn.Module | None = None
        self.tokenizer: Any | None = None
        
        # 带验证的配置
        self.cache_dir = config.cache_dir
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.predownload = False
        self.cache_path = os.path.join(config.cache_dir, self.model_name.replace('/', '_'))
        self.name_or_path = ""
        
        # 设备配置 - 优先尊重用户显式设置
        self.device = config.device
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.device_map = config.device_map
        self.max_memory = getattr(config, 'max_memory', None)
        
        # 模型参数（带默认值）
        self.max_tokens = getattr(config, 'max_tokens', 512)
        self.batch_size = getattr(config, 'batch_size', 2)
        self.compile_model = getattr(config, 'compile_model', True)
        self.use_fp16 = getattr(config, 'use_fp16', False)
        self.local_files_only = getattr(config, 'local_files_only', False)
        self.trust_remote_code = getattr(config, 'trust_remote_code', False)
        
        # 运行时状态
        self._is_initialized = False
        self._using_device_map = False  # 跟踪是否使用 device_map 加载

    async def startup(self) -> None:
        """使用全面的错误处理初始化嵌入模型。"""
        if self._is_initialized:
            logger.warning("模型已经初始化")
            return

        try:
            # 验证模型路径或缓存
            if self.model_path is None and self.local_files_only:
                raise ValueError("当 local_files_only=True 时未指定本地模型路径")

            # 检查是否有有效的本地模型路径或需要下载
            if self.model_path is not None and os.path.exists(self.model_path):
                if self._validate_model_files(self.model_path):
                    self.predownload = True
                    logger.info(f"使用预下载的模型: {self.model_path}")
                else:
                    logger.info(f"模型路径 {self.model_path} 存在但包含无效的模型文件")
                    self.predownload = False
                    # 确保缓存目录存在
                    os.makedirs(self.cache_dir, exist_ok=True)
                    logger.info(f"将从 hub 下载模型: {self.model_name}, 并缓存到: {self.cache_dir}")
            else:
                # 检查缓存或从 hub 下载
                if os.path.exists(self.cache_path) and self._validate_model_files(self.cache_path):
                    self.model_path = self.cache_path
                    self.predownload = True
                    logger.info(f"使用缓存的模型: {self.cache_path}")
                else:
                    self.predownload = False
                    # 确保缓存目录存在
                    os.makedirs(self.cache_dir, exist_ok=True)
                    logger.info(f"将从 hub 下载模型: {self.model_name}, 并缓存到: {self.cache_dir}")

            self.name_or_path = self.model_path if self.predownload else self.model_name

            logger.debug(f"Embedding 模型名称: {self.model_name}")

            # 加载分词器（带错误处理）
            self.tokenizer = self._load_tokenizer()
            
            # 使用优化设置加载模型
            self.model = self._load_model()
            
            # 根据实际模型大小和可用资源更新批次大小
            self._batch_size = self._auto_detect_batch_size()
            logger.info(f"自动检测的批次大小: {self._batch_size}")
            
            # 模型优化
            self._optimize_model()
            
            self._is_initialized = True
            logger.info(f"嵌入模型 {self.model_name} 成功初始化在设备: {self.device}")
            
        except Exception as e:
            self._is_initialized = False
            logger.exception(f"初始化模型 {self.model_name} 失败: {str(e)}")
            raise RuntimeError(f"模型初始化失败: {str(e)}")

    def _load_tokenizer(self) -> Any:
        """使用全面配置加载分词器。"""
        try:
            
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.name_or_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
                model_max_length=self.max_tokens,
                padding_side='right',
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir  # 使用配置的缓存目录
            )
            logger.debug("分词器加载成功")
            return tokenizer
        except Exception as e:
            logger.error(f"加载分词器失败: {str(e)}")
            raise

    def _load_model(self) -> torch.nn.Module:
        """使用正确的设备和精度设置加载模型。"""
        load_kwargs = {
            "pretrained_model_name_or_path": self.name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
            "cache_dir": self.cache_dir  # 使用配置的缓存目录
        }

        # 确定设备配置
        if torch.cuda.is_available():
            if self.device_map is not None:
                # 使用 device_map 进行多 GPU 加载
                load_kwargs.update({
                    "device_map": self.device_map,
                    "max_memory": self.max_memory,
                })
                self._using_device_map = True
                target_device = None
                logger.debug(f"使用 device_map 加载: {self.device_map}")
            else:
                # 单设备加载
                self._using_device_map = False
                target_device = self.device
                load_kwargs["torch_dtype"] = torch.float16 if self.use_fp16 else torch.float32
        else:
            # 如果没有 GPU，则使用 CPU
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
            target_device = "cpu"

        try:
            model = AutoModel.from_pretrained(**load_kwargs)
            
            # 如果不使用 device_map，则移动到目标设备
            if not self._using_device_map and target_device is not None:
                model = model.to(target_device)
                logger.debug(f"模型移动到设备: {target_device}")
            
            logger.debug("模型加载成功")
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

    def _validate_model_files(self, model_path: str) -> bool:
        """
        验证模型目录包含必需的文件。
        这是尽力而为的检查；实际加载可能仍然失败。
        """
        try:
            required_files = ["config.json", "tokenizer_config.json"]
            
            # 检查至少一个模型权重文件
            model_files = ["pytorch_model.bin", "model.safetensors*", "*.pt"]
            found_model_file = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
            
            # 检查至少一个词汇文件
            vocab_files = ["vocab.txt", "vocab.json", "tokenizer.json"]
            found_vocab_file = any(os.path.exists(os.path.join(model_path, f)) for f in vocab_files)
            
            # 检查必需的配置文件
            config_valid = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
            
            return config_valid and found_model_file and found_vocab_file
            
        except Exception as e:
            logger.warning(f"模型验证检查失败: {str(e)}")
            return False

    def _auto_detect_batch_size(self) -> int:
        """根据可用硬件和模型大小动态确定安全的批次大小。"""
        if not torch.cuda.is_available() or not self._is_initialized or self.model is None:
            return 32  # 保守的默认值
        
        try:
            # 获取当前 GPU 内存状态
            device = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            free_mem = total_mem - reserved_mem
            
            # 基于模型配置估算内存需求
            if hasattr(self.model, 'config'):
                hidden_size = getattr(self.model.config, 'hidden_size', 768)
                num_layers = getattr(self.model.config, 'num_hidden_layers', 12)
                
                # 粗略估算每个批次项的内存
                # 这是一个简化的计算，可能需要针对特定架构进行调整
                bytes_per_param = 2 if self.use_fp16 else 4
                est_mem_per_batch = (hidden_size * num_layers * bytes_per_param * 1024)  # 1024 序列长度因子
            else:
                # 备用估算
                est_mem_per_batch = 1.0 * (1024**3)  # 每个批次 1.0GB
            
            # 计算安全的批次大小（使用 60% 的可用内存以确保安全）
            safe_batch = int((free_mem * 0.6) / est_mem_per_batch)
            return max(1, min(safe_batch, 128))  # 限制在 1 到 128 之间
            
        except Exception as e:
            logger.warning(f"批次大小检测失败: {e}, 使用备用值 32")
            return 32

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 0,
        normalize: bool = True,
        raise_on_error: bool = True,
        **kwargs
    ) -> EmbeddingResponse:
        """
        使用自动批处理和全面监控生成嵌入向量。
        """
        # 验证和清理输入文本
        valid_texts = []
        original_indices = []
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                original_indices.append(i)
            else:
                logger.warning(f"跳过索引 {i} 处的无效或空文本")
        
        if not valid_texts:
            logger.warning("没有提供有效的文本用于嵌入")
            return self._empty_response()
        
        # 记录请求大小用于监控
        total_chars = sum(len(text) for text in valid_texts)
        self.REQUEST_SIZE_GAUGE.labels(model_name=self.model_name).set(total_chars)
        
        # 确定批次大小
        effective_batch_size = batch_size if batch_size > 0 else self.batch_size
        
        try:
            with self.LATENCY_HIST.labels(model_name=self.model_name).time():
                return await self._process_batches(valid_texts, original_indices, effective_batch_size, normalize)
                
        except Exception as e:
            return self._handle_embed_error(e, effective_batch_size, raise_on_error)

    async def _process_batches(
        self,
        texts: list[str],
        original_indices: list[int],
        batch_size: int,
        normalize: bool
    ) -> EmbeddingResponse:
        """分批处理文本并返回嵌入向量。"""
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_indices = original_indices[i:i + batch_size]
            
            try:
                embeddings, tokens = await self._process_single_batch(batch, normalize)
                all_embeddings.append((batch_indices, embeddings))
                total_tokens += tokens
                
                # 更新内存指标
                if torch.cuda.is_available():
                    # 使用当前 CUDA 设备，不需要传递 device 参数
                    mem_used = torch.cuda.memory_allocated() / (1024**2)  # MB
                    device_str = f"cuda:{torch.cuda.current_device()}"
                    self.MEMORY_GAUGE.labels(device_id=device_str).set(mem_used)
                    
            except Exception as e:
                logger.error(f"处理从索引 {i} 开始的批次失败: {str(e)}")
                # 继续处理剩余批次但记录错误
                continue
        
        return self._build_response(all_embeddings, total_tokens, len(texts))

    async def _process_single_batch(
        self,
        batch: list[str],
        normalize: bool
    ) -> tuple[torch.Tensor, int]:
        """处理单个文本批次。"""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("模型和分词器必须已初始化")
        
        # 分词
        encoded_input = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )
        
        # 跳过空输入
        if encoded_input['input_ids'].numel() == 0:
            return torch.empty((0, self.embedding_dim)), 0
        
        # 如果不使用 device_map，则将输入移动到正确的设备
        if not self._using_device_map and hasattr(self.model, 'device'):
            device = self.model.device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        
        # 池化
        embeddings = self._mean_pooling(
            outputs.last_hidden_state,
            encoded_input['attention_mask']
        )
        
        # 如果需要则归一化
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 计算此批次中的令牌数
        tokens = encoded_input['input_ids'].numel()
        
        return embeddings.cpu(), tokens

    def _empty_response(self) -> EmbeddingResponse:
        """返回具有正确结构的空响应。"""
        return EmbeddingResponse(
            data=[],
            model=self.model_name,
            object="list",
            usage={"prompt_tokens": 0, "total_tokens": 0}
        )
    
    def _build_response(
        self,
        all_embeddings: list[tuple[list[int], torch.Tensor]],
        total_tokens: int,
        total_texts: int
    ) -> EmbeddingResponse:
        """构建响应，确保每个原始输入文本都有嵌入向量。"""
        # 为所有文本初始化空数组
        if not all_embeddings:
            return self._empty_response()

        # 创建列表以按原始顺序保存所有嵌入向量
        final_embeddings: list[torch.Tensor | None] = [None] * total_texts
        
        # 将每个批次的嵌入向量放在正确的位置
        for indices, embeddings in all_embeddings:
            for idx, embedding in zip(indices, embeddings):
                if idx < total_texts:
                    final_embeddings[idx] = embedding
        
        # 过滤掉 None 值（失败的批次）并创建响应
        data = []
        valid_count = 0
        
        for idx, embedding in enumerate(final_embeddings):
            if embedding is not None:
                # 确保嵌入向量是 1D 向量
                if embedding.dim() != 1:
                    logger.warning(f"索引 {idx} 处的嵌入向量维度异常: {embedding.dim()}")
                    embedding = embedding.squeeze()
                    if embedding.dim() != 1:
                        # 如果仍然不是 1D，使用平均值或跳过
                        embedding = embedding.mean(dim=0)
                
                data.append(EmbeddingDataItem(
                    embedding=embedding.tolist(),
                    index=idx,
                    object="embedding"
                ))
                valid_count += 1
        
        logger.info(f"成功处理 {valid_count}/{total_texts} 个文本")
        
        return EmbeddingResponse(
            data=data,
            model=self.model_name,
            object="list",
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )

    def _handle_embed_error(
        self,
        error: Exception,
        batch_size: int,
        raise_on_error: bool
    ) -> EmbeddingResponse:
        """处理嵌入生成期间的错误。"""
        error_name = type(error).__name__
        self.ERROR_COUNTER.labels(
            model_name=self.model_name,
            error_type=error_name
        ).inc()
        
        logger.error(f"嵌入错误: {error_name} - {str(error)}")
        
        # 对 OOM 错误的特殊处理
        if isinstance(error, torch.cuda.OutOfMemoryError):
            new_batch_size = max(1, batch_size // 2)
            self._batch_size = new_batch_size
            logger.warning(
                f"CUDA 内存不足。批次大小自动从 {batch_size}→{new_batch_size} 调整"
            )
            
            if raise_on_error:
                raise RuntimeError(
                    f"CUDA 内存不足。建议 batch_size={new_batch_size}"
                ) from error
        
        elif raise_on_error:
            raise error
            
        return self._empty_response()

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        通过注意力掩码加权计算平均令牌嵌入向量。
        返回形状为 (batch_size, hidden_size) 的池化嵌入向量
        """
        # 输入验证
        if token_embeddings.dim() != 3:
            raise ValueError(f"令牌嵌入向量必须是 3D，得到 {token_embeddings.dim()}D")
        if attention_mask.dim() != 2:
            raise ValueError(f"注意力掩码必须是 2D，得到 {attention_mask.dim()}D")
        
        # 确保相同设备
        if token_embeddings.device != attention_mask.device:
            attention_mask = attention_mask.to(token_embeddings.device)
            
        # 处理空输入
        if token_embeddings.numel() == 0:
            return torch.zeros((0, token_embeddings.size(-1)), 
                             device=token_embeddings.device)

        # 扩展掩码并计算加权平均值
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        pooled = sum_embeddings / sum_mask
        
        # 确保正确的输出形状
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
            
        return pooled

    def _optimize_model(self) -> None:
        """应用模型优化，如编译和评估模式。"""
        if self.model is None:
            return

        self.model.eval()

        # 模型编译 (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile( # type: ignore
                    self.model,
                    mode='max-autotune' if torch.cuda.is_available() else None
                )
                logger.debug("模型编译成功")
            except Exception as e:
                logger.warning(f"模型编译失败: {e}")

    async def shutdown(self) -> None:
        """安全且彻底地清理资源。"""
        if not self._is_initialized:
            return
            
        try:
            # 将模型移动到 CPU 以释放 GPU 内存
            if self.model is not None:
                if hasattr(self.model, "device") and str(self.model.device) != "cpu":
                    self.model.to("cpu")
                
                del self.model
                self.model = None
            
            # 清理分词器
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # 强制垃圾回收
            gc.collect()
            
            # 如果可用则清空 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self._is_initialized = False
            logger.info("模型资源成功释放")
            
        except Exception as e:
            logger.error(f"关闭期间出错: {e}")
            raise

    @property
    def embedding_dim(self) -> int:
        """获取嵌入向量的输出维度。"""
        if self.model is None:
            raise RuntimeError("模型未初始化")
        return self.model.config.hidden_size  # type: ignore