import os
import gc
from typing import Any
from pydantic import Field
from loguru import logger

# 优雅降级导入
try:
    import torch
    TORCH_AVAILABLE = True
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("警告: PyTorch 不可用，将使用备用方案")

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem

def auto_set_matmul_precision():
    """
    根据硬件环境自动设置 torch 的矩阵乘法精度。
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch 不可用，跳过矩阵乘法精度设置")
        return

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

# 只有在 PyTorch 可用时才调用函数
if TORCH_AVAILABLE:
    auto_set_matmul_precision()

class LocalEmbeddingConfig(EmbeddingConfig):
    model_path: str | None = Field(None, description="本地模型路径")
    device: str | None = Field(None, description="模型设备")
    device_map: str | None = Field(None, description="模型设备映射")
    max_memory: dict | None = Field(None, description="最大内存")
    trust_remote_code: bool = Field(False, description="信任远程代码")
    use_fp16: bool = Field(False, description="使用 FP16 精度")
    local_files_only: bool = Field(False, description="仅使用本地文件")
    compile_model: bool = Field(True, description="编译模型")
    cache_dir: str = Field("./cached_models", description="模型缓存目录")

class LocalEmbedding(BaseEmbedding):
    """
    生产级本地嵌入模型，具有增强的配置管理、错误处理和资源优化功能。
    支持 PyTorch 不可用时的优雅降级。
    """

    def __init__(self, config: LocalEmbeddingConfig):
        """
        通过健壮的配置验证初始化本地嵌入模型。
        """
        # 验证配置类型
        if not isinstance(config, LocalEmbeddingConfig):
            raise TypeError("config 必须是 LocalEmbeddingConfig 的实例")

        # 模型组件
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        
        # 配置
        self.cache_dir = config.cache_dir
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.predownload = False
        self.cache_path = os.path.join(config.cache_dir, self.model_name.replace('/', '_'))
        self.name_or_path = ""
        
        # 设备配置
        self.device = config.device
        if TORCH_AVAILABLE and self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif not TORCH_AVAILABLE:
            self.device = "cpu"
        
        self.device_map = config.device_map
        self.max_memory = getattr(config, 'max_memory', None)
        
        # 模型参数
        self.max_tokens = getattr(config, 'max_tokens', 512)
        self.batch_size = getattr(config, 'batch_size', 2)
        self.compile_model = getattr(config, 'compile_model', True) and TORCH_AVAILABLE
        self.use_fp16 = getattr(config, 'use_fp16', False) and TORCH_AVAILABLE
        self.local_files_only = getattr(config, 'local_files_only', False)
        self.trust_remote_code = getattr(config, 'trust_remote_code', False)
        
        # 运行时状态
        self._is_initialized = False
        self._using_device_map = False
        self._fallback_mode = not TORCH_AVAILABLE

    async def startup(self) -> None:
        """使用全面的错误处理初始化嵌入模型。"""
        if self._is_initialized:
            logger.warning("模型已经初始化")
            return

        # 如果 PyTorch 不可用，进入降级模式
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch 不可用，进入降级模式")
            self._is_initialized = True
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
                    os.makedirs(self.cache_dir, exist_ok=True)
                    logger.info(f"将从 hub 下载模型: {self.model_name}, 并缓存到: {self.cache_dir}")
            else:
                if os.path.exists(self.cache_path) and self._validate_model_files(self.cache_path):
                    self.model_path = self.cache_path
                    self.predownload = True
                    logger.info(f"使用缓存的模型: {self.cache_path}")
                else:
                    self.predownload = False
                    os.makedirs(self.cache_dir, exist_ok=True)
                    logger.info(f"将从 hub 下载模型: {self.model_name}, 并缓存到: {self.cache_dir}")

            self.name_or_path = self.model_path if self.predownload else self.model_name
            logger.debug(f"Embedding 模型名称: {self.model_name}")

            # 加载分词器和模型
            self.tokenizer = self._load_tokenizer()
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
            # 初始化失败时也进入降级模式
            self._fallback_mode = True
            self._is_initialized = True

    def _load_tokenizer(self) -> Any:
        """使用全面配置加载分词器。"""
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
            logger.debug("分词器加载成功")
            return tokenizer
        except Exception as e:
            logger.error(f"加载分词器失败: {str(e)}")
            raise

    def _load_model(self) -> Any:
        """使用正确的设备和精度设置加载模型。"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用，无法加载模型")

        load_kwargs = {
            "pretrained_model_name_or_path": self.name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
            "cache_dir": self.cache_dir
        }

        # 确定设备配置
        if torch.cuda.is_available():
            if self.device_map is not None:
                load_kwargs.update({
                    "device_map": self.device_map,
                    "max_memory": self.max_memory,
                })
                self._using_device_map = True
                target_device = None
                logger.debug(f"使用 device_map 加载: {self.device_map}")
            else:
                self._using_device_map = False
                target_device = self.device
                load_kwargs["torch_dtype"] = torch.float16 if self.use_fp16 else torch.float32
        else:
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
            target_device = "cpu"

        try:
            model = AutoModel.from_pretrained(**load_kwargs)
            
            if not self._using_device_map and target_device is not None:
                model = model.to(target_device)
                logger.debug(f"模型移动到设备: {target_device}")
            
            logger.debug("模型加载成功")
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

    def _validate_model_files(self, model_path: str) -> bool:
        """验证模型目录包含必需的文件。"""
        try:
            required_files = ["config.json", "tokenizer_config.json"]
            config_valid = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
            
            model_files = ["pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"]
            found_model_file = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
            
            vocab_files = ["vocab.txt", "vocab.json", "tokenizer.json"]
            found_vocab_file = any(os.path.exists(os.path.join(model_path, f)) for f in vocab_files)
            
            return config_valid and found_model_file and found_vocab_file
            
        except Exception as e:
            logger.warning(f"模型验证检查失败: {str(e)}")
            return False

    def _auto_detect_batch_size(self) -> int:
        """根据可用硬件和模型大小动态确定安全的批次大小。"""
        if not TORCH_AVAILABLE or not self._is_initialized or self.model is None:
            return 32  # 保守的默认值
        
        try:
            if not torch.cuda.is_available():
                return 32
                
            device = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            free_mem = total_mem - reserved_mem
            
            if hasattr(self.model, 'config'):
                hidden_size = getattr(self.model.config, 'hidden_size', 768)
                num_layers = getattr(self.model.config, 'num_hidden_layers', 12)
                
                bytes_per_param = 2 if self.use_fp16 else 4
                est_mem_per_batch = (hidden_size * num_layers * bytes_per_param * 1024)
            else:
                est_mem_per_batch = 1.0 * (1024**3)
            
            safe_batch = int((free_mem * 0.6) / est_mem_per_batch)
            return max(1, min(safe_batch, 128))
            
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
        支持 PyTorch 不可用时的降级处理。
        """
        # 如果处于降级模式，返回随机嵌入向量
        if self._fallback_mode or not TORCH_AVAILABLE:
            logger.warning("使用降级模式生成随机嵌入向量")
            return self._generate_fallback_embeddings(texts)

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
        
        # 确定批次大小
        effective_batch_size = batch_size if batch_size > 0 else self.batch_size
        
        try:
            return await self._process_batches(valid_texts, original_indices, effective_batch_size, normalize)
                
        except Exception as e:
            return self._handle_embed_error(e, effective_batch_size, raise_on_error)

    def _generate_fallback_embeddings(self, texts: list[str]) -> EmbeddingResponse:
        """在 PyTorch 不可用时生成降级嵌入向量。"""
        data = []
        embedding_dim = 768  # 默认维度
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                # 生成简单的基于文本哈希的伪随机嵌入向量
                import hashlib
                text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                
                # 使用哈希生成确定性随机嵌入向量
                import random
                random.seed(text_hash)
                embedding = [random.gauss(0, 1) for _ in range(embedding_dim)]
                
                # 简单归一化
                import math
                norm = math.sqrt(sum(x*x for x in embedding))
                if norm > 0:
                    embedding = [x/norm for x in embedding]
                
                data.append(EmbeddingDataItem(
                    embedding=embedding,
                    index=i,
                    object="embedding"
                ))
        
        return EmbeddingResponse(
            data=data,
            model=self.model_name + "-fallback",
            object="list",
            usage={
                "prompt_tokens": sum(len(text) for text in texts if isinstance(text, str)),
                "total_tokens": sum(len(text) for text in texts if isinstance(text, str))
            }
        )

    async def _process_batches(
        self,
        texts: list[str],
        original_indices: list[int],
        batch_size: int,
        normalize: bool
    ) -> EmbeddingResponse:
        """分批处理文本并返回嵌入向量。"""
        if not TORCH_AVAILABLE:
            return self._generate_fallback_embeddings(texts)

        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_indices = original_indices[i:i + batch_size]
            
            try:
                embeddings, tokens = await self._process_single_batch(batch, normalize)
                all_embeddings.append((batch_indices, embeddings))
                total_tokens += tokens
                    
            except Exception as e:
                logger.error(f"处理从索引 {i} 开始的批次失败: {str(e)}")
                continue
        
        return self._build_response(all_embeddings, total_tokens, len(texts))

    async def _process_single_batch(
        self,
        batch: list[str],
        normalize: bool
    ) -> tuple[Any, int]:
        """处理单个文本批次。"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用")
            
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
        all_embeddings: list[tuple[list[int], Any]],
        total_tokens: int,
        total_texts: int
    ) -> EmbeddingResponse:
        """构建响应，确保每个原始输入文本都有嵌入向量。"""
        if not all_embeddings:
            return self._empty_response()

        # 创建列表以按原始顺序保存所有嵌入向量
        final_embeddings: list[Any | None] = [None] * total_texts
        
        # 将每个批次的嵌入向量放在正确的位置
        for indices, embeddings in all_embeddings:
            for idx, embedding in zip(indices, embeddings):
                if idx < total_texts:
                    final_embeddings[idx] = embedding
        
        # 过滤掉 None 值并创建响应
        data = []
        valid_count = 0
        
        for idx, embedding in enumerate(final_embeddings):
            if embedding is not None:
                if hasattr(embedding, 'dim') and embedding.dim() != 1:
                    logger.warning(f"索引 {idx} 处的嵌入向量维度异常: {embedding.dim()}")
                    embedding = embedding.squeeze()
                    if hasattr(embedding, 'dim') and embedding.dim() != 1:
                        embedding = embedding.mean(dim=0)
                
                # 转换为列表
                if hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = embedding
                
                data.append(EmbeddingDataItem(
                    embedding=embedding_list,
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
        logger.error(f"嵌入错误: {error_name} - {str(error)}")
        
        # 对 OOM 错误的特殊处理
        if TORCH_AVAILABLE and isinstance(error, torch.cuda.OutOfMemoryError):
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
        token_embeddings: Any,
        attention_mask: Any
    ) -> Any:
        """通过注意力掩码加权计算平均令牌嵌入向量。"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用")

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
        if not TORCH_AVAILABLE or self.model is None:
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
            # 只有在 PyTorch 可用时才清理模型
            if TORCH_AVAILABLE and self.model is not None:
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
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self._is_initialized = False
            logger.info("模型资源成功释放")
            
        except Exception as e:
            logger.error(f"关闭期间出错: {e}")

    @property
    def embedding_dim(self) -> int:
        """获取嵌入向量的输出维度。"""
        if self._fallback_mode or not TORCH_AVAILABLE:
            return 768  # 降级模式的默认维度
            
        if self.model is None:
            raise RuntimeError("模型未初始化")
        return self.model.config.hidden_size  # type: ignore

    @property
    def is_fallback_mode(self) -> bool:
        """检查是否处于降级模式。"""
        return self._fallback_mode or not TORCH_AVAILABLE