import os
import torch
import asyncio
from typing import Any
from pydantic import Field
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """通用 Reranker 重排器基类"""

    def __init__(self, config: LocalRerankerConfig):
        """
        初始化通用 Transformer 重排器
        
        Args:
            config: 模型配置
        """
        # 模型组件
        self.model: torch.nn.Module | None = None
        self.tokenizer: Any | None = None
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.predownload = False  # 是否为本地预下载模型
        self.cache_path = os.path.join(config.cache_dir, self.model_name) # 模型缓存路径
        self.name_or_path = ""
        self.device = config.device
        self.device_map = config.device_map
        self.local_files_only = getattr(config, 'local_files_only', False)
        self.trust_remote_code = getattr(config, 'trust_remote_code', False)
        self.max_tokens = getattr(config, 'max_tokens', 512)  # 默认值调整为512以减少显存
        self.compile_model = getattr(config, 'compile_model', True)
        self.use_fp16 = getattr(config, 'use_fp16', False)
        self.max_memory = getattr(config, 'max_memory', None)
        self.batch_size = getattr(config, 'batch_size', 16)  # 批处理大小

        # 运行时状态
        self._is_initialized = False
            
        logger.info(f"正在初始化 {self.__class__.__name__}，模型: {self.model_name}")
    
    def _validate_reranker_model(self, model_path: str) -> bool:
        """检查模型目录是否包含必要文件"""
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
        """将模型保存到缓存目录"""
        self.model.save_pretrained(self.cache_path) # type: ignore
        self.tokenizer.save_pretrained(self.cache_path) # type: ignore
    
    async def startup(self) -> None:
        """初始化 reranker 模型"""
        if self._is_initialized:
            return

        if self.model_path is None and self.local_files_only:
            raise ValueError("未指定本地模型路径")
        
        if self.model_path is not None:
            valid_path = self._validate_reranker_model(self.model_path)
            if valid_path:
                self.predownload = True
            else:
                valid_cache = self._validate_reranker_model(self.cache_path)
                if valid_cache:
                    logger.info(f"使用缓存的 reranker 模型: {self.cache_path}")
                    self.model_path = os.path.abspath(self.cache_path)
                    self.predownload = True
                else:
                    self.predownload = False

        if self.predownload:
            self.name_or_path = self.model_path
        else:
            self.name_or_path = self.model_name

        logger.debug(f"Reranker 模型名称: {self.model_name}, 路径: {self.model_path}")
            
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = self.name_or_path,
            trust_remote_code = self.trust_remote_code,
            use_fast = True,  # 使用快速分词器减少内存消耗
            model_max_length = self.max_tokens,
            padding_side = 'right',
            local_files_only = self.local_files_only
        )
            
        # 使用优化设置加载模型
        load_kwargs = {
            "pretrained_model_name_or_path": self.name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
        }
            
        # 设备配置
        if torch.cuda.is_available():
            if self.device_map is not None:  # 多 GPU
                load_kwargs.update({
                    "device_map": self.device_map,
                    "max_memory": self.max_memory,
                })
            else:  # 单 GPU
                target_device = self.device or "cuda:0"
                
            # 精度控制 - 使用半精度显著减少显存占用
            load_kwargs["torch_dtype"] = torch.float16 if self.use_fp16 else torch.float32
        else:  # CPU 回退
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
            target_device = "cpu"

        self.model = AutoModelForSequenceClassification.from_pretrained(**load_kwargs)
        
        # 如果是首次使用从 HuggingFace 下载的模型，则将模型从默认缓存路径保存到本地
        if self.predownload is not True:
            try:
                self._cache_model()
                logger.debug(f"Reranker 模型 {self.model_name} 已下载到本地缓存: {self.cache_path}")
            except Exception as e:
                logger.error(f"保存 reranker 模型到本地缓存时出错: {e}")

        # 如果没有使用 device_map，则使用 .to() 方法将模型移动到指定设备
        if self.device_map is None:
            self.model = self.model.to(target_device) # type: ignore
            # 确保 self.device 与实际使用的设备一致
            self.device = target_device
            logger.debug(f"Reranker 模型已加载到设备: {target_device}")
        else:
            logger.debug(f"Reranker 模型已使用 device_map 加载: {self.device_map}")
        
        # 记录模型参数所在的设备
        sample_param = next(self.model.parameters()) # type: ignore
        logger.debug(f"Reranker 模型参数位于设备: {sample_param.device}")
        
        self.model.eval() # type: ignore

        # 模型编译 (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile( # type: ignore
                self.model,
                mode='max-autotune' if torch.cuda.is_available() else None
            )

        self._is_initialized = True
        logger.info(f"Reranker 模型 {self.model_name} 初始化成功")
    
    async def _process_batch(self, query: str, batch_documents: list[str]) -> list[float]:
        """处理一个批次的文档，返回分数列表"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未初始化，请先调用 startup() 方法")
        
        pairs = [(query, doc) for doc in batch_documents]
        
        # 分词处理
        with torch.no_grad():  # 禁用梯度计算以减少显存
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_tokens  # 控制序列长度以减少显存
            )
            
            # 根据模型配置方式处理设备分配
            if self.device_map is None:
                # 单设备模式：将输入移动到模型所在的设备
                inputs = inputs.to(self.device)
            
            # 获取分数
            logits = self.model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().tolist()
            
            # 显式释放中间变量
            del inputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return scores
    
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        根据与查询的相关性对文档进行重排序
        
        Args:
            query: 搜索查询
            documents: 需要重排序的文档列表
            top_k: 返回的顶部文档数量（None 表示返回所有）
            
        Returns:
            包含 'index' 和 'score' 键的字典列表
            
        Raises:
            RuntimeError: 模型未初始化时抛出
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未初始化，请先调用 startup() 方法")
        
        if not documents:
            return []
        
        # 如果未指定 top_k，则设置为文档数量
        if top_k is None:
            top_k = len(documents)
        else:
            top_k = min(top_k, len(documents))
        
        try:
            all_scores = []
            
            # 分批处理文档以避免显存溢出
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i:i + self.batch_size]
                batch_scores = await self._process_batch(query, batch_docs)
                all_scores.extend(batch_scores)
                
                # 记录显存使用情况
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    cached = torch.cuda.memory_reserved() / (1024 ** 2)
                    logger.debug(f"批次 {i//self.batch_size + 1}: GPU 内存已分配: {allocated:.2f}MB, 已缓存: {cached:.2f}MB")
            
            # 创建 (索引, 分数) 元组列表
            scored_results = [(i, score) for i, score in enumerate(all_scores)]
            
            # 按分数降序排序
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # 限制到 top_k 个结果
            scored_results = scored_results[:top_k]
            
            # 返回请求格式的结果
            return [{"index": idx, "score": float(score)} for idx, score in scored_results]
        
        except Exception as e:
            logger.exception(f"重排序过程中发生错误: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """清理资源"""
        if self.model:
            # 将模型移动到 CPU 以释放 GPU 内存
            if self.device != "cpu" and not self.device_map:
                self.model = self.model.to("cpu")
            
            # 清除 CUDA 缓存（如果可用）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 删除模型和分词器
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            logger.info(f"{self.__class__.__name__} 模型资源已释放")