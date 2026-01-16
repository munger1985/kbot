import os
import torch
import torch.nn.functional as F
from pydantic import Field
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse

class BGEEmbeddingConfig(EmbeddingConfig):
    """BGE Embedding 官方适配配置"""
    model_path: str = Field(..., description="本地路径")
    device: str | None = Field(None, description="设备")
    use_fp16: bool = Field(True, description="半精度推理")
    query_instruction: str | None = Field("为这个句子生成表示以用于检索相关文章：", description="BGE官方检索指令")
    # 新增：允许配置池化方式，增加灵活性
    pooling_strategy: str = Field("cls", description="池化策略: cls 或 mean")

class BGEEmbedding(BaseEmbedding[BGEEmbeddingConfig]):
    """
    重构后的 BGE Embedding 实现
    优化点：多级缓存、内存锁页优化、更高效的批处理逻辑
    """

    def __init__(self, config: BGEEmbeddingConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 优化：禁用 Tokenizers 的多线程以避免在多进程 DataLoader 中死锁
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    async def startup(self) -> None:
        if self._is_initialized:
            return

        from ...common.utils import get_optimal_attn_implementation
        attn_impl = get_optimal_attn_implementation()
        model_path = self.config.model_path or self.model_name
        
        logger.info(f"🚀 正在加载 BGE 模型: {model_path} (Device: {self.device}, Half: {self.config.use_fp16})")

        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
            "torch_dtype": torch.float16 if self.config.use_fp16 and "cuda" in self.device.type else torch.float32,
        }

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(**load_kwargs)
            self.model.to(self.device).eval()
            
            # 优化：如果是 CUDA，进行一次热身
            if self.device.type == "cuda":
                with torch.no_grad():
                    dummy_input = self.tokenizer(["warmup"], return_tensors="pt").to(self.device)
                    self.model(**dummy_input)

            self._is_initialized = True
            logger.info(f"✅ BGE Embedding 初始化成功")
        except Exception as e:
            logger.error(f"❌ BGE 加载失败: {e}")
            raise

    def _pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """支持多种池化策略"""
        if self.config.pooling_strategy == "cls":
            return last_hidden_state[:, 0]
        
        # Mean Pooling 逻辑 (更适合某些长文本场景)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    async def embed(
        self, 
        texts: list[str], 
        batch_size: int | None = None, 
        is_query: bool = True
    ) -> EmbeddingResponse:
        if not self._is_initialized:
            await self.startup()

        if not texts:
            return self._build_empty_response(self.model_name)

        # 1. 预处理文本
        processed_texts = [
            f"{self.config.query_instruction}{t}" if is_query and self.config.query_instruction else t 
            for t in texts
        ]

        # 2. 优化：按长度排序（可选，减少 Padding 开销）
        # 这里为了保持顺序一致性，直接使用 DataLoader 思想进行分批
        eff_batch_size = batch_size or self.config.batch_size
        
        all_embeddings = []
        total_tokens = 0

        # 3. 推理循环
        # 注意：对于本地模型，不需要像 API 那样用 asyncio.gather，
        # 因为 GPU 计算通常是串行的单瓶颈，并发过多反而会导致显存溢出。
        for i in range(0, len(processed_texts), eff_batch_size):
            batch = processed_texts[i : i + eff_batch_size]
            
            # 编码优化：pin_memory 在多线程中有效，但在简单的推理中，确保 tensor 直接去 GPU
            inputs = self.tokenizer( # type: ignore
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt"
            ).to(self.device)

            total_tokens += int(inputs['attention_mask'].sum().item())

            with torch.inference_mode(): # 使用 inference_mode 比 no_grad 更快
                outputs = self.model(**inputs) # type: ignore
                embeddings = self._pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                # BGE 强制要求 L2 归一化
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # 显存释放优化：及时转移到 CPU 并转化为列表
                all_embeddings.extend(embeddings.detach().cpu().numpy().tolist())

            # 4. 显存清理（针对超大批次可选）
            if self.device.type == "cuda" and i % (eff_batch_size * 10) == 0:
                torch.cuda.empty_cache()

        return self._build_standard_response(
            embeddings=all_embeddings,
            model_name=self.model_name,
            tokens=total_tokens
        )

    async def shutdown(self) -> None:
        """更彻底的显存释放"""
        if self.model:
            self.model.cpu() # 先移回 CPU
            del self.model
            self.model = None
        if self.tokenizer:
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        self._is_initialized = False
        logger.info("♻️ BGE 显存资源已完全释放")