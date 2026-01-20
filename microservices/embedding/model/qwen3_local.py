import os
from pathlib import Path
import torch
import torch.nn.functional as F
from pydantic import Field
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse
from ...common.utils import get_optimal_attn_implementation

class Qwen3EmbeddingConfig(EmbeddingConfig):
    """Qwen3 Embedding 官方适配配置"""
    model_path: str = Field(..., description="本地路径")
    device: str | None = Field(None, description="设备")
    use_fp16: bool = Field(True, description="半精度推理")
    batch_size: int = Field(16, description="建议批处理大小")
    instruction: str | None = Field(
        "Given a web search query, retrieve relevant passages that answer the query", 
        description="官方检索指令"
    )

class Qwen3Embedding(BaseEmbedding[Qwen3EmbeddingConfig]):
    """
    针对 Qwen2/Qwen3 架构优化的 Embedding 实现
    优化点：Inference Mode、精细化池化逻辑、显存预热
    """

    def __init__(self, config: Qwen3EmbeddingConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 性能开关
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    async def startup(self) -> None:
        if self._is_initialized:
            return

        attn_impl = get_optimal_attn_implementation()
        model_path = self.config.model_path or self.model_name

        # 修正本地路径格式：确保 Hugging Face 识别为本地路径
        if model_path and not model_path.startswith('/') and not model_path.startswith('./'):
            if '/' in model_path:
                # 可能是相对路径，转换为绝对路径
                model_path = str(Path(model_path).resolve())
            else:
                model_path = f"./{model_path}"
            logger.info(f"修正模型路径: {self.config.model_path} -> {model_path}")

        logger.info(f"🚀 正在初始化 Qwen Embedding: {model_path} (Impl: {attn_impl})")

        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,  # 强制从本地加载
            "attn_implementation": attn_impl,
            "torch_dtype": torch.float16 if self.config.use_fp16 and "cuda" in self.device.type else torch.float32,
        }

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            # Qwen Embedding 通常需要 padding 在右侧以配合 Last Token Pooling 逻辑
            self.tokenizer.padding_side = "right"
            
            self.model = AutoModel.from_pretrained(model_path, **load_kwargs)
            self.model.to(self.device).eval()
            
            # CUDA 预热
            if self.device.type == "cuda":
                with torch.inference_mode():
                    self.model(**self.tokenizer(["warmup"], return_tensors="pt").to(self.device))
            
            self._is_initialized = True
            logger.info(f"✅ Qwen3 Embedding 初始化成功")
        except Exception as e:
            logger.error(f"❌ Qwen 加载失败: {e}")
            raise

    def _last_token_pooling(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        健壮的 Last Token Pooling 实现
        不受左/右 Padding 干扰，通过 attention_mask 精确锁定最后一个有效 Token
        """
        # 获取每个序列中最后一个有效 token 的索引 (即 sum(mask) - 1)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        
        # 使用索引抽取向量
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    async def embed(
        self, 
        texts: list[str], 
        batch_size: int | None = None, 
        is_query: bool = True
    ) -> EmbeddingResponse:
        """
        执行向量化：支持指令增强与推理模式优化
        """
        if not self._is_initialized:
            await self.startup()

        if not texts:
            return self._build_empty_response(self.model_name)

        # 1. 构造指令 (遵循 Qwen 官方格式)
        processed_texts = [
            f"Instruct: {self.config.instruction}\nQuery: {t}" if is_query else t 
            for t in texts
        ]

        eff_batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        total_tokens = 0

        # 2. 批处理循环
        for i in range(0, len(processed_texts), eff_batch_size):
            batch = processed_texts[i : i + eff_batch_size]
            
            inputs = self.tokenizer( # type: ignore
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt"
            ).to(self.device)

            total_tokens += int(inputs['attention_mask'].sum().item())

            with torch.inference_mode(): # 性能优于 no_grad
                outputs = self.model(**inputs) # type: ignore
                
                # 池化与归一化
                embeddings = self._last_token_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # 显存释放：detach -> cpu -> numpy 流程最稳
                all_embeddings.extend(embeddings.detach().cpu().numpy().tolist())

            # 针对大任务的主动显存清理
            if self.device.type == "cuda" and i % (eff_batch_size * 20) == 0:
                torch.cuda.empty_cache()

        return self._build_standard_response(
            embeddings=all_embeddings,
            model_name=self.model_name,
            tokens=total_tokens
        )

    async def shutdown(self) -> None:
        if self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self._is_initialized = False
        logger.info("♻️ Qwen3 资源已释放")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized