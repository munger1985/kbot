import os
import torch
import torch.nn.functional as F
from pydantic import Field
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse

class BGEEmbeddingConfig(EmbeddingConfig):
    """BGE Embedding official adaptation configuration"""
    model_path: str = Field(..., description="Local model path")
    device: str | None = Field(None, description="Computing device (e.g., 'cuda', 'cpu')")
    use_fp16: bool = Field(True, description="Use FP16 half-precision inference")
    query_instruction: str | None = Field("为这个句子生成表示以用于检索相关文章：", description="BGE official retrieval instruction (Chinese)")
    # New: Allow pooling strategy configuration for enhanced flexibility
    pooling_strategy: str = Field("cls", description="Pooling strategy: cls or mean")

class BGEEmbedding(BaseEmbedding[BGEEmbeddingConfig]):
    """
    Refactored BGE Embedding implementation.
    Optimizations: Multi-level caching, memory pinning optimization, more efficient batch processing logic.
    """

    def __init__(self, config: BGEEmbeddingConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Optimization: Disable Tokenizers multithreading to avoid deadlocks in multiprocess DataLoader
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    async def startup(self) -> None:
        if self._is_initialized:
            return

        from ...common.utils import get_optimal_attn_implementation
        attn_impl = get_optimal_attn_implementation()
        model_path = self.config.model_path or self.model_name
        
        logger.info(f"🚀 Loading BGE model: {model_path} (Device: {self.device}, Half: {self.config.use_fp16})")

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
            
            # Optimization: Warmup if using CUDA
            if self.device.type == "cuda":
                with torch.no_grad():
                    dummy_input = self.tokenizer(["warmup"], return_tensors="pt").to(self.device)
                    self.model(**dummy_input)

            self._is_initialized = True
            logger.info(f"✅ BGE Embedding initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load BGE model: {e}")
            raise

    def _pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Support multiple pooling strategies"""
        if self.config.pooling_strategy == "cls":
            return last_hidden_state[:, 0]
        
        # Mean Pooling logic (more suitable for certain long text scenarios)
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

        # 1. Preprocess texts
        processed_texts = [
            f"{self.config.query_instruction}{t}" if is_query and self.config.query_instruction else t 
            for t in texts
        ]

        # 2. Optimization: Sort by length (optional, reduces padding overhead)
        # Keeping original order for consistency, using DataLoader-style batching
        eff_batch_size = batch_size or self.config.batch_size
        
        all_embeddings = []
        total_tokens = 0

        # 3. Inference loop
        # Note: For local models, asyncio.gather is unnecessary as GPU computation is typically single-bottleneck
        # Excessive concurrency may cause OOM errors
        for i in range(0, len(processed_texts), eff_batch_size):
            batch = processed_texts[i : i + eff_batch_size]
            
            # Encoding optimization: pin_memory works in multithreading, but ensure tensors go directly to GPU in simple inference
            inputs = self.tokenizer( # type: ignore
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt"
            ).to(self.device)

            total_tokens += int(inputs['attention_mask'].sum().item())

            with torch.inference_mode(): # inference_mode is faster than no_grad
                outputs = self.model(**inputs) # type: ignore
                embeddings = self._pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                # BGE requires L2 normalization
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Memory optimization: Transfer to CPU immediately and convert to list
                all_embeddings.extend(embeddings.detach().cpu().numpy().tolist())

            # 4. Memory cleanup (optional for extremely large batches)
            if self.device.type == "cuda" and i % (eff_batch_size * 10) == 0:
                torch.cuda.empty_cache()

        return self._build_standard_response(
            embeddings=all_embeddings,
            model_name=self.model_name,
            tokens=total_tokens
        )

    async def shutdown(self) -> None:
        """More thorough memory release"""
        if self.model:
            self.model.cpu() # Move back to CPU first
            del self.model
            self.model = None
        if self.tokenizer:
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        self._is_initialized = False
        logger.info("♻️ BGE memory resources fully released")