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
    """Qwen3 Embedding official adaptation configuration"""
    model_path: str = Field(..., description="Local model path")
    device: str | None = Field(None, description="Computing device (cuda/cpu)")
    use_fp16: bool = Field(True, description="Use FP16 half-precision inference")
    batch_size: int = Field(16, description="Recommended batch processing size")
    instruction: str | None = Field(
        "Given a web search query, retrieve relevant passages that answer the query", 
        description="Official retrieval instruction for Qwen models"
    )

class Qwen3Embedding(BaseEmbedding[Qwen3EmbeddingConfig]):
    """
    Optimized Embedding implementation for Qwen2/Qwen3 architecture.
    Optimizations: Inference Mode, refined pooling logic, GPU memory warmup.
    """

    def __init__(self, config: Qwen3EmbeddingConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Performance optimization flag
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    async def startup(self) -> None:
        if self._is_initialized:
            return

        attn_impl = get_optimal_attn_implementation()
        model_path = self.config.model_path or self.model_name

        # Fix local path format: Ensure Hugging Face recognizes as local path
        if model_path and not model_path.startswith('/') and not model_path.startswith('./'):
            if '/' in model_path:
                # Convert relative path to absolute path
                model_path = str(Path(model_path).resolve())
            else:
                model_path = f"./{model_path}"
            logger.info(f"Corrected model path: {self.config.model_path} -> {model_path}")

        logger.info(f"🚀 Initializing Qwen Embedding: {model_path} (Attention impl: {attn_impl})")

        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,  # Force load from local files
            "attn_implementation": attn_impl,
            "torch_dtype": torch.float16 if self.config.use_fp16 and "cuda" in self.device.type else torch.float32,
        }

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            # Qwen Embedding typically requires right padding for Last Token Pooling logic
            self.tokenizer.padding_side = "right"
            
            self.model = AutoModel.from_pretrained(model_path, **load_kwargs)
            self.model.to(self.device).eval()
            
            # GPU warmup for CUDA devices
            if self.device.type == "cuda":
                with torch.inference_mode():
                    self.model(**self.tokenizer(["warmup"], return_tensors="pt").to(self.device))
            
            self._is_initialized = True
            logger.info(f"✅ Qwen3 Embedding initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen model: {e}")
            raise

    def _last_token_pooling(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Robust Last Token Pooling implementation.
        Unaffected by left/right padding - precisely locate last valid token via attention_mask.
        """
        # Get index of last valid token in each sequence (sum(mask) - 1)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        
        # Extract vectors using indices
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    async def embed(
        self, 
        texts: list[str], 
        batch_size: int | None = None, 
        is_query: bool = True
    ) -> EmbeddingResponse:
        """
        Execute text embedding with instruction enhancement and inference optimization.
        """
        if not self._is_initialized:
            await self.startup()

        if not texts:
            return self._build_empty_response(self.model_name)

        # 1. Construct instruction (follow Qwen official format)
        processed_texts = [
            f"Instruct: {self.config.instruction}\nQuery: {t}" if is_query else t 
            for t in texts
        ]

        eff_batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        total_tokens = 0

        # 2. Batch processing loop
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

            with torch.inference_mode(): # More performant than no_grad
                outputs = self.model(**inputs) # type: ignore
                
                # Pooling and normalization
                embeddings = self._last_token_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Memory optimization: detach -> cpu -> numpy is the most stable workflow
                all_embeddings.extend(embeddings.detach().cpu().numpy().tolist())

            # Active memory cleanup for large tasks
            if self.device.type == "cuda" and i % (eff_batch_size * 20) == 0:
                torch.cuda.empty_cache()

        return self._build_standard_response(
            embeddings=all_embeddings,
            model_name=self.model_name,
            tokens=total_tokens
        )

    async def shutdown(self) -> None:
        """Clean up model resources and release GPU memory"""
        if self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self._is_initialized = False
        logger.info("♻️ Qwen3 resources released successfully")

    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized and ready for inference"""
        return self._is_initialized