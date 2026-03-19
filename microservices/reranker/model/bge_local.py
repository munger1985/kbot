import os
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from pydantic import Field
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base import BaseReranker, RerankerConfig
from ...common.utils import get_optimal_attn_implementation


class BGERerankerConfig(RerankerConfig):
    """BGE Reranker specific configuration class.
    
    Extends the base reranker configuration with BGE Cross-Encoder specific
    parameters for model loading, inference optimization, and batch processing.
    """
    model_path: str = Field(..., description="Local filesystem path to the BGE reranker model")
    device: str | None = Field(None, description="Target device (cuda/cpu, auto-detected if None)")
    use_fp16: bool = Field(True, description="Whether to use FP16 precision (only for CUDA devices)")
    batch_size: int = Field(16, description="Recommended batch size for inference")

class BGEReranker(BaseReranker[BGERerankerConfig]):
    """
    Optimized reranker implementation for BGE Cross-Encoder models.
    
    Key optimizations:
    - PyTorch inference mode for faster evaluation
    - Fine-grained thread management for tokenization
    - Automatic GPU memory management and cleanup
    - Sorted batching for consistent performance
    - Flash Attention 2 integration when available
    """
    config: BGERerankerConfig

    def __init__(self, config: BGERerankerConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        # Auto-detect device if not specified (CUDA preferred over CPU)
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Single-thread executor for CPU-bound tokenization to prevent event loop blocking
        self._executor = ThreadPoolExecutor(max_workers=1)
        # Disable tokenizers parallelism to prevent deadlocks with custom executor
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    async def startup(self) -> None:
        """Initialize BGE reranker with Flash Attention 2 and inference optimizations.
        
        Loads model weights and tokenizer with optimal attention implementation,
        configures precision settings, and performs CUDA warmup for consistent
        first-request performance. Idempotent - safe to call multiple times.
        
        Raises:
            Exception: If model loading or initialization fails
        """
        if self._is_initialized:
            return

        # Get optimal attention implementation (Flash Attention 2 if available)
        attn_impl = get_optimal_attn_implementation()
        
        logger.info(f"🚀 Loading BGE Reranker: {self.config.model_path} (Device: {self.device})")

        # Model loading parameters with precision optimization
        load_kwargs = {
            "pretrained_model_name_or_path": self.config.model_path,
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
            # Use FP16 for CUDA, FP32 for CPU to prevent precision issues
            "torch_dtype": torch.float16 if self.config.use_fp16 and "cuda" in self.device.type else torch.float32,
        }

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(**load_kwargs)
            
            # Move model to target device and set to evaluation mode
            self.model.to(self.device).eval()
            
            # CUDA warmup to eliminate first-request latency
            if self.device.type == "cuda":
                with torch.inference_mode():
                    dummy_inputs = self.tokenizer([["warmup", "test"]], return_tensors="pt").to(self.device)
                    self.model(**dummy_inputs)

            self._is_initialized = True
            logger.info("✅ BGE Reranker initialized successfully")
        except Exception as e:
            logger.error(f"❌ Reranker initialization failed: {e}")
            raise

    async def rerank(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        High-performance reranking with thread-pool tokenization and GPU inference.
        
        Processes documents in batches with asynchronous tokenization (to prevent
        event loop blocking) and optimized inference mode. Includes automatic
        GPU memory cleanup and error handling for robust batch processing.
        
        Args:
            query: Input query text for relevance scoring
            documents: List of document texts to rerank
            top_k: Number of top relevant documents to return (None returns all)
            
        Returns:
            list[dict[str, Any]]: Reranked results sorted by relevance score (descending),
                each containing "index" (original position) and "score" (0-1 relevance)
        """
        # Ensure model is initialized before processing
        if not self._is_initialized:
            await self.startup()
        # Return empty list for empty document input
        if not documents:
            return []

        all_scores = []
        total_docs = len(documents)
        
        # Process documents in configured batches
        for batch_start in range(0, total_docs, self.config.batch_size):
            # Get current batch documents
            batch_docs = documents[batch_start : batch_start + self.config.batch_size]
            # Create query-document pairs for cross-encoder input
            text_pairs = [[query, doc] for doc in batch_docs]

            try:
                # Step 1: Asynchronous tokenization to avoid blocking event loop
                inputs = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda: self.tokenizer(  # type: ignore
                        text_pairs,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_tokens,
                        return_tensors="pt"
                    ).to(self.device)
                )

                # Step 2: Optimized inference with torch.inference_mode()
                with torch.inference_mode():
                    outputs = self.model(** inputs)  # type: ignore
                    # BGE Reranker outputs single logit per pair
                    logits = outputs.logits.view(-1).float()
                    # Normalize scores to 0-1 range with sigmoid
                    scores = torch.sigmoid(logits).cpu().numpy().tolist()
                    all_scores.extend(scores)

            except Exception as e:
                # Log error and assign 0.0 score to failed batch items
                logger.error(f"❌ Batch {batch_start} inference error: {e}")
                all_scores.extend([0.0] * len(batch_docs))

            # Step 3: Periodic GPU memory cleanup to prevent OOM errors
            if self.device.type == "cuda" and batch_start % (self.config.batch_size * 5) == 0:
                torch.cuda.empty_cache()

        # Step 4: Assemble results with original indices and relevance scores
        results = [
            {"index": idx, "score": score} 
            for idx, score in enumerate(all_scores)
        ]
        
        # Step 5: Sort by relevance score (descending) and apply top-k filtering
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k] if top_k else results

    async def shutdown(self) -> None:
        """Clean up resources including GPU memory and thread pool.
        
        Moves model to CPU, deletes model reference, clears CUDA cache,
        and shuts down the tokenization executor to prevent resource leaks.
        """
        # Clean up model resources
        if self.model:
            # Move model to CPU first to properly release GPU memory
            self.model.cpu()
            del self.model
            self.model = None
        
        # Clear CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Shutdown tokenization executor
        self._executor.shutdown(wait=True)
        
        # Reset initialization state
        self._is_initialized = False
        logger.info("♻️ BGE Reranker memory released safely")

    @property
    def is_initialized(self) -> bool:
        """Check if reranker is properly initialized and ready for requests.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self._is_initialized