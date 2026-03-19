import os
from pathlib import Path
import torch
from typing import Any
from pydantic import Field
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseReranker, RerankerConfig

class Qwen3RerankerConfig(RerankerConfig):
    """Qwen3 Reranker official configuration class.
    
    Extends base reranker configuration with Qwen3-specific parameters for
    generative reranker architecture, hardware optimization, and batch processing.
    """
    model_path: str = Field(..., description="Local filesystem path to Qwen3 reranker model")
    device: str | None = Field(None, description="Target device (cuda/cpu, auto-detected if None)")
    use_fp16: bool = Field(True, description="Use BF16 precision (recommended for RTX 5080/4090 GPUs)")
    batch_size: int = Field(8, description="Batch size for inference (optimized for Qwen3 architecture)")
    max_tokens: int = Field(1024, description="Maximum input sequence length")
    instruction: str | None = Field(
        "Given a query and a relevant document, retrieve the relevance score of the document to the query.", 
        description="Official default instruction for Qwen3 reranking task"
    )

class Qwen3Reranker(BaseReranker[Qwen3RerankerConfig]):
    """
    Optimized reranker implementation for Qwen3 generative architecture.
    
    Key optimizations:
    - PyTorch inference mode for accelerated evaluation
    - BF16 precision optimization for RTX 50/40 series GPUs
    - Robust token position handling for generative reranking
    - Left-padding requirement enforcement for causal language models
    - Precomputed token IDs to avoid redundant encoding
    """

    def __init__(self, config: Qwen3RerankerConfig):
        """Initialize Qwen3 reranker with configuration.
        
        Args:
            config: Qwen3-specific reranker configuration object
        """
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        # Unified device handling with auto-detection
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Precomputed token IDs for "yes"/"no" tokens (relevance scoring)
        self.yes_id = None
        self.no_id = None
        
        # Disable tokenizers parallelism to prevent deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    async def startup(self) -> None:
        """Initialize Qwen3 reranker with RTX 50 series hardware optimizations.
        
        Loads model with optimal attention implementation, configures precision
        settings (BF16 for modern GPUs), and precomputes critical token IDs.
        Includes path normalization for reliable local model loading.
        
        Raises:
            Exception: If model loading or initialization fails
        """
        if self._is_initialized:
            return

        # Get optimal attention implementation (Flash Attention 2 if available)
        from ...common.utils import get_optimal_attn_implementation
        attn_impl = get_optimal_attn_implementation()

        # Normalize model path for reliable local loading
        # Hugging Face from_pretrained requires proper absolute/relative path formatting
        model_path = self.config.model_path
        if model_path and not model_path.startswith('/') and not model_path.startswith('./'):
            if '/' in model_path:
                # Convert relative path to absolute path
                model_path = str(Path(model_path).resolve())
            else:
                # Add ./ prefix to simple relative paths
                model_path = f"./{model_path}"
            logger.info(f"Normalized model path: {self.config.model_path} -> {model_path}")

        try:
            # Step 1: Initialize tokenizer with Qwen3-specific settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                local_files_only=True
            )
            # Critical for generative rerankers: left padding ensures last token is prediction position
            self.tokenizer.padding_side = "left"
            # Set pad token to EOS if not defined (Qwen3 specific)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Step 2: Precompute "yes"/"no" token IDs (avoid redundant encoding in inference loop)
            self.yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[-1]
            self.no_id = self.tokenizer.encode("no", add_special_tokens=False)[-1]

            # Step 3: Configure precision (BF16 for modern GPUs, FP32 for CPU)
            compute_dtype = torch.bfloat16 if (self.config.use_fp16 and "cuda" in self.device.type) else torch.float32
            
            logger.info(f"🚀 Loading Qwen3 Reranker: {model_path} (Dtype: {compute_dtype}, Attention: {attn_impl})")

            # Load Qwen3 causal language model with hardware optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,  # Force local loading only
                attn_implementation=attn_impl,
                torch_dtype=compute_dtype,
                device_map={"": self.device}  # Explicit single device mapping
            )
            # Set model to evaluation mode
            self.model.eval()
            
            # CUDA warmup to eliminate first-request latency
            if self.device.type == "cuda":
                with torch.inference_mode():
                    warmup_text = "warmup"
                    self.model(**self.tokenizer([warmup_text], return_tensors="pt").to(self.device))

            self._is_initialized = True
            logger.info(f"✅ Qwen3 Reranker ready (Yes token ID: {self.yes_id})")
        except Exception as e:
            logger.error(f"❌ Qwen3 Reranker initialization failed: {e}")
            raise

    async def rerank(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Perform reranking with Qwen3 generative architecture.
        
        Uses official Qwen3 prompt template and computes relevance scores as
        the difference between "yes" and "no" token logits. Processes documents
        in optimized batches with robust error handling and global reordering.
        
        Args:
            query: Input query text for relevance scoring
            documents: List of document texts to rerank
            top_k: Number of top relevant documents to return (None returns all)
            
        Returns:
            list[dict[str, Any]]: Reranked results sorted by relevance score (descending),
                each containing "index" (original position), "score" (yes-no logit difference),
                and "document" (original document text)
        """
        # Ensure model is initialized before processing
        if not self._is_initialized:
            await self.startup()
        # Return empty list for empty document input
        if not documents:
            return []

        all_results = []
        
        # Step 1: Process documents in configured batches
        for batch_start in range(0, len(documents), self.config.batch_size):
            # Get current batch documents
            batch_docs = documents[batch_start : batch_start + self.config.batch_size]
            
            # Format inputs with official Qwen3 reranking prompt template
            formatted_texts = [
                f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>: {d}\nRelevant (yes/no):"
                for d in batch_docs
            ]

            try:
                # Step 2: Tokenization with left padding (critical for generative models)
                inputs = self.tokenizer(  # type: ignore
                    formatted_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_tokens,
                    return_tensors="pt"
                ).to(self.device)

                # Step 3: Optimized inference with torch.inference_mode()
                with torch.inference_mode():
                    outputs = self.model(** inputs)  # type: ignore
                    # Get logits for last token position (prediction position)
                    last_token_logits = outputs.logits[:, -1, :] 
                    
                    # Step 4: Calculate relevance score (yes logit - no logit)
                    # Convert to float for stable subtraction with BF16 precision
                    yes_logits = last_token_logits[:, self.yes_id].float()
                    no_logits = last_token_logits[:, self.no_id].float()
                    scores = (yes_logits - no_logits).cpu().numpy().tolist()

                # Step 5: Assemble batch results with global indices
                for batch_idx, score in enumerate(scores):
                    all_results.append({
                        "index": batch_start + batch_idx,  # Global document index
                        "score": score,
                        "document": batch_docs[batch_idx]
                    })
                    
            except Exception as e:
                # Log error and continue processing remaining batches
                logger.error(f"❌ Batch inference failed [{batch_start}]: {e}")
                continue

        # Step 6: Global reordering by relevance score (descending)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top-k filtering if specified
        return all_results[:top_k] if top_k else all_results

    async def shutdown(self) -> None:
        """Clean up Qwen3 reranker resources completely.
        
        Releases model memory, clears CUDA cache, and resets initialization state
        to prevent memory leaks and ensure proper cleanup.
        """
        if self.model:
            # Delete model reference to free memory
            del self.model
            # Clean up CUDA resources if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        # Reset initialization state
        self._is_initialized = False
        logger.info("♻️ Qwen3 Reranker resources fully released")

    @property
    def is_initialized(self) -> bool:
        """Check if reranker is properly initialized and ready for requests.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self._is_initialized