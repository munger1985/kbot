import os
import gc
import torch
import inspect
from typing import Any
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from prometheus_client import Histogram, Counter, Gauge
from models.embedding.base import BaseEmbedding, LocalEmbeddingConfig, EmbeddingResponse, EmbeddingDataItem
from core.config import settings



class LocalEmbedding(BaseEmbedding):
    """
    Production-grade local embedding model with enhanced configuration management,
    error handling, and resource optimization.
    """

    # Prometheus metrics
    LATENCY_HIST = Histogram(
        'local_embedding_latency_seconds',
        'Latency for local embedding requests',
        ['model_name']
    )
    
    ERROR_COUNTER = Counter(
        'local_embedding_errors_total',
        'Count of local embedding errors',
        ['model_name', 'error_type']
    )
    
    MEMORY_GAUGE = Gauge(
        'local_embedding_memory_usage_mb',
        'GPU memory usage in MB',
        ['device_id']
    )
    
    REQUEST_SIZE_GAUGE = Gauge(
        'local_embedding_request_size',
        'Size of embedding requests in characters',
        ['model_name']
    )

    def __init__(self, config: LocalEmbeddingConfig):
        """
        Initialize local embedding model with robust configuration validation.
        
        Args:
            config: Configuration containing:
                - model_name: HuggingFace model identifier (e.g. "BAAI/bge-small-en-v1.5")
                - model_path: Optional local path to model files
                - device: Target device (e.g., "cuda:0", "cpu")
                - device_map: For multi-GPU setups (e.g., "auto", "balanced")
                - max_tokens: Maximum input sequence length
                - compile_model: Whether to compile model with torch.compile()
                - use_fp16: Use half-precision inference
                - local_files_only: Only use local model files
                - trust_remote_code: Trust custom model code from HuggingFace
                - max_memory: Dict of GPU memory limits (e.g. {"0": "24GB", "1": "24GB"})
        
        Raises:
            TypeError: If config is not of type LocalEmbeddingConfig
            ValueError: If required settings are missing or invalid
        """
        # Validate config type
        if not isinstance(config, LocalEmbeddingConfig):
            raise TypeError("config must be an instance of LocalEmbeddingConfig")

        # Model components
        self.model: torch.nn.Module | None = None
        self.tokenizer: Any | None = None
        
        # Configuration with validation
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.predownload = False
        self.cache_path = os.path.join("./models/local_model_cache", self.model_name)
        self.name_or_path = ""
        
        # Device configuration
        self.device = config.device
        self.device_map = config.device_map
        self.max_memory = getattr(config, 'max_memory', None)
        
        # Model parameters with defaults
        try:
            self.max_tokens = getattr(config, 'max_tokens', settings['embed']['max_tokens'])
        except (KeyError, TypeError) as e:
            logger.error("Invalid settings structure for max_tokens")
            raise ValueError("Missing or invalid max_tokens setting") from e
        
        self.compile_model = getattr(config, 'compile_model', True)
        self.use_fp16 = getattr(config, 'use_fp16', False)
        self.local_files_only = getattr(config, 'local_files_only', False)
        self.trust_remote_code = getattr(config, 'trust_remote_code', False)
        
        # Runtime state
        self._batch_size = self._auto_detect_batch_size()
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize the embedding model with comprehensive error handling."""
        if self._is_initialized:
            logger.warning("Model is already initialized")
            return

        try:
            # Validate model path or cache
            if self.model_path is None and self.local_files_only:
                raise ValueError("Local model path not specified when local_files_only=True")

            if self.model_path is not None:
                if self._validate_embedding_model(self.model_path):
                    self.predownload = True
                else:
                    if self._validate_embedding_model(self.cache_path):
                        logger.info(f"Using cached embedding model: {self.cache_path}")
                        self.model_path = os.path.abspath(self.cache_path)
                        self.predownload = True
                    else:
                        self.predownload = False

            self.name_or_path = self.model_path if self.predownload else self.model_name

            logger.debug(f"Embedding model name: {self.model_name}, path: {self.model_path}")
            logger.debug(f"Embedding model name or path: {self.name_or_path}")

            # Load tokenizer with error handling
            self.tokenizer = self._load_tokenizer()
            
            # Load model with optimized settings
            self.model = self._load_model()
            
            # Cache model if downloaded
            if not self.predownload:
                self._cache_model()
                logger.debug(f"Embedding model {self.model_name} cached at: {self.cache_path}")
            
            # Model optimization
            self._optimize_model()
            
            self._is_initialized = True
            logger.success(f"Embedding model {self.model_name} initialized successfully")
            
        except Exception as e:
            self._is_initialized = False
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _load_tokenizer(self) -> Any:
        """Load tokenizer with comprehensive configuration."""
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.name_or_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
            model_max_length=self.max_tokens,
            padding_side='right',
            local_files_only=self.local_files_only
        )

    def _load_model(self) -> torch.nn.Module:
        """Load model with proper device and precision settings."""
        load_kwargs = {
            "pretrained_model_name_or_path": self.name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
        }

        # Device and precision configuration
        if torch.cuda.is_available():
            if self.device_map is not None:  # Multi-GPU
                load_kwargs.update({
                    "device_map": self.device_map,
                    "max_memory": self.max_memory,
                })
                target_device = None
            else:  # Single GPU
                target_device = self.device or "cuda:0"
            
            load_kwargs["torch_dtype"] = torch.float16 if self.use_fp16 else torch.float32
        else:  # CPU fallback
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
            target_device = "cpu"

        model = AutoModel.from_pretrained(**load_kwargs)
        
        # Move to target device if not using device_map
        if target_device is not None:
            model = model.to(target_device)
            logger.debug(f"Model loaded to device: {target_device}")
        else:
            logger.debug(f"Model loaded with device_map: {self.device_map}")
            
        return model

    def _optimize_model(self) -> None:
        """Apply model optimizations like compilation and eval mode."""
        if self.model is None:
            return
            
        self.model.eval()
        
        # Model compilation (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile( # type: ignore
                    self.model,
                    mode='max-autotune' if torch.cuda.is_available() else None
                )
                logger.debug("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

    def _validate_embedding_model(self, model_path: str) -> bool:
        """Validate that the model directory contains all required files."""
        must_have = ["config.json", "tokenizer_config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        vocab_files = ["vocab.txt", "vocab.json", "tokenizer.json"]
        
        # Check required files
        for f in must_have:
            if not os.path.exists(os.path.join(model_path, f)):
                return False
        
        # Check model weights (at least one)
        if not any(os.path.exists(os.path.join(model_path, f)) for f in model_files):
            return False
        
        # Check vocabulary files (at least one)
        if not any(os.path.exists(os.path.join(model_path, f)) for f in vocab_files):
            return False
        
        return True

    def _cache_model(self) -> None:
        """Save model to cache directory with error handling."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be initialized before caching")
            
        try:
            os.makedirs(self.cache_path, exist_ok=True)
            self.model.save_pretrained(self.cache_path) # type: ignore
            self.tokenizer.save_pretrained(self.cache_path)
        except Exception as e:
            logger.error(f"Failed to cache model: {e}")
            raise

    def _auto_detect_batch_size(self) -> int:
        """Dynamically determine safe batch size based on available hardware."""
        if not torch.cuda.is_available():
            return 32  # Conservative CPU batch size
        
        try:
            # Get current GPU memory state
            device = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            free_mem = total_mem - reserved_mem
            
            # Estimate memory per batch based on model size
            model_size = self.model_name.split("-")[-1].lower()
            est_mem_per_batch = {
                'small': 0.3 * (1024**3),  # 0.3GB
                'base': 1.0 * (1024**3),   # 1.0GB
                'large': 2.5 * (1024**3),  # 2.5GB
                'xl': 4.0 * (1024**3)      # 4.0GB
            }.get(model_size, 1.0 * (1024**3))  # Default 1.0GB
            
            # Calculate safe batch size (using 70% of free memory)
            safe_batch = int((free_mem * 0.7) / est_mem_per_batch)
            return max(1, min(safe_batch, 128))  # Clamp between 1 and 128
        except Exception as e:
            logger.warning(f"Batch size detection failed: {e}, using fallback 32")
            return 32

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 0,
        normalize: bool = True,
        raise_on_error: bool = True,
    ) -> EmbeddingResponse:
        """
        Generate embeddings with automatic batch processing and comprehensive monitoring.
        
        Args:
            texts: list of input texts to embed
            batch_size: Override auto-detected batch size (0 for auto)
            normalize: L2-normalize output embeddings
            raise_on_error: Whether to raise exceptions or return empty response
            
        Returns:
            EmbeddingResponse: Standardized response with embeddings
            
        Raises:
            RuntimeError: If model not initialized and raise_on_error=True
        """
        # Record request size for monitoring
        total_chars = sum(len(text) for text in texts)
        self.REQUEST_SIZE_GAUGE.labels(model_name=self.model_name).set(total_chars)
        
        # Validate inputs
        if not self._validate_inputs(texts, raise_on_error):
            return self._empty_response()
        
        # Determine batch size
        effective_batch_size = batch_size if batch_size > 0 else self._batch_size
        
        try:
            with self.LATENCY_HIST.labels(model_name=self.model_name).time():
                return await self._process_batches(texts, effective_batch_size, normalize)
        except Exception as e:
            return self._handle_embed_error(e, effective_batch_size, raise_on_error)

    def _validate_inputs(self, texts: list[str], raise_on_error: bool) -> bool:
        """Validate input texts and model state."""
        if not self._is_initialized:
            if raise_on_error:
                raise RuntimeError("Model not initialized. Call startup() first.")
            logger.error("Model not initialized")
            return False
            
        if not texts:
            logger.warning(
                "Empty input list - "
                f"Caller: {self._get_caller_info()}, "
                f"Model: {self.model_name}, "
                f"Initialized: {self._is_initialized}"
            )
            return False
            
        return True

    def _get_caller_info(self) -> str:
        """Get information about the function that called this method."""
        try:
            frame = inspect.currentframe()
            if frame is None or frame.f_back is None or frame.f_back.f_back is None:
                return "unknown"
            
            caller_frame = frame.f_back.f_back
            return (
                f"{caller_frame.f_code.co_name}() in "
                f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
            )
        except Exception:
            return "unknown"

    async def _process_batches(
        self,
        texts: list[str],
        batch_size: int,
        normalize: bool
    ) -> EmbeddingResponse:
        """Process texts in batches and return embeddings."""
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings, tokens = await self._process_single_batch(batch, normalize)
            all_embeddings.append(embeddings)
            total_tokens += tokens
            
            # Update memory gauge
            if torch.cuda.is_available():
                device = self.model.device if hasattr(self.model, 'device') else "cuda:0" # type: ignore
                mem_used = torch.cuda.memory_allocated(device) / (1024**2) # type: ignore # MB
                self.MEMORY_GAUGE.labels(device_id=str(device)).set(mem_used)
        
        return self._build_response(all_embeddings, total_tokens)

    async def _process_single_batch(
        self,
        batch: list[str],
        normalize: bool
    ) -> tuple[torch.Tensor, int]:
        """Process a single batch of texts."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model and tokenizer must be initialized")
        
        # Tokenize
        encoded_input = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )
        
        # Skip empty inputs
        if encoded_input['input_ids'].numel() == 0:
            return torch.empty((0, self.embedding_dim)), 0
        
        # Move inputs to correct device
        if not hasattr(self.model, 'device_map'):  # Single device
            device = self.model.device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        logger.debug(f"Input IDs shape: {encoded_input['input_ids'].shape}")
        logger.debug(f"Attention mask shape: {encoded_input['attention_mask'].shape}")

        # Inference
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            logger.debug(f"Model outputs last_hidden_state shape: {outputs.last_hidden_state.shape}")
        
        # Pooling
        embeddings = self._mean_pooling(
            outputs.last_hidden_state,
            encoded_input['attention_mask']
        )
        
        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Calculate tokens in this batch
        tokens = encoded_input['input_ids'].numel()
        
        return embeddings.cpu(), tokens

    def _build_response(
        self,
        all_embeddings: list[torch.Tensor],
        total_tokens: int
    ) -> EmbeddingResponse:
        """确保每个输入文本对应一个embedding"""
        if not all_embeddings:
            return self._empty_response()

        # 合并所有批次的embeddings [total_texts, hidden_dim]
        embeddings_np = torch.cat(all_embeddings, dim=0).cpu().numpy()
        
        # 创建响应项
        data = []
        for idx, embedding in enumerate(embeddings_np):
            if embedding.ndim == 0:  # 标量情况
                embedding = [float(embedding)]
            elif embedding.ndim > 1:  # 异常高维
                embedding = embedding.squeeze().tolist()
            
            data.append(EmbeddingDataItem(
                embedding=embedding.tolist() if hasattr(embedding, 'tolist') else embedding, # type: ignore
                index=idx,
                object="embedding"
            ))
        
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
        """Handle errors during embedding generation."""
        error_name = type(error).__name__
        self.ERROR_COUNTER.labels(
            model_name=self.model_name,
            error_type=error_name
        ).inc()
        
        # Special handling for OOM errors
        if isinstance(error, torch.cuda.OutOfMemoryError):
            new_batch_size = max(1, batch_size // 2)
            self._batch_size = new_batch_size
            logger.warning(
                f"CUDA OOM. Batch size auto-adjusted from {batch_size}→{new_batch_size}"
            )
            
            if raise_on_error:
                raise RuntimeError(
                    f"CUDA OOM. Suggested batch_size={new_batch_size}"
                ) from error
        
        elif raise_on_error:
            raise error
            
        return self._empty_response()

    def _empty_response(self) -> EmbeddingResponse:
        """Return an empty response with proper structure."""
        return EmbeddingResponse(
            data=[],
            model=self.model_name,
            object="list",
            usage={"prompt_tokens": 0, "total_tokens": 0}
        )

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean token embeddings weighted by attention mask.
        
        Args:
            token_embeddings: Tensor of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Tensor of shape (batch_size, sequence_length)
            
        Returns:
            Pooled embeddings of shape (batch_size, hidden_size)
            
        Raises:
            ValueError: If input tensors have incorrect shapes or types
        """
        # Input validation
        if token_embeddings.dim() != 3:
            raise ValueError(f"Token embeddings must be 3D, got {token_embeddings.dim()}D")
        if attention_mask.dim() != 2:
            raise ValueError(f"Attention mask must be 2D, got {attention_mask.dim()}D")
        if token_embeddings.shape[0] != attention_mask.shape[0]:
            raise ValueError(f"Batch size mismatch: {token_embeddings.shape[0]} != {attention_mask.shape[0]}")
            
        # Ensure same device
        if token_embeddings.device != attention_mask.device:
            attention_mask = attention_mask.to(token_embeddings.device)
            
        # Handle empty input
        if token_embeddings.numel() == 0:
            return torch.zeros((0, token_embeddings.size(-1)), 
                             device=token_embeddings.device)
        logger.debug(f"Pooling input shape: {token_embeddings.shape}")

        # 扩展mask维度以匹配embeddings [batch_size, seq_len, 1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        
        # 加权求和 [batch_size, hidden_dim]
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        
        # 计算有效token数 [batch_size, 1]
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # 平均池化 [batch_size, hidden_dim]
        pooled = sum_embeddings / sum_mask

        logger.debug(f"Pooled output shape: {pooled.shape}")

        # Ensure correct output shape
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
            
        return pooled

    async def shutdown(self) -> None:
        """Clean up resources safely and thoroughly."""
        if not self._is_initialized:
            return
            
        try:
            # Move model to CPU to free GPU memory
            if self.model is not None:
                if hasattr(self.model, "device") and str(self.model.device) != "cpu":
                    self.model.to("cpu")
                
                # Additional cleanup if available
                if hasattr(self.model, "cleanup"):
                    self.model.cleanup() # type: ignore
                
                del self.model
                self.model = None
            
            # Cleanup tokenizer
            if self.tokenizer is not None:
                if hasattr(self.tokenizer, "cleanup"):
                    self.tokenizer.cleanup()
                
                del self.tokenizer
                self.tokenizer = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self._is_initialized = False
            logger.info("Model resources released successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

    async def warmup(self, sample_text: str = "This is a warmup text.") -> None:
        """Warm up the model to avoid first-time inference delays."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized")
        
        logger.info("Warming up embedding model...")
        try:
            with torch.no_grad():
                # Small batch warmup
                await self.embed([sample_text] * 2, batch_size=2, raise_on_error=True)
                # Medium batch warmup
                await self.embed([sample_text] * self._batch_size, 
                               batch_size=self._batch_size, 
                               raise_on_error=False)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def health_check(self) -> dict[str, Any]:
        """Check model health and resource status."""
        status = {
            "initialized": self._is_initialized,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "batch_size": self._batch_size,
            "model_name": self.model_name
        }
        
        if torch.cuda.is_available():
            device = self.model.device if self.model else "cuda:0"
            status.update({
                "gpu_memory_used_mb": torch.cuda.memory_allocated(device) / (1024**2), # type: ignore
                "gpu_memory_total_mb": torch.cuda.get_device_properties(device).total_memory / (1024**2), # type: ignore
                "device": str(device)
            })
        
        return status

    def model_info(self) -> dict[str, Any]:
        """Get detailed information about the loaded model."""
        if not self._is_initialized or self.model is None:
            raise RuntimeError("Model not initialized")
        
        info = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "embedding_dim": self.embedding_dim,
            "max_sequence_length": self.max_tokens,
            "device": str(self.model.device) if hasattr(self.model, "device") else str(self.device_map),
            "dtype": str(self.model.dtype) if hasattr(self.model, "dtype") else "unknown"
        }
        
        if hasattr(self.model, "config"):
            config = self.model.config.to_dict() # type: ignore
            info.update({
                "architecture": config.get("architectures", ["unknown"])[0],
                "hidden_size": config.get("hidden_size", "unknown"),
                "num_layers": config.get("num_hidden_layers", "unknown"),
                "model_type": config.get("model_type", "unknown")
            })
        
        return info

    @property
    def embedding_dim(self) -> int:
        """Get the output dimension of embeddings."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model.config.hidden_size # type: ignore

    @property
    def recommended_batch_size(self) -> int:
        """Get the currently recommended batch size."""
        return self._batch_size