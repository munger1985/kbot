import os
import gc
import torch
import logging
from typing import Any, List, Dict, Optional
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from prometheus_client import Histogram, Counter, Gauge
from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem

class LocalEmbeddingConfig(EmbeddingConfig):
    model_path: Optional[str] = None
    device: Optional[str] = None
    device_map: Optional[str] = None
    max_memory: Optional[Dict] = None
    trust_remote_code: bool = False
    use_fp16: bool = False
    local_files_only: bool = False
    compile_model: bool = True  # True when PyTorch 2.0+ else False
    cache_dir: str = "./cached_models"  # 从nacos manager中读取cache_dir配置，用于缓存模型


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
            config: Configuration containing model and device settings
        
        Raises:
            TypeError: If config is not of type LocalEmbeddingConfig
            ValueError: If required settings are missing or invalid
        """
        # Validate config type
        if not isinstance(config, LocalEmbeddingConfig):
            raise TypeError("config must be an instance of LocalEmbeddingConfig")

        # Model components
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[Any] = None
        
        # Configuration with validation
        self.config = config
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.predownload = False
        self.cache_path = os.path.join(config.cache_dir, self.model_name.replace('/', '_'))
        self.name_or_path = ""
        
        # Device configuration - respect explicit user settings first
        self.device = config.device
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.device_map = config.device_map
        self.max_memory = getattr(config, 'max_memory', None)
        
        # Model parameters with defaults
        self.max_tokens = getattr(config, 'max_tokens', 512)
        self.compile_model = getattr(config, 'compile_model', True)
        self.use_fp16 = getattr(config, 'use_fp16', False)
        self.local_files_only = getattr(config, 'local_files_only', False)
        self.trust_remote_code = getattr(config, 'trust_remote_code', False)
        
        # Runtime state
        self._batch_size = 32  # Default, will be updated after model loading
        self._is_initialized = False
        self._using_device_map = False  # Track if using device_map loading

    async def startup(self) -> None:
        """Initialize the embedding model with comprehensive error handling."""
        if self._is_initialized:
            logger.warning("Model is already initialized")
            return

        try:
            # Validate model path or cache
            if self.model_path is None and self.local_files_only:
                raise ValueError("Local model path not specified when local_files_only=True")

            # Check if we have a valid local model path or need to download
            if self.model_path is not None and os.path.exists(self.model_path):
                if self._validate_model_files(self.model_path):
                    self.predownload = True
                    logger.info(f"Using pre-downloaded model from: {self.model_path}")
                else:
                    raise ValueError(f"Model path {self.model_path} exists but contains invalid model files")
            else:
                # Check cache or download from hub
                if os.path.exists(self.cache_path) and self._validate_model_files(self.cache_path):
                    self.model_path = self.cache_path
                    self.predownload = True
                    logger.info(f"Using cached model from: {self.cache_path}")
                else:
                    self.predownload = False
                    logger.info(f"Will download model from hub: {self.model_name}")

            self.name_or_path = self.model_path if self.predownload else self.model_name

            logger.debug(f"Embedding model name: {self.model_name}, path: {self.model_path}")
            logger.debug(f"Embedding model name or path: {self.name_or_path}")

            # Load tokenizer with error handling
            self.tokenizer = self._load_tokenizer()
            
            # Load model with optimized settings
            self.model = self._load_model()
            
            # Update batch size based on actual model size and available resources
            self._batch_size = self._auto_detect_batch_size()
            logger.info(f"Auto-detected batch size: {self._batch_size}")
            
            # Model optimization
            self._optimize_model()
            
            self._is_initialized = True
            logger.info(f"Embedding model {self.model_name} initialized successfully on device: {self.device}")
            
        except Exception as e:
            self._is_initialized = False
            logger.exception(f"Failed to initialize model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _load_tokenizer(self) -> Any:
        """Load tokenizer with comprehensive configuration."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.name_or_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
                model_max_length=self.max_tokens,
                padding_side='right',
                local_files_only=self.local_files_only,
                cache_dir=self.config.cache_dir  # Use configured cache directory
            )
            logger.debug("Tokenizer loaded successfully")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise

    def _load_model(self) -> torch.nn.Module:
        """Load model with proper device and precision settings."""
        load_kwargs = {
            "pretrained_model_name_or_path": self.name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
            "cache_dir": self.config.cache_dir  # Use configured cache directory
        }

        # Determine device configuration
        if self.device_map is not None:
            # Use device_map for multi-GPU loading
            load_kwargs.update({
                "device_map": self.device_map,
                "max_memory": self.max_memory,
            })
            self._using_device_map = True
            target_device = None
            logger.debug(f"Loading with device_map: {self.device_map}")
        else:
            # Single device loading
            self._using_device_map = False
            target_device = self.device
            load_kwargs["torch_dtype"] = torch.float16 if self.use_fp16 else torch.float32

        try:
            model = AutoModel.from_pretrained(**load_kwargs)
            
            # Move to target device if not using device_map
            if not self._using_device_map and target_device is not None:
                model = model.to(target_device)
                logger.debug(f"Model moved to device: {target_device}")
            
            logger.debug("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _validate_model_files(self, model_path: str) -> bool:
        """
        Validate that the model directory contains required files.
        This is a best-effort check; the actual loading may still fail.
        """
        try:
            required_files = ["config.json", "tokenizer_config.json"]
            
            # Check for at least one model weights file
            model_files = ["pytorch_model.bin", "model.safetensors", "*.pt"]
            found_model_file = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
            
            # Check for at least one vocabulary file
            vocab_files = ["vocab.txt", "vocab.json", "tokenizer.json"]
            found_vocab_file = any(os.path.exists(os.path.join(model_path, f)) for f in vocab_files)
            
            # Check required config files
            config_valid = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
            
            return config_valid and found_model_file and found_vocab_file
            
        except Exception as e:
            logger.warning(f"Model validation check failed: {str(e)}")
            return False

    def _auto_detect_batch_size(self) -> int:
        """Dynamically determine safe batch size based on available hardware and model size."""
        if not torch.cuda.is_available() or not self._is_initialized or self.model is None:
            return 32  # Conservative default
        
        try:
            # Get current GPU memory state
            device = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            free_mem = total_mem - reserved_mem
            
            # Estimate memory requirements based on model configuration
            if hasattr(self.model, 'config'):
                hidden_size = getattr(self.model.config, 'hidden_size', 768)
                num_layers = getattr(self.model.config, 'num_hidden_layers', 12)
                
                # Rough estimation of memory per batch item
                # This is a simplified calculation and may need adjustment for specific architectures
                bytes_per_param = 2 if self.use_fp16 else 4
                est_mem_per_batch = (hidden_size * num_layers * bytes_per_param * 1024)  # 1024 sequence length factor
            else:
                # Fallback estimation
                est_mem_per_batch = 1.0 * (1024**3)  # 1.0GB per batch
            
            # Calculate safe batch size (using 60% of free memory for safety)
            safe_batch = int((free_mem * 0.6) / est_mem_per_batch)
            return max(1, min(safe_batch, 128))  # Clamp between 1 and 128
            
        except Exception as e:
            logger.warning(f"Batch size detection failed: {e}, using fallback 32")
            return 32

    async def embed(
        self,
        texts: List[str],
        batch_size: int = 0,
        normalize: bool = True,
        raise_on_error: bool = True,
    ) -> EmbeddingResponse:
        """
        Generate embeddings with automatic batch processing and comprehensive monitoring.
        """
        # Validate and clean input texts
        valid_texts = []
        original_indices = []
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                original_indices.append(i)
            else:
                logger.warning(f"Skipping invalid or empty text at index {i}")
        
        if not valid_texts:
            logger.warning("No valid texts provided for embedding")
            return self._empty_response()
        
        # Record request size for monitoring
        total_chars = sum(len(text) for text in valid_texts)
        self.REQUEST_SIZE_GAUGE.labels(model_name=self.model_name).set(total_chars)
        
        # Determine batch size
        effective_batch_size = batch_size if batch_size > 0 else self._batch_size
        
        try:
            with self.LATENCY_HIST.labels(model_name=self.model_name).time():
                return await self._process_batches(valid_texts, original_indices, effective_batch_size, normalize)
                
        except Exception as e:
            return self._handle_embed_error(e, effective_batch_size, raise_on_error)

    async def _process_batches(
        self,
        texts: List[str],
        original_indices: List[int],
        batch_size: int,
        normalize: bool
    ) -> EmbeddingResponse:
        """Process texts in batches and return embeddings."""
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_indices = original_indices[i:i + batch_size]
            
            try:
                embeddings, tokens = await self._process_single_batch(batch, normalize)
                all_embeddings.append((batch_indices, embeddings))
                total_tokens += tokens
                
                # Update memory gauge
                if torch.cuda.is_available():
                    # 使用当前 CUDA 设备，不需要传递 device 参数
                    mem_used = torch.cuda.memory_allocated() / (1024**2)  # MB
                    device_str = f"cuda:{torch.cuda.current_device()}"
                    self.MEMORY_GAUGE.labels(device_id=device_str).set(mem_used)
                    
            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}: {str(e)}")
                # Continue with remaining batches but log the error
                continue
        
        return self._build_response(all_embeddings, total_tokens, len(texts))

    async def _process_single_batch(
        self,
        batch: List[str],
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
        
        # Move inputs to correct device if not using device_map
        if not self._using_device_map and hasattr(self.model, 'device'):
            device = self.model.device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        
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

    def _empty_response(self) -> EmbeddingResponse:
        """Return an empty response with proper structure."""
        return EmbeddingResponse(
            data=[],
            model=self.model_name,
            object="list",
            usage={"prompt_tokens": 0, "total_tokens": 0}
        )
    
    def _build_response(
        self,
        all_embeddings: List[tuple[List[int], torch.Tensor]],
        total_tokens: int,
        total_texts: int
    ) -> EmbeddingResponse:
        """Build response ensuring each original input text has an embedding."""
        # Initialize empty array for all texts
        if not all_embeddings:
            return self._empty_response()

        # Create a list to hold all embeddings in original order
        final_embeddings: List[Optional[torch.Tensor]] = [None] * total_texts
        
        # Place each batch's embeddings in the correct positions
        for indices, embeddings in all_embeddings:
            for idx, embedding in zip(indices, embeddings):
                if idx < total_texts:
                    final_embeddings[idx] = embedding
        
        # Filter out None values (failed batches) and create response
        data = []
        valid_count = 0
        
        for idx, embedding in enumerate(final_embeddings):
            if embedding is not None:
                # Ensure embedding is 1D vector
                if embedding.dim() != 1:
                    logger.warning(f"Unexpected embedding dimension at index {idx}: {embedding.dim()}")
                    embedding = embedding.squeeze()
                    if embedding.dim() != 1:
                        # If still not 1D, use mean or skip
                        embedding = embedding.mean(dim=0)
                
                data.append(EmbeddingDataItem(
                    embedding=embedding.tolist(),
                    index=idx,
                    object="embedding"
                ))
                valid_count += 1
        
        logger.info(f"Successfully processed {valid_count}/{total_texts} texts")
        
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
        
        logger.error(f"Embedding error: {error_name} - {str(error)}")
        
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

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean token embeddings weighted by attention mask.
        Returns pooled embeddings of shape (batch_size, hidden_size)
        """
        # Input validation
        if token_embeddings.dim() != 3:
            raise ValueError(f"Token embeddings must be 3D, got {token_embeddings.dim()}D")
        if attention_mask.dim() != 2:
            raise ValueError(f"Attention mask must be 2D, got {attention_mask.dim()}D")
        
        # Ensure same device
        if token_embeddings.device != attention_mask.device:
            attention_mask = attention_mask.to(token_embeddings.device)
            
        # Handle empty input
        if token_embeddings.numel() == 0:
            return torch.zeros((0, token_embeddings.size(-1)), 
                             device=token_embeddings.device)

        # Expand mask and compute weighted mean
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        pooled = sum_embeddings / sum_mask
        
        # Ensure correct output shape
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
            
        return pooled

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

    async def shutdown(self) -> None:
        """Clean up resources safely and thoroughly."""
        if not self._is_initialized:
            return
            
        try:
            # Move model to CPU to free GPU memory
            if self.model is not None:
                if hasattr(self.model, "device") and str(self.model.device) != "cpu":
                    self.model.to("cpu")
                
                del self.model
                self.model = None
            
            # Cleanup tokenizer
            if self.tokenizer is not None:
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

    @property
    def embedding_dim(self) -> int:
        """Get the output dimension of embeddings."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model.config.hidden_size  # type: ignore

    @property
    def recommended_batch_size(self) -> int:
        """Get the currently recommended batch size."""
        return self._batch_size

    def health_check(self) -> Dict[str, Any]:
        """Check model health and resource status."""
        status = {
            "initialized": self._is_initialized,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "batch_size": self._batch_size,
            "model_name": self.model_name,
            "device": self.device,
            "using_device_map": self._using_device_map
        }
        
        if torch.cuda.is_available():
            device = self.model.device if self.model and hasattr(self.model, 'device') else "cuda:0"
            status.update({
                "gpu_memory_used_mb": torch.cuda.memory_allocated(device) / (1024**2), # type: ignore
                "gpu_memory_total_mb": torch.cuda.get_device_properties(device).total_memory / (1024**2), # type: ignore
                "device": str(device)
            })
        
        return status