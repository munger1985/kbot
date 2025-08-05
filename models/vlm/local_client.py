import os
import re
import gc
import torch
from typing import Any, Literal
from loguru import logger
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.utils.quantization_config import BitsAndBytesConfig
from prometheus_client import Histogram, Counter, Gauge
from models.vlm.base import BaseVLM, LocalVLMConfig
from core.config import settings


class LocalVL(BaseVLM):
    """
    Universal local deployment for Vision-Language Models (VLMs) including:
    - Qwen-VL
    - DeepSeek-VL
    - Other compatible VLMs
    
    Features:
    - Comprehensive configuration management
    - Advanced error handling
    - Resource optimization
    - Detailed monitoring
    """
    # 模型识别规则
    MODEL_PATTERNS = {
        "qwen-vl": [r"Qwen-VL", r"qwen-vl", r"Qwen/VL"],
        "deepseek-vl": [r"deepseek-vl", r"deepseek/vl"]
    }

    # Prometheus metrics
    LATENCY_HIST = Histogram(
        'local_vl_latency_seconds',
        'Latency for VLM inference',
        ['model_type', 'quantization']
    )
    
    ERROR_COUNTER = Counter(
        'local_vl_errors_total',
        'Count of VLM errors',
        ['model_type', 'error_type']
    )
    
    MEMORY_GAUGE = Gauge(
        'local_vl_memory_usage_mb',
        'GPU memory usage in MB',
        ['model_type', 'device_id']
    )
    
    INPUT_SIZE_GAUGE = Gauge(
        'local_vl_input_size',
        'Size of input data (text length + image resolution)',
        ['model_type', 'input_type']
    )

    SUPPORTED_MODELS = Literal["qwen-vl", "deepseek-vl"]

    def __init__(self, config: LocalVLMConfig):
        """
        Initialize with validated configuration.
        
        Args:
            config: LocalVLMConfig containing model configuration
            model_type: Type of VLM model ("qwen-vl" or "deepseek-vl")
        """
        if not isinstance(config, LocalVLMConfig):
            raise TypeError("config must be LocalVLMConfig")

        # Model components
        self.model: torch.nn.Module | None = None
        self.processor: Any | None = None
        self.model_type = self._detect_model_type(config.model_path or config.model_name)
        
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
        self.local_files_only = getattr(config, 'local_files_only', False)
        self.trust_remote_code = getattr(config, 'trust_remote_code', False)
        self.quantization = getattr(config, 'quantization', None)
        self.use_fp16 = getattr(config, 'use_fp16', False)

        # Configuration
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.cache_path = os.path.join("./models/local_model_cache", self.model_name)
        self.name_or_path = ""
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize model with full error handling."""
        if self._is_initialized:
            logger.warning("Model already initialized")
            return

        try:
            # Validate model files
            if not self._validate_model_files():
                raise ValueError("Invalid model files")

            # Device setup
            if not self.device:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path or self.model_name,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only
            )

            # Model loading
            load_kwargs = self._get_load_kwargs()
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            
            # Optimization
            self._optimize_model()
            
            # Cache model if downloaded
            if not self.local_files_only and not self.model_path:
                self._cache_model()

            self._is_initialized = True
            logger.success(f"{self.model_type.upper()} initialized on {self.device}")
            
        except Exception as e:
            self._handle_error(e, "startup")
            raise

    def _get_load_kwargs(self) -> dict[str, Any]:
        """Generate kwargs for model loading."""
        kwargs = {
            "pretrained_model_name_or_path": self.model_path or self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }

        # Device configuration
        if torch.cuda.is_available():
            if self.device_map:
                kwargs.update({
                    "device_map": self.device_map,
                    "max_memory": self.max_memory,
                })
            else:
                kwargs["device"] = self.device or "cuda:0"
            
            # Precision
            if self.use_fp16:
                kwargs["torch_dtype"] = torch.float16
            elif self.quantization:
                kwargs.update(self._get_quantization_config())
        else:
            kwargs["device_map"] = "cpu"
            
        return kwargs

    def _get_quantization_config(self) -> dict[str, Any]:
        """Generate quantization config."""
        if not self.quantization:
            return {}

        try:
            if self.quantization == "4bit":
                return {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                }
            elif self.quantization == "8bit":
                return {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                }
        except ImportError:
            logger.warning("bitsandbytes not available, quantization disabled")
        return {}

    def _optimize_model(self) -> None:
        """Apply model optimizations."""
        if self.model is None:
            return

        self.model.eval()
        
        # Model compilation
        if self.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)  # type: ignore
                logger.debug("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

    def _validate_model_files(self) -> bool:
        """Validate required model files exist."""
        # Common required files for all models
        common_files = [
            "config.json",
            "tokenizer_config.json",
            "vocab.json"
        ]
        
        # Model-specific files
        model_specific_files = {
            "qwen-vl": ["configuration_qwen.py", "modeling_qwen.py"],
            "deepseek-vl": ["configuration_deepseek.py", "modeling_deepseek.py"]
        }.get(self.model_type, [])
        
        required_files = common_files + model_specific_files
        
        check_path = self.model_path or self.model_name
        if self.model_path:
            return all(os.path.exists(os.path.join(check_path, f)) for f in required_files)
        return True  # Remote model assumed valid

    def _cache_model(self) -> None:
        """Cache downloaded model locally."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        os.makedirs(self.cache_path, exist_ok=True)
        try:
            self.model.save_pretrained(self.cache_path)  # type: ignore
            self.processor.save_pretrained(self.cache_path)
            logger.info(f"Model cached at {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache model: {e}")
            raise

    async def inference(
        self, 
        text: str, 
        image: str | Image.Image,
        **kwargs
    ) -> str:
        """
        Generate response from image and text input.
        
        Args:
            text: Input text prompt
            image: Image path or PIL Image
            **kwargs: Generation parameters (temperature, top_p, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If model not initialized or inference fails
        """
        # Validate inputs
        if not self._is_initialized:
            raise RuntimeError("Model not initialized")
            
        if not text or image is None:
            raise ValueError("Both text and image must be provided")

        # Load image if path provided
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                self._handle_error(e, "image_load")
                raise

        # Record input size metrics
        self._record_input_metrics(text, image)

        try:
            with self.LATENCY_HIST.labels(
                model_type=self.model_type,
                quantization=self.quantization or "none"
            ).time():
                return await self._run_inference(text, image, **kwargs)
        except Exception as e:
            self._handle_error(e, "inference")
            raise

    async def _run_inference(
        self,
        text: str,
        image: Image.Image,
        **kwargs
    ) -> str:
        """Actual inference logic."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model components not loaded")

        # Prepare inputs
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            **kwargs
        }

        # Inference
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)  # type: ignore

        # Decode output
        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

    def _record_input_metrics(self, text: str, image: Image.Image) -> None:
        """Record metrics about input size."""
        self.INPUT_SIZE_GAUGE.labels(
            model_type=self.model_type,
            input_type="text"
        ).set(len(text))
        
        self.INPUT_SIZE_GAUGE.labels(
            model_type=self.model_type,
            input_type="image"
        ).set(image.size[0] * image.size[1])
        
        if torch.cuda.is_available():
            device = self.model.device if self.model else "cuda:0"  # type: ignore
            self.MEMORY_GAUGE.labels(
                model_type=self.model_type,
                device_id=str(device)
            ).set(torch.cuda.memory_allocated(device) / (1024 ** 2))  # type: ignore

    def _handle_error(self, error: Exception, context: str) -> None:
        """Log and record errors."""
        error_name = type(error).__name__
        self.ERROR_COUNTER.labels(
            model_type=self.model_type,
            error_type=error_name
        ).inc()
        logger.error(f"{self.model_type} {context} error: {error}")
        
        # Special handling for CUDA OOM
        if isinstance(error, torch.cuda.OutOfMemoryError):
            logger.critical("CUDA OOM - Try reducing input size or enable quantization")

    async def shutdown(self) -> None:
        """Clean up resources safely."""
        if not self._is_initialized:
            return

        try:
            # Move model to CPU first
            if self.model is not None:
                if hasattr(self.model, "device") and str(self.model.device) != "cpu":
                    self.model.to("cpu")
                del self.model
                self.model = None

            # Clean processor
            if self.processor is not None:
                del self.processor
                self.processor = None

            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_initialized = False
            logger.info("Resources released")
            
        except Exception as e:
            self._handle_error(e, "shutdown")
            raise

    async def warmup(self, sample_text: str = "Describe this image") -> None:
        """Warm up the model for first-time inference."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized")

        logger.info(f"Warming up {self.model_type.upper()}...")
        try:
            # Use blank image for warmup
            blank_image = Image.new("RGB", (224, 224), (255, 255, 255))
            await self.inference(sample_text, blank_image)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def health_check(self) -> dict[str, Any]:
        """Get current system health status."""
        status = {
            "model_type": self.model_type,
            "initialized": self._is_initialized,
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "device": self.device,
            "quantization": self.quantization
        }
        
        if torch.cuda.is_available():
            device = self.model.device if self.model else "cuda:0"  # type: ignore
            status.update({
                "gpu_memory_used_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),  # type: ignore
                "gpu_memory_total_mb": torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # type: ignore
            })
            
        return status

    def model_info(self) -> dict[str, Any]:
        """Get detailed model information."""
        if not self._is_initialized or self.model is None:
            raise RuntimeError("Model not initialized")

        info = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "max_tokens": self.max_tokens,
            "device": str(self.model.device) if hasattr(self.model, "device") else str(self.device_map),
            "dtype": str(self.model.dtype) if hasattr(self.model, "dtype") else "unknown",
            "quantization": self.quantization
        }
        
        if hasattr(self.model, "config"):
            config = self.model.config.to_dict()  # type: ignore
            info.update({
                "vocab_size": config.get("vocab_size", "unknown"),
                "hidden_size": config.get("hidden_size", "unknown"),
                "num_attention_heads": config.get("num_attention_heads", "unknown"),
                "num_hidden_layers": config.get("num_hidden_layers", "unknown")
            })
            
        return info
    
    def _detect_model_type(self, model_identifier: str) -> str:
        """
        Auto-detect model type from path/name
        
        Args:
            model_identifier: model_path or model_name
            
        Returns:
            Detected model type (e.g. "qwen-vl")
            
        Raises:
            ValueError: If model type cannot be determined
        """
        if not model_identifier:
            raise ValueError("Cannot detect model type from empty identifier")
            
        model_lower = model_identifier.lower()
        
        for model_type, patterns in self.MODEL_PATTERNS.items():
            if any(re.search(pattern, model_lower) for pattern in patterns):
                return model_type
                
        raise ValueError(
            f"Unsupported model: {model_identifier}. "
            f"Supported patterns: {self.MODEL_PATTERNS}"
        )
    