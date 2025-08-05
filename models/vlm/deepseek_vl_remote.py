import aiohttp
import asyncio
import base64
from loguru import logger
from PIL import Image
from prometheus_client import Histogram, Counter
from typing import Any
from models.vlm.base import BaseVLM, RemoteVLMConfig

class DeepSeekVLCloud(BaseVLM):
    """
    Production-ready DeepSeek-VL cloud API implementation
    Features:
    - Compatible with DeepSeek official API
    - Automatic image encoding with validation
    - Retry mechanism with exponential backoff
    - Detailed Prometheus metrics
    """

    # Custom metrics (inherits base metrics from BaseVLM)
    API_LATENCY = Histogram(
        'deepseekvl_cloud_latency_seconds', 
        'API call latency for DeepSeek-VL',
        ['api_endpoint']
    )
    API_ERRORS = Counter(
        'deepseekvl_cloud_errors_total',
        'API error counts for DeepSeek-VL',
        ['error_code']
    )

    def __init__(self, config: RemoteVLMConfig):
        if not isinstance(config, RemoteVLMConfig):
            raise TypeError("config must be RemoteVLMConfig")

        self.config = config
        self.session: aiohttp.ClientSession | None = None
        self._endpoint = config.api_endpoint or "https://api.deepseek.com/v1/multimodal/chat"
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize async HTTP session with custom headers"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "X-API-Version": self.config.api_version
            }
        )
        self._is_initialized = True
        logger.debug("DeepSeek-VL session initialized")

    async def inference(
        self,
        text: str,
        image: str | Image.Image,
        **kwargs
    ) -> str | None:
        """
        Execute multimodal inference with:
        - Automatic image preprocessing
        - Configurable temperature and max_tokens
        - Built-in retry mechanism
        
        Args:
            text: Input text prompt
            image: Path to image or PIL Image object
            kwargs: Overrides for inference parameters
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: After max retries exceeded
            ValueError: For invalid image inputs
        """
        if not self.session:
            raise RuntimeError("Call startup() first")

        # Prepare payload with DeepSeek-specific format
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": await self._encode_image(image)},
                        {"type": "text", "text": text}
                    ]
                }
            ],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]}
        }

        # Execute with monitoring
        with self.API_LATENCY.labels(api_endpoint=self._endpoint).time():
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(
                        url=self._endpoint,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_response(data)
                        else:
                            error = await response.text()
                            self.API_ERRORS.labels(error_code=response.status).inc()
                            self.ERROR_COUNTER.labels(provider="deepseek").inc()
                            raise RuntimeError(f"API error {response.status}: {error[:200]}")
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        logger.error(f"DeepSeek-VL inference failed after {attempt+1} attempts: {str(e)}")
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _encode_image(self, image: str | Image.Image) -> str:
        """Convert image to DeepSeek-compatible base64 URL"""
        if isinstance(image, str):
            with open(image, "rb") as f:
                img_data = f.read()
        else:
            import io
            buf = io.BytesIO()
            image.save(buf, format="JPEG" if image.mode == "RGB" else "PNG")
            img_data = buf.getvalue()

        if len(img_data) > 10 * 1024 * 1024:  # DeepSeek's 10MB limit
            raise ValueError("Image size exceeds 10MB limit")

        return f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"

    def _parse_response(self, data: dict) -> str:
        """Extract text from DeepSeek's response structure"""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Malformed API response: {data}")
            raise RuntimeError("Failed to parse API response") from e

    async def shutdown(self) -> None:
        """Cleanup resources gracefully"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            self._is_initialized = False
            logger.debug("DeepSeek-VL session closed")

    async def health_check(self) -> dict[str, Any]:
        """Check service health status."""
        return {
            "initialized": self._is_initialized,
            "model": self.config.model_name,
            "last_error": None,
            "throughput": "N/A"  # Could track actual metrics
        }