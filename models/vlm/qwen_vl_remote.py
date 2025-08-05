import aiohttp
import asyncio
import base64
from loguru import logger
from typing import Any
from PIL import Image
from prometheus_client import Histogram, Counter
from models.vlm.base import BaseVLM, RemoteVLMConfig

class QwenVLCloud(BaseVLM):
    """
    Production-ready Qwen-VL cloud API implementation
    Supports: 
    - AliCloud DashScope API
    - Azure Vision API (compatible)
    """

    # Metrics
    API_LATENCY = Histogram(
        'qwenvl_cloud_latency_seconds', 
        'API call latency',
        ['api_endpoint']
    )
    API_ERRORS = Counter(
        'qwenvl_cloud_errors_total',
        'API error counts',
        ['error_code']
    )

    def __init__(self, config: RemoteVLMConfig):
        if not isinstance(config, RemoteVLMConfig):
            raise TypeError("config must be RemoteVLMConfig")

        self.config = config
        self.session = None
        self._endpoint = config.api_endpoint or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize async HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        self._is_initialized = True
        logger.debug("Qwen-VL session initialized")

    async def inference(
        self,
        text: str,
        image: str | Image.Image,
        **kwargs
    ) -> str | None:
        """
        Execute cloud inference with:
        - Automatic image encoding
        - Retry mechanism
        - Detailed monitoring
        """
        if not self.session:
            raise RuntimeError("Call startup() first")

        # Prepare payload
        payload = {
            "model": self.config.model_name,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"image": await self._encode_image(image)},
                        {"text": text}
                    ]
                }]
            },
            "parameters": {
                "max_tokens": self.config.max_tokens,
                "temperature": kwargs.get("temperature", self.config.temperature)
            }
        }

        # Execute with metrics
        with self.API_LATENCY.labels(api_endpoint=self._endpoint).time():
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(
                        url=self._endpoint,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["output"]["choices"][0]["message"]["content"]
                        else:
                            error = await response.text()
                            self.API_ERRORS.labels(error_code=response.status).inc()
                            raise RuntimeError(f"API error {response.status}: {error}")
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        logger.error(f"Final attempt failed: {str(e)}")
                        raise
                    await asyncio.sleep(1 * (attempt + 1))

    async def _encode_image(self, image: str | Image.Image) -> str:
        """Convert image to base64 with validation"""
        if isinstance(image, str):
            with open(image, "rb") as f:
                img_data = f.read()
        else:
            import io
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            img_data = buf.getvalue()

        if len(img_data) > 20 * 1024 * 1024:  # 20MB limit
            raise ValueError("Image size exceeds 20MB limit")

        return base64.b64encode(img_data).decode('utf-8')

    async def shutdown(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
            self._is_initialized = False
        logger.debug("Qwen-VL session shutdown")

    async def health_check(self) -> dict[str, Any]:
        """Check service health status."""
        return {
            "initialized": self._is_initialized,
            "model": self.config.model_name,
            "last_error": None,
            "throughput": "N/A"  # Could track actual metrics
        }