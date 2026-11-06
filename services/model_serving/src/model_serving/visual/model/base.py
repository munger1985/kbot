"""视觉嵌入模型配置与基类。"""

from pydantic import BaseModel


class VisualModelConfig(BaseModel):
    """视觉嵌入模型配置"""
    model_name: str
    provider: str
    model_path: str | None = None
    device: str = "cuda"
    dimension: int = 128
    timeout: int = 300
    max_retries: int = 3
    model_revision: str | None = None
    processor_revision: str | None = None
    input_template: str | None = None
    retrieval_directions: tuple[str, ...] = ()
    max_image_bytes: int = 8388608
    max_image_pixels: int = 25000000
    max_query_chars: int = 16000
    max_representation_elements: int = 2097152


class BaseVisualEmbedding:
    """视觉嵌入模型基类"""

    def __init__(self, config: VisualModelConfig):
        self.config = config

    async def startup(self):
        raise NotImplementedError

    async def shutdown(self):
        raise NotImplementedError

    async def embed(self, image_base64: str) -> list[float]:
        raise NotImplementedError
