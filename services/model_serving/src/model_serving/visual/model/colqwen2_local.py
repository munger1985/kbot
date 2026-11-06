"""ColQwen2 视觉嵌入模型"""

from loguru import logger
from .base import BaseVisualEmbedding, VisualModelConfig
from .native_representation import NativeColQwenRepresentation


class ColQwen2EmbeddingConfig(VisualModelConfig):
    """ColQwen2 嵌入模型配置（继承基类配置）"""
    pass


class ColQwen2Embedding(NativeColQwenRepresentation, BaseVisualEmbedding):
    """ColQwen2 视觉嵌入模型"""

    def __init__(self, config: ColQwen2EmbeddingConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        import asyncio
        self._inference_lock = asyncio.Lock()

    async def startup(self):
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self._model = ColQwen2.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device if torch.cuda.is_available() else "cpu",
        ).eval()
        self._processor = ColQwen2Processor.from_pretrained(self.config.model_path)
        logger.success(f"ColQwen2 loaded: {self.config.model_path}")

    async def shutdown(self):
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None

    async def embed(self, image_base64: str) -> list[float]:
        raise ValueError("ColQwen2 仅提供原生多向量；请使用 represent 接口并配置检索方向")
