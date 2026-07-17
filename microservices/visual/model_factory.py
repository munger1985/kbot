"""视觉嵌入模型工厂。"""

from loguru import logger
from .model import VisualModelConfig, BaseVisualEmbedding
from core.dictionary import VisualEmbeddingProvider


class ColQwen2Embedding(BaseVisualEmbedding):
    """ColQwen2 视觉嵌入模型"""

    def __init__(self, config: VisualModelConfig):
        super().__init__(config)
        self._model = None
        self._processor = None

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
        import base64, io, torch
        from PIL import Image

        # 去除 data URI 前缀
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_data = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        import asyncio
        loop = asyncio.get_running_loop()

        def _run():
            inputs = self._processor.process_images([pil_image])
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            emb = outputs.mean(dim=1).squeeze(0)
            emb = emb / emb.norm()
            return emb.cpu().float().tolist()

        return await loop.run_in_executor(None, _run)


def create_visual_model(config: VisualModelConfig) -> BaseVisualEmbedding:
    """根据 provider 创建视觉嵌入模型实例"""
    provider = config.provider

    if provider == VisualEmbeddingProvider.COLQWEN2.value:
        return ColQwen2Embedding(config)

    raise ValueError(f"不支持的视觉嵌入 Provider: {provider}")
