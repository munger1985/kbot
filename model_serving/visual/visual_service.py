"""视觉嵌入服务 — 通过模型池管理 ColQwen2 等视觉模型。

模型配置从数据库 model 表读取（category=IMG_EMBEDDING），
服务级配置从 TOML [visual] 段读取。
"""

from typing import Callable
from loguru import logger
from .model_pool import VisualModelPool


class VisualService:
    """视觉嵌入服务 — 模型生命周期 + 推理"""

    def __init__(self):
        self._pool = VisualModelPool()
        self._initialized = False

    def bind_session_factory(self, session_factory: Callable) -> None:
        self._pool.set_session_factory(session_factory)

    async def initialize(self):
        """初始化模型池，加载数据库中所有激活的视觉模型"""
        if self._initialized:
            return

        await self._pool.initialize()
        self._initialized = True
        logger.success("[VisualService] initialized")

    async def shutdown(self):
        if self._initialized:
            await self._pool.shutdown()
            self._initialized = False
            logger.info("[VisualService] shutdown complete")

    async def embed(self, model_name: str, image_base64: str) -> list[float]:
        """图片 → 视觉 embedding"""
        if not self._initialized:
            await self.initialize()

        model = await self._pool.load_model(model_name)
        return await model.embed(image_base64)
