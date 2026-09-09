"""对话图片文字提取的 OCR 业务服务。"""

import base64
import binascii
from typing import Any, Callable
from uuid import UUID

from .model import OCRModel
from .model_pool import OCRModelPool


class OCRService:
    """以共享模型池管理 OCR 模型的加载、失效和推理。"""

    def __init__(self, *, uow_factory: Callable | None = None):
        self._model_pool = OCRModelPool()
        self._uow_factory = uow_factory
        self._initialized = False

    def bind_session_factory(self, session_factory: Callable) -> None:
        self._model_pool.set_session_factory(session_factory)

    def bind_uow_factory(self, uow_factory: Callable) -> None:
        self._uow_factory = uow_factory

    async def initialize(self) -> None:
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True

    async def warmup(self) -> None:
        if not self._initialized:
            await self.initialize()
        await self._model_pool.warmup()

    async def shutdown(self) -> None:
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False

    async def invalidate_model(self, served_model_name: str) -> None:
        if self._initialized:
            await self._model_pool.unload_model(served_model_name)

    def is_model_loaded(self, served_model_name: str) -> bool:
        return self._model_pool.is_model_loaded(served_model_name)

    async def infer(
        self, *, model_id: UUID, image_base64: str
    ) -> dict[str, Any]:
        try:
            image = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_base64 无效") from exc
        if not image:
            raise ValueError("图片内容为空")
        model_data = await self._model_definition(model_id)
        if not self._initialized:
            await self.initialize()
        model: OCRModel = await self._model_pool.load_model(
            model_data["served_model_name"],
        )
        text, blocks = await model.infer(image)
        return {
            "model_id": model_id,
            "provider": model.provider,
            "text": text,
            "blocks": blocks,
            "model_revision": model.revision,
        }

    async def _model_definition(self, model_id: UUID) -> dict[str, Any]:
        if self._uow_factory is None:
            raise RuntimeError("OCR 服务未配置模型目录事务工厂")
        async with self._uow_factory() as uow:
            from platform_core.dictionary import ModelCategory, Status

            assert uow.models is not None
            model = await uow.models.get_by_id(model_id)
            if (
                int(model.category) != ModelCategory.OCR.value
                or int(model.status) != Status.ENABLED.value
            ):
                raise LookupError("OCR 模型不存在或未启用")
            return self._model_pool._map_entity_to_dict(model)
