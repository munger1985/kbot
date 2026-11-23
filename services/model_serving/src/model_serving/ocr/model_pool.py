"""OCR 模型生命周期池。"""

from typing import Any

from model_serving.common.model_pool import BaseModelPool
from platform_core.dictionary import ModelCategory

from .model import OCRModel


class OCRModelPool(BaseModelPool[OCRModel]):
    """按 model_id 隔离和复用 OCR 引擎实例。"""

    def _get_model_category(self) -> int:
        return ModelCategory.OCR.value

    async def _start_model(
        self, model_id: str, model_data: dict[str, Any],
    ) -> OCRModel:
        model = OCRModel(model_data=model_data)
        await model.startup()
        return model

    async def _shutdown_model_instance(self, model: OCRModel) -> None:
        await model.shutdown()

    async def _perform_model_health_check(
        self, model_id: str, model: OCRModel,
    ) -> None:
        await model.health_check()
