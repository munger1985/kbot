"""与 Docling 解耦、可共享模型文件路径的 OCR 运行时。"""

import asyncio
import base64
import binascii
import json
from typing import Any
from uuid import UUID

from platform_core.dictionary import ModelCategory, OCRProvider


class OCRService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory
        self._engines: dict[UUID, object] = {}

    async def infer(
        self, *, model_id: UUID, image_base64: str
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            assert uow.models is not None
            model = await uow.models.get_by_id(model_id)
            if (
                int(model.category) != ModelCategory.OCR.value
                or int(model.status) != 1
            ):
                raise LookupError("OCR 模型不存在或未激活")
            provider = str(model.provider)
            params = dict(model.model_params or {})
            revision = str(
                params.get("revision") or model.provider_model_name
            )
        try:
            image = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_base64 无效") from exc
        if not image:
            raise ValueError("图片内容为空")
        if provider == OCRProvider.DOCLING.value:
            result = await asyncio.to_thread(
                self._run_rapidocr, model_id, image, params
            )
        elif provider == OCRProvider.DEEPSEEK_OCR.value:
            raise RuntimeError("DeepSeek OCR Provider 尚未配置推理适配器")
        else:
            raise ValueError("不支持的 OCR Provider")
        raw_blocks = result.to_json() or []
        blocks = (
            json.loads(raw_blocks)
            if isinstance(raw_blocks, str)
            else raw_blocks
        )
        if isinstance(blocks, dict):
            blocks = blocks.get("res") or blocks.get("result") or [blocks]
        return {
            "model_id": model_id,
            "provider": provider,
            "text": "\n".join(result.txts or ()),
            "blocks": list(blocks),
            "model_revision": revision,
        }

    def is_model_loaded(self, served_model_name: str) -> bool:
        """OCR 引擎按模型 UUID 缓存，目录名称不作为缓存键。"""
        return False

    async def invalidate_model(self, served_model_name: str) -> None:
        """目录变化时清空轻量引擎缓存，避免继续使用旧参数。"""
        self._engines.clear()

    def _run_rapidocr(self, model_id: UUID, image: bytes, params: dict):
        from rapidocr import RapidOCR

        engine = self._engines.get(model_id)
        if engine is None:
            engine = RapidOCR(
                config_path=params.get("config_path"),
                params=params.get("rapidocr_params"),
            )
            self._engines[model_id] = engine
        return engine(image)
