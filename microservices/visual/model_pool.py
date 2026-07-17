"""视觉嵌入模型池。"""

from pathlib import Path
from core.config.settings import get_visual_config
from core.dictionary import ModelCategory, EmbeddingProvider
from loguru import logger
from typing import Any
from microservices.common.model_pool import BaseModelPool
from .model import *
from .model_factory import create_visual_model


class VisualModelPool(BaseModelPool[BaseVisualEmbedding]):
    """视觉嵌入模型池 — 管理 ColQwen2 等视觉模型的生命周期"""

    def _get_model_category(self) -> int:
        return ModelCategory.IMG_EMBEDDING.value

    async def _shutdown_model_instance(self, model: BaseVisualEmbedding) -> None:
        await model.shutdown()

    async def _perform_model_health_check(self, model_name: str, model: BaseVisualEmbedding) -> None:
        # 用一个小 blank 图片做健康检查
        blank_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        await model.embed(blank_b64)
        logger.debug(f"[VisualModel] {model_name} health check OK")

    async def _start_model(self, model_name: str, model_data: dict[str, Any]) -> BaseVisualEmbedding:
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"视觉模型 {model_name} 缺少 provider")

        global_config = get_visual_config()
        model_config = self._build_config(model_name, provider, model_data, global_config)

        model = create_visual_model(model_config)
        await model.startup()
        logger.success(f"Visual model {model_name} ({provider}) started")
        return model

    def _build_config(self, name: str, provider: str, data: dict[str, Any], global_cfg: Any) -> VisualModelConfig:
        """将数据库数据映射为特定 provider 的 Pydantic Config"""
        params = data.get("model_params", {})
        model_tech_name = data.get("model_tech_name", name)
        path = data.get("model_path") or ""
        path = str(Path(path).expanduser()) if path else None

        common_kwargs = {
            "model_name": model_tech_name,
            "provider": provider,
            "model_path": path,
            "device": params.get("device", "cuda"),
            "dimension": params.get("dimension", 128),
            "timeout": params.get("timeout", global_cfg.timeout),
            "max_retries": params.get("max_retries", global_cfg.max_retries),
        }

        if provider == EmbeddingProvider.LOCAL_QWEN.value:
            if not path:
                raise ValueError(f"视觉模型 {name} 缺少 model_path")
            return ColQwen2EmbeddingConfig(**common_kwargs)

        raise ValueError(f"不支持的视觉模型 Provider: {provider}")
