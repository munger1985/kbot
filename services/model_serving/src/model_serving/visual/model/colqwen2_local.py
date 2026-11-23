"""从本地基础权重与 Vidore adapter 加载 ColQwen2。"""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger
from .base import BaseVisualEmbedding, VisualModelConfig
from .native_representation import NativeColQwenRepresentation


_BASE_KEY_MAPPING = {
    r"^base_model\.model\.custom_text_proj": "custom_text_proj",
    r"^model\.layers": "language_model.layers",
    r"^model\.embed_tokens": "language_model.embed_tokens",
    r"^model\.norm": "language_model.norm",
}
_ADAPTER_LAYER_PREFIX = "base_model.model.model.layers."
_RUNTIME_ADAPTER_LAYER_PREFIX = "base_model.model.language_model.layers."


class ColQwen2EmbeddingConfig(VisualModelConfig):
    """ColQwen2 嵌入模型配置（继承基类配置）"""
    pass


class ColQwen2Embedding(NativeColQwenRepresentation, BaseVisualEmbedding):
    """ColQwen2 视觉嵌入模型"""

    def __init__(self, config: ColQwen2EmbeddingConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._inference_lock = asyncio.Lock()

    @staticmethod
    def _resolve_paths(model_path: str | None) -> tuple[Path, Path]:
        if not model_path:
            raise ValueError("ColQwen2 缺少本地 adapter 路径")
        adapter_path = Path(model_path).expanduser().resolve()
        config_path = adapter_path / "adapter_config.json"
        if not config_path.is_file() or not (adapter_path / "adapter_model.safetensors").is_file():
            raise ValueError("ColQwen2 adapter 文件不完整")
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        base_path = Path(str(metadata.get("base_model_name_or_path") or "").strip()).expanduser()
        if not base_path.is_absolute() or not base_path.is_dir():
            raise ValueError("ColQwen2 基础权重必须指向可用的绝对本地路径")
        return adapter_path, base_path.resolve()

    @staticmethod
    def _remap_adapter_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        """把 Vidore adapter 的层名映射到 Transformers 5 Qwen2-VL 模型。"""
        remapped: dict[str, Any] = {}
        found_layers = False
        for key, value in state_dict.items():
            mapped_key = key
            if key.startswith(_ADAPTER_LAYER_PREFIX):
                mapped_key = _RUNTIME_ADAPTER_LAYER_PREFIX + key[len(_ADAPTER_LAYER_PREFIX):]
                found_layers = True
            if mapped_key in remapped:
                raise ValueError(f"ColQwen2 adapter 权重键冲突：{mapped_key}")
            remapped[mapped_key] = value
        if not found_layers:
            raise ValueError("ColQwen2 adapter 不符合当前 Vidore 权重合同")
        return remapped

    @staticmethod
    def _require_complete_weights(loading_info: dict[str, Any], artifact: str) -> None:
        counts = {
            key: len(loading_info.get(key) or ())
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        }
        if any(counts.values()):
            raise RuntimeError(f"ColQwen2 {artifact}权重未完整加载：{counts}")

    async def startup(self):
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        from safetensors.torch import load_file

        adapter, base = self._resolve_paths(self.config.model_path)
        model, base_info = ColQwen2.from_pretrained(
            str(base),
            torch_dtype=torch.bfloat16,
            device_map=self.config.device if torch.cuda.is_available() else "cpu",
            key_mapping=_BASE_KEY_MAPPING,
            local_files_only=True,
            output_loading_info=True,
        )
        self._require_complete_weights(base_info, "基础")
        state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
        adapter_info = model.load_adapter(
            str(adapter), adapter_state_dict=self._remap_adapter_state_dict(state),
        )
        self._require_complete_weights(
            {
                "missing_keys": adapter_info.missing_keys,
                "unexpected_keys": adapter_info.unexpected_keys,
            },
            "adapter ",
        )
        processor = ColQwen2Processor.from_pretrained(str(adapter), local_files_only=True)
        self._model = model.eval()
        self._processor = processor
        logger.success("ColQwen2 本地权重加载完成：{}", adapter)

    async def shutdown(self):
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None

    async def embed(self, image_base64: str) -> list[float]:
        raise ValueError("ColQwen2 仅提供原生多向量；请使用 represent 接口并配置检索方向")
