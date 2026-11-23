"""ColQwen2 必须完整、离线加载本地基础权重和 adapter。"""

import asyncio
from base64 import b64encode
import json
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from model_serving.visual.model.colqwen2_local import ColQwen2Embedding, ColQwen2EmbeddingConfig
from model_serving.visual.model.native_representation import VisualImageInputUnsupported


def test_resolves_local_base_model_and_rejects_remote_reference(tmp_path):
    adapter = tmp_path / "adapter"
    base = tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    (adapter / "adapter_model.safetensors").touch()
    config = adapter / "adapter_config.json"
    config.write_text(json.dumps({"base_model_name_or_path": str(base)}), encoding="utf-8")
    assert ColQwen2Embedding._resolve_paths(str(adapter)) == (adapter, base)

    config.write_text(json.dumps({"base_model_name_or_path": "vidore/colqwen2-base"}), encoding="utf-8")
    with pytest.raises(ValueError, match="绝对本地路径"):
        ColQwen2Embedding._resolve_paths(str(adapter))


def test_remaps_layer_weights_without_changing_projection():
    layer_value, projection_value = object(), object()
    params = ColQwen2Embedding._remap_adapter_state_dict({
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": layer_value,
        "base_model.model.custom_text_proj.lora_A.weight": projection_value,
    })
    assert params == {
        "base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": layer_value,
        "base_model.model.custom_text_proj.lora_A.weight": projection_value,
    }


def test_rejects_incomplete_checkpoint():
    with pytest.raises(RuntimeError, match="权重未完整加载"):
        ColQwen2Embedding._require_complete_weights(
            {"missing_keys": ["language_model.norm.weight"], "unexpected_keys": []},
            "基础",
        )


def _encoded_image(size, *, animated=False):
    buffer = BytesIO()
    with Image.new("RGB", size) as first, Image.new("RGB", size, "red") as second:
        first.save(buffer, "PNG", save_all=animated, append_images=[second] if animated else [])
    return b64encode(buffer.getvalue()).decode("ascii")


def _visual_model(**limits):
    model = ColQwen2Embedding(ColQwen2EmbeddingConfig(model_name="test", provider="local_qwen",
        model_revision="model-revision", processor_revision="processor-revision", input_template="image",
        retrieval_directions=("IMAGE_TO_IMAGE",), **limits))
    model._processor = model._model = object()
    model._run_cpu_inference = AsyncMock(return_value={"mock": "native-inference"})
    return model


@pytest.mark.parametrize("size,animated,limits,reason", [
    ((1400, 3), False, {}, "ASPECT_RATIO"),
    ((3, 1400), False, {}, "ASPECT_RATIO"),
    ((40, 40), False, {"max_image_pixels": 1024}, "PIXEL_LIMIT"),
    ((40, 40), True, {}, "FRAME_COUNT"),
    ((40, 40), False, {"max_image_bytes": 4}, "BYTE_LIMIT"),
])
def test_colqwen_rejects_unsupported_single_images_before_inference(size, animated, limits, reason):
    model = _visual_model(**limits)
    with pytest.raises(VisualImageInputUnsupported, match=reason):
        asyncio.run(model.represent(input_kind="IMAGE", direction="IMAGE_TO_IMAGE",
                                     image_base64=_encoded_image(size, animated=animated)))
    model._run_cpu_inference.assert_not_awaited()


@pytest.mark.parametrize("size", [(640, 288), (400, 2)])
def test_colqwen_permitted_images_still_reach_inference(size):
    model = _visual_model()
    assert asyncio.run(model.represent(input_kind="IMAGE", direction="IMAGE_TO_IMAGE",
                                       image_base64=_encoded_image(size))) == {"mock": "native-inference"}
    model._run_cpu_inference.assert_awaited_once()


def test_visual_input_errors_are_not_a_blanket_model_failure_handler():
    model = _visual_model()
    model._run_cpu_inference.side_effect = ValueError("unrelated model failure")
    with pytest.raises(ValueError, match="unrelated model failure") as error:
        asyncio.run(model.represent(input_kind="IMAGE", direction="IMAGE_TO_IMAGE",
                                    image_base64=_encoded_image((40, 40))))
    assert type(error.value) is ValueError
    with pytest.raises(ValueError) as error:
        asyncio.run(model.represent(input_kind="IMAGE", direction="IMAGE_TO_IMAGE",
                                    image_base64="invalid!"))
    assert not isinstance(error.value, VisualImageInputUnsupported)
