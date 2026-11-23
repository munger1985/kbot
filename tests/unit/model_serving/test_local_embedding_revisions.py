"""本地 Embedding 的绑定和推理使用实际制品指纹及输入规则。"""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from model_serving.common.frozen_binding import FrozenModelConflict
from model_serving.common import local_embedding_templates
from model_serving.common.local_embedding_revisions import local_embedding_revisions, with_local_embedding_metadata
from model_serving.embedding import model_pool as embedding_pool
from platform_core.model_representation import registered_image_search_directions, representation_metadata


def checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic weight")
    (path / "tokenizer.json").write_text("{}")
    return path


def test_revisions_change_independently_on_file_replacement(tmp_path):
    source = checkpoint(tmp_path / "source")
    before = local_embedding_revisions(str(source))
    assert local_embedding_revisions(str(checkpoint(tmp_path / "copied"))) == before
    (source / "tokenizer.json").write_text('{"tokenizer": 2}')
    tokenizer_changed = local_embedding_revisions(str(source))
    assert tokenizer_changed["processor_revision"] != before["processor_revision"]
    assert tokenizer_changed["model_revision"] == before["model_revision"]
    (source / "model.safetensors").write_bytes(b"other synthetic weight")
    weights_changed = local_embedding_revisions(str(source))
    assert weights_changed["model_revision"] != tokenizer_changed["model_revision"]
    assert weights_changed["processor_revision"] == tokenizer_changed["processor_revision"]


def test_visual_versions_cover_base_adapter_and_its_processor(tmp_path):
    base = checkpoint(tmp_path / "base")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "preprocessor_config.json").write_text("{}")
    original = local_embedding_revisions(str(adapter), visual=True)
    (base / "model.safetensors").write_bytes(b"different base weights")
    updated = local_embedding_revisions(str(adapter), visual=True)
    assert updated["model_revision"] != original["model_revision"]
    assert updated["processor_revision"] == original["processor_revision"]
    (adapter / "adapter_model.safetensors").write_bytes(b"different adapter weights")
    updated_adapter = local_embedding_revisions(str(adapter), visual=True)
    assert updated_adapter["model_revision"] != updated["model_revision"]
    assert updated_adapter["processor_revision"] == updated["processor_revision"]
    (adapter / "preprocessor_config.json").write_text('{"size": 28}')
    updated_processor = local_embedding_revisions(str(adapter), visual=True)
    assert updated_processor["model_revision"] == updated_adapter["model_revision"]
    assert updated_processor["processor_revision"] != updated_adapter["processor_revision"]
    assert registered_image_search_directions({
        "category": 3, "provider": "local_qwen", "model_params": {
            "dimension": 128, "retrieval_directions": ["IMAGE_TO_IMAGE"],
        },
    }) == ["IMAGE_TO_IMAGE"]


def test_visual_template_covers_processor_library_version(monkeypatch):
    monkeypatch.setattr(local_embedding_templates, "version", lambda name: "0.3.0")
    template = local_embedding_templates.colqwen_input_template()
    assert template == local_embedding_templates.colqwen_input_template()
    monkeypatch.setattr(local_embedding_templates, "version", lambda name: "0.3.1")
    assert template != local_embedding_templates.colqwen_input_template()


class Pool(embedding_pool.EmbeddingModelPool):
    def __init__(self, path):
        super().__init__()
        self.data = {"category": 2, "provider": "local_qwen", "model_path": str(path),
                     "provider_model_name": "qwen-test", "model_name": "qwen-test",
                     "credential_mode": "NONE", "model_params": {"embedding_dimension": 4,
                     "model_revision": "ignored", "processor_revision": "ignored", "input_template": 42}}

    async def _fetch_model_data(self, model_id, **kwargs):
        return self.data

    async def _start_model(self, model_id, model_data):
        return model_data["model_params"]

    async def _shutdown_model_instance(self, model):
        pass

    async def _perform_model_health_check(self, model_id, model):
        pass


@pytest.mark.asyncio
async def test_binding_and_loaded_model_use_same_computed_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_pool, "get_embedding_config", lambda: SimpleNamespace(max_tokens=512))
    source = checkpoint(tmp_path / "source")
    model_id = uuid4()
    pool = Pool(source)
    binding_args = {}
    binding = await pool.inference_binding(model_id, representation_kind="DENSE", **binding_args)
    before = local_embedding_revisions(str(source))
    assert {key: binding["representation"][key] for key in before} == before
    runtime_params = await pool.load_model(model_id, expected_revision=binding["revision"], **binding_args)
    assert {key: runtime_params[key] for key in before} == before
    template = binding["representation"]["input_template"]
    assert template.startswith("local-text-embedding/v1:sha256:")
    assert runtime_params["input_template"] == template
    assert pool.data["model_params"]["model_revision"] == "ignored"
    assert pool.data["model_params"].pop("input_template") == 42
    assert await pool.inference_binding(model_id, representation_kind="DENSE", **binding_args) == binding
    pool.data["model_params"]["instruction"] = "Find relevant passages."
    new_input = await pool.inference_binding(model_id, representation_kind="DENSE", **binding_args)
    assert new_input["representation"]["input_template"] != template
    assert {key: new_input["representation"][key] for key in before} == before
    with pytest.raises(FrozenModelConflict):
        await pool.load_model(model_id, expected_revision=binding["revision"], **binding_args)
    pool.data["model_params"]["max_tokens"] = 1024
    new_limit = await pool.inference_binding(model_id, representation_kind="DENSE", **binding_args)
    assert new_limit["representation"]["input_template"] != new_input["representation"]["input_template"]
    assert {key: new_limit["representation"][key] for key in before} == before
    (source / "tokenizer.json").write_text('{"tokenizer": 2}')
    next_binding = await pool.inference_binding(model_id, representation_kind="DENSE", **binding_args)
    assert next_binding["revision"] != new_limit["revision"]
    with pytest.raises(FrozenModelConflict):
        await pool.load_model(model_id, expected_revision=new_limit["revision"], **binding_args)
    await pool.unload_model(model_id)
    params = await pool.load_model(model_id, expected_revision=next_binding["revision"], **binding_args)
    assert params["processor_revision"] == next_binding["representation"]["processor_revision"]


@pytest.mark.asyncio
async def test_remote_models_keep_registered_values_and_missing_local_files_fail(tmp_path):
    remote = {"category": 2, "provider": "api_qwen", "model_params": {
        "embedding_dimension": 4, "model_revision": "vendor-v1", "processor_revision": "vendor-v1",
    }}
    assert await with_local_embedding_metadata(remote, input_template=None) is remote
    with pytest.raises(ValueError, match="完整表示规格"):
        representation_metadata(remote, "DENSE")
    remote["model_params"]["input_template"] = "vendor-query-v1"
    assert representation_metadata(remote, "DENSE")["input_template"] == "vendor-query-v1"
    local = {"category": 2, "provider": "local_bge", "model_path": str(tmp_path)}
    with pytest.raises(ValueError, match="缺少.*权重"):
        await with_local_embedding_metadata(local, input_template=None)
