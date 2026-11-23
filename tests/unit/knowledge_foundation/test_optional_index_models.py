"""普通知识库与运维库均允许不配置向量模型。"""

from uuid import uuid4

import pytest

from platform_core.contracts.aiops.knowledge_base import KnowledgeBaseWrite
from platform_core.contracts.knowledge_app import KnowledgeLibraryCreate, KnowledgeLibraryUpdate


@pytest.mark.parametrize("contract", [KnowledgeBaseWrite, KnowledgeLibraryCreate])
def test_creation_accepts_no_embedding_or_visual_only(contract):
    base = {"display_name": "知识库"}
    assert contract.model_validate(base).index_models is None
    model_id = uuid4()
    command = contract.model_validate({**base, "index_models": {"visual_embedding": model_id}})
    assert command.index_models.embedding is None
    assert command.index_models.visual_embedding == model_id


@pytest.mark.parametrize("contract, base", [
    (KnowledgeBaseWrite, {"display_name": "知识库", "expected_row_version": 1}),
    (KnowledgeLibraryUpdate, {"expected_revision": 1}),
])
def test_update_distinguishes_omitting_from_clearing_index_bindings(contract, base):
    assert "index_models" not in contract.model_validate(base).model_dump(exclude_unset=True)
    command = contract.model_validate({**base, "index_models": None})
    assert command.model_dump(exclude_unset=True)["index_models"] is None
