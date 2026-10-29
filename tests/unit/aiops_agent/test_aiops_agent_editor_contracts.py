"""Agent 编辑器关闭知识能力时保留选择，不自动增加知识来源。"""

import pytest
from pydantic import ValidationError

from platform_core.contracts.aiops import AgentKnowledgePolicy
from platform_core.identity import uuid7


@pytest.mark.parametrize("sources", [(), ("MANUAL",), ("DIAGNOSIS_CASE",)])
def test_disabled_knowledge_preserves_source_selection(sources):
    base_id, tag_id = uuid7(), uuid7()
    policy = AgentKnowledgePolicy(
        enabled=False, source_kinds=sources, knowledge_base_ids=(base_id,),
        tag_ids=(tag_id,), max_security_level=4, maximum_references=6,
    )
    assert policy.source_kinds == sources
    assert policy.knowledge_base_ids == (base_id,)
    assert policy.tag_ids == (tag_id,)


def test_enabled_knowledge_requires_source_and_knowledge_base():
    with pytest.raises(ValidationError, match="知识来源"):
        AgentKnowledgePolicy(enabled=True, source_kinds=(), knowledge_base_ids=(uuid7(),))
    with pytest.raises(ValidationError, match="至少需要绑定"):
        AgentKnowledgePolicy(enabled=True, source_kinds=("MANUAL",))
    with pytest.raises(ValidationError, match="知识来源"):
        AgentKnowledgePolicy(enabled=False, source_kinds=("UNAPPROVED",))
