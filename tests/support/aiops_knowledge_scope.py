"""知识检索测试的显式 Agent 授权替身，不跳过生产范围校验。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from platform_core.identity import uuid7


KNOWLEDGE_BASE_ID = "019ffff0-0000-7000-8000-000000001001"


def active_knowledge_bases_fixture(repository):
    """夹具显式声明绑定库当前有效；测试可替换为空或模拟检索期间归档。"""
    repository.active_knowledge_base_ids = AsyncMock(
        side_effect=lambda **kwargs: list(kwargs["knowledge_base_ids"]),
    )
    return repository


def knowledge_agent_repository(*, knowledge_base_ids=(KNOWLEDGE_BASE_ID,), **policy):
    return SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status="ACTIVE", current_version_id=uuid7())),
        version=AsyncMock(return_value=SimpleNamespace(models_json={})),
        version_has_target=AsyncMock(return_value=True),
        version_knowledge_policy=AsyncMock(return_value={
            "enabled": True, "source_kinds": ["MANUAL", "DIAGNOSIS_CASE"],
            "knowledge_base_ids": list(knowledge_base_ids), "max_security_level": 5,
            "tag_ids": [], **policy,
        }),
    )
