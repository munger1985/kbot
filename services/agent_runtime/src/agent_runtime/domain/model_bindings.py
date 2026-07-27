"""Agent Runtime 按功能角色管理模型绑定。"""

import re
from collections.abc import Mapping
from uuid import UUID


AGENT_REQUIRED_MODEL_ROLES = frozenset(
    {
        "context_llm",
        "composer_llm",
        "memory_llm",
        "memory_embedding",
    }
)
AGENT_IMMUTABLE_MODEL_ROLES = frozenset({"memory_embedding"})
_ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def normalize_agent_models(
    models: Mapping[str, UUID | str],
) -> dict[str, str]:
    """校验 Agent 模型角色，并统一保存模型 UUIDv7。"""
    normalized: dict[str, str] = {}
    for raw_role, raw_model_id in models.items():
        role = str(raw_role).strip()
        if not _ROLE_PATTERN.fullmatch(role):
            raise ValueError(f"模型角色名称非法：{raw_role}")
        try:
            model_id = (
                raw_model_id
                if isinstance(raw_model_id, UUID)
                else UUID(str(raw_model_id))
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"模型角色 {role} 必须绑定 UUID") from exc
        if model_id.version != 7:
            raise ValueError(f"模型角色 {role} 必须绑定 UUIDv7")
        normalized[role] = str(model_id)
    missing = AGENT_REQUIRED_MODEL_ROLES - set(normalized)
    if missing:
        raise ValueError(f"Agent 缺少必选模型角色：{sorted(missing)}")
    return normalized


def agent_model_id(agent, role: str) -> UUID | None:
    """从 Agent Entity 读取指定角色的模型 UUID。"""
    raw = dict(agent.models_json or {}).get(role)
    return UUID(str(raw)) if raw else None


def agent_model_name(snapshot: Mapping, role: str) -> str | None:
    """从冻结的 Agent 快照读取模型调用名称。"""
    value = dict(snapshot.get("models") or {}).get(role)
    if not isinstance(value, Mapping):
        return None
    name = str(value.get("served_model_name") or "").strip()
    return name or None
