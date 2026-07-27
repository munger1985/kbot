"""Knowledge Core 按功能角色管理模型绑定。"""

import re
from collections.abc import Mapping
from uuid import UUID


KC_REQUIRED_MODEL_ROLES = frozenset(
    {"parser_llm", "retrieval_llm", "embedding"}
)
KC_IMMUTABLE_MODEL_ROLES = frozenset(
    {"embedding", "visual_embedding"}
)
_ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def normalize_collection_models(
    models: Mapping[str, UUID | str],
) -> dict[str, str]:
    """校验可扩展角色映射，并转换为可持久化 JSON。"""
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
    missing = KC_REQUIRED_MODEL_ROLES - set(normalized)
    if missing:
        raise ValueError(
            f"Collection 缺少必选模型角色：{sorted(missing)}"
        )
    return normalized


def collection_model_id(collection, role: str) -> UUID | None:
    """从 Collection Entity 或 Snapshot 获取指定角色的模型 UUID。"""
    raw = dict(collection.models_json or {}).get(role)
    return UUID(str(raw)) if raw else None
