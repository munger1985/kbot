"""模型目录快照：只保留可对外暴露的安全字段。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from assistant_app.application.errors import AssistantApplicationError


SAFE_SNAPSHOT_FIELDS = (
    "model_id",
    "display_name",
    "served_model_name",
    "provider",
    "status",
    "supports_x_search",
    "supports_image_generation",
    "supports_responses_streaming",
    "capability_verified_at",
    "category",
)

ROLE_CAPABILITY = {
    "KNOWLEDGE": None,
    "X_SEARCH": "supports_x_search",
    "IMAGE_GENERATION": "supports_image_generation",
}


def _iso(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def safe_model_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    """绑定与 Run 只保存安全投影，不得写入 Secret 或连接材料。"""
    snapshot = {
        "model_id": str(row.get("model_id") or ""),
        "display_name": row.get("display_name"),
        "served_model_name": row.get("served_model_name"),
        "provider": row.get("provider"),
        "status": row.get("status"),
        "supports_x_search": bool(row.get("supports_x_search")),
        "supports_image_generation": bool(row.get("supports_image_generation")),
        "supports_responses_streaming": bool(row.get("supports_responses_streaming")),
        "capability_verified_at": _iso(row.get("capability_verified_at")),
        "category": int(row["category"]) if row.get("category") is not None else None,
    }
    return {key: snapshot[key] for key in SAFE_SNAPSHOT_FIELDS}


def is_role_ready(role: str, snapshot: dict[str, Any]) -> bool:
    if str(snapshot.get("status") or "").upper() != "ACTIVE":
        return False
    flag = ROLE_CAPABILITY.get(role)
    if flag is None:
        return True
    return bool(snapshot.get(flag))


async def load_catalog_model(catalog_client, model_id: UUID) -> dict[str, Any]:
    try:
        row = await catalog_client.get_model(model_id)
    except LookupError as exc:
        raise AssistantApplicationError(
            "MODEL_BINDING_MISSING", "绑定的模型在目录中不存在", status_code=422,
        ) from exc
    except Exception as exc:
        raise AssistantApplicationError(
            "MODEL_CATALOG_UNAVAILABLE", "模型目录暂时不可用", status_code=503,
        ) from exc
    if not isinstance(row, dict):
        raise AssistantApplicationError(
            "MODEL_CATALOG_UNAVAILABLE", "模型目录返回了无效定义", status_code=503,
        )
    return row
