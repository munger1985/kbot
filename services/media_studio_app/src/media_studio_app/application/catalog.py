"""模型目录快照：只保留可对外暴露的安全字段。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from media_studio_app.application.errors import MediaStudioApplicationError


SAFE_SNAPSHOT_FIELDS = (
    "model_id",
    "display_name",
    "served_model_name",
    "provider",
    "status",
    "category",
)

ROLE_MODEL_CATEGORY = {"IMAGE_GENERATION": 8}


def safe_model_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    """绑定与 Run 只保存安全投影，不得写入 Secret 或连接材料。"""
    snapshot = {
        "model_id": str(row.get("model_id") or ""),
        "display_name": row.get("display_name"),
        "served_model_name": row.get("served_model_name"),
        "provider": row.get("provider"),
        "status": row.get("status"),
        "category": int(row["category"]) if row.get("category") is not None else None,
    }
    return {key: snapshot[key] for key in SAFE_SNAPSHOT_FIELDS}


def is_role_ready(role: str, snapshot: dict[str, Any]) -> bool:
    expected_category = ROLE_MODEL_CATEGORY.get(role)
    return (
        str(snapshot.get("status") or "").upper() == "ACTIVE"
        and (
            expected_category is None
            or int(snapshot.get("category") or 0) == expected_category
        )
    )


async def load_catalog_model(catalog_client, model_id: UUID) -> dict[str, Any]:
    try:
        row = await catalog_client.get_model(model_id)
    except LookupError as exc:
        raise MediaStudioApplicationError(
            "MODEL_BINDING_MISSING", "绑定的模型在目录中不存在", status_code=422,
        ) from exc
    except Exception as exc:
        raise MediaStudioApplicationError(
            "MODEL_CATALOG_UNAVAILABLE", "模型目录暂时不可用", status_code=503,
        ) from exc
    if not isinstance(row, dict):
        raise MediaStudioApplicationError(
            "MODEL_CATALOG_UNAVAILABLE", "模型目录返回了无效定义", status_code=503,
        )
    return row
