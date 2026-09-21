"""X Search 使用的安全模型目录投影。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from knowledge_retrieval_app.application.errors import KnowledgeRetrievalApplicationError
from platform_core.contracts import is_grok_provider_model_name


SAFE_SNAPSHOT_FIELDS = (
    "model_id", "display_name", "served_model_name", "provider_model_name",
    "provider", "status", "category",
)


def safe_model_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    snapshot = {
        "model_id": str(row.get("model_id") or ""),
        "display_name": row.get("display_name"),
        "served_model_name": row.get("served_model_name"),
        "provider_model_name": row.get("provider_model_name"),
        "provider": row.get("provider"),
        "status": row.get("status"),
        "category": int(row["category"]) if row.get("category") is not None else None,
    }
    return {key: snapshot[key] for key in SAFE_SNAPSHOT_FIELDS}


def is_grok_model(row: dict[str, Any]) -> bool:
    return is_grok_provider_model_name(row.get("provider_model_name"))


async def load_catalog_model(catalog_client, model_id: UUID) -> dict[str, Any]:
    try:
        row = await catalog_client.get_model(model_id)
    except LookupError as exc:
        raise KnowledgeRetrievalApplicationError(
            "MODEL_BINDING_MISSING", "Agent 绑定的模型在目录中不存在", status_code=422,
        ) from exc
    except Exception as exc:
        raise KnowledgeRetrievalApplicationError(
            "MODEL_CATALOG_UNAVAILABLE", "模型目录暂时不可用", status_code=503,
        ) from exc
    if not isinstance(row, dict):
        raise KnowledgeRetrievalApplicationError(
            "MODEL_CATALOG_UNAVAILABLE", "模型目录返回了无效定义", status_code=503,
        )
    return row
