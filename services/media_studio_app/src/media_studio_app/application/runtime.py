"""多媒体生成 Run 的共享投影与阶段写入。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from media_studio_app.application.errors import MediaStudioApplicationError
from media_studio_app.entities import MediaStudioRunEntity, MediaStudioRunEventEntity
from platform_core.identity import uuid7


TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "REJECTED"})
IMAGE_KIND = "IMAGE_GENERATION"


def iso(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def public_result(result_json: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result_json:
        return None
    data = dict(result_json)
    data.pop("upstream_attempted", None)
    return data


def run_view(row: MediaStudioRunEntity) -> dict[str, Any]:
    snapshot = dict(row.model_snapshot_json or {})
    return {
        "run_id": str(row.run_id),
        "domain_id": int(row.domain_id),
        "kind": row.kind,
        "status": row.status,
        "actor_id": row.actor_id,
        "request_id": row.request_id,
        "trace_id": row.trace_id,
        "idempotency_key": row.idempotency_key,
        "model_id": str(row.model_id),
        "model_display_name": snapshot.get("display_name"),
        "request": dict(row.request_json or {}),
        "result": public_result(row.result_json),
        "usage": dict(row.usage_json) if row.usage_json else None,
        "provider_request_id": row.provider_request_id,
        "prompt_revision_id": str(row.prompt_revision_id) if row.prompt_revision_id else None,
        "error_code": row.error_code,
        "error_message": row.error_message,
        "row_version": int(row.row_version),
        "created_at": iso(row.created_at),
        "started_at": iso(row.started_at),
        "completed_at": iso(row.completed_at),
        "updated_at": iso(row.updated_at),
    }


def event_view(row) -> dict[str, Any]:
    payload = dict(row.payload_json) if row.payload_json else None
    return {
        "event_id": str(row.event_id),
        "run_id": str(row.run_id),
        "sequence_no": int(row.sequence_no),
        "stage": row.stage,
        "message": row.message,
        "payload": payload,
        "created_at": iso(row.created_at),
    }


def media_view(row, *, content_path: str | None = None) -> dict[str, Any]:
    return {
        "asset_id": str(row.asset_id),
        "domain_id": int(row.domain_id),
        "run_id": str(row.run_id),
        "prompt_revision_id": str(row.prompt_revision_id) if row.prompt_revision_id else None,
        "object_key": row.object_key,
        "mime_type": row.mime_type,
        "byte_size": int(row.byte_size),
        "content_sha256": row.content_sha256,
        "width": int(row.width) if row.width is not None else None,
        "height": int(row.height) if row.height is not None else None,
        "status": row.status,
        "provider_artifact_id": row.provider_artifact_id,
        "created_by": row.created_by,
        "created_at": iso(row.created_at),
        "content_path": content_path,
    }


def not_found(kind: str = "RUN") -> None:
    raise MediaStudioApplicationError(f"{kind}_NOT_FOUND", "资源不存在", status_code=404)


async def append_event(uow, row: MediaStudioRunEntity, *, stage: str, message: str | None = None, payload: dict[str, Any] | None = None) -> None:
    assert uow.run_events is not None
    sequence_no = await uow.run_events.next_sequence(run_id=row.run_id)
    now = datetime.now(timezone.utc)
    await uow.run_events.add(MediaStudioRunEventEntity(
        event_id=uuid7(),
        run_id=row.run_id,
        domain_id=int(row.domain_id),
        sequence_no=sequence_no,
        stage=stage,
        message=message,
        payload_json=payload,
        created_at=now,
    ))
    row.status = stage
    row.updated_at = now
    row.row_version = int(row.row_version) + 1
    if stage in TERMINAL_STATUSES:
        row.completed_at = now


def mark_upstream_attempted(row: MediaStudioRunEntity) -> None:
    result = dict(row.result_json or {})
    result["upstream_attempted"] = True
    row.result_json = result


def already_attempted(row: MediaStudioRunEntity) -> bool:
    if row.provider_request_id:
        return True
    result = row.result_json or {}
    return bool(result.get("upstream_attempted"))
