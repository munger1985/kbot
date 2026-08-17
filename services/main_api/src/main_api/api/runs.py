"""Agent Run 的公开命令、结果和 SSE 事件接口。"""

from __future__ import annotations

import asyncio
import json
from time import monotonic
from typing import Any, AsyncIterator, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from platform_clients import AgentRuntimeClient, KnowledgeCoreClient
from platform_core.contracts import (
    AgentArtifact,
    AgentRunReceipt,
    AgentRunSummary,
    PUBLIC_API_V1,
)
from main_api.application import AccessConfigurationError, AccessDeniedError
from platform_clients import KnowledgeRetrievalAppClient
from platform_core.security import get_auth_context


async def _require_use(request: Request) -> None:
    context = get_auth_context(request)
    try:
        await request.app.state.access_control_service.require(
            app_id="knowledge_retrieval",
            domain_id=int(context.domain_id or "0"),
            user_id=context.asserted_user_id or context.client_id,
            permission_code="knowledge_retrieval:use",
        )
    except AccessDeniedError as exc:
        raise HTTPException(
            403,
            {"code": "APP_PERMISSION_DENIED", "permission": "knowledge_retrieval:use"},
        ) from exc


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/knowledge-retrieval/runs",
    tags=["Knowledge Retrieval Runs"],
    dependencies=[Depends(_require_use)],
)

_TERMINAL_EVENTS = {
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_CANCELLED",
    "RUN_EXPIRED",
}


class CancelRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_row_version: int = Field(ge=1)


class KnowledgeRunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: UUID
    input: str = Field(min_length=1, max_length=32000)
    collection_ids: tuple[UUID, ...] = ()
    security_level: int = Field(default=3, ge=0, le=3)
    client_metadata: dict = Field(default_factory=dict)


class DocumentReferencePreview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reference_type: Literal["DOCUMENT"] = "DOCUMENT"
    citation_label: str
    title: str
    mime_type: str
    preview_type: Literal["PDF", "IMAGE", "TEXT", "DOWNLOAD"]
    page_no: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    content_url: str
    download_available: bool = True


class _DocumentReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    reference_type: Literal["DOCUMENT"]
    citation_label: str
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    document_id: UUID
    document_version_id: UUID
    title: str
    locator: dict[str, Any] = Field(default_factory=dict)
    locator_schema_version: str


def _client(request: Request) -> AgentRuntimeClient:
    return cast(
        AgentRuntimeClient,
        request.app.state.agent_runtime_client,
    )


def _knowledge_client(request: Request) -> KnowledgeCoreClient:
    return cast(KnowledgeCoreClient, request.app.state.knowledge_core_client)


async def _authorized_spec(request: Request, agent_id: UUID) -> dict:
    context = get_auth_context(request)
    domain_id = int(context.domain_id or "0")
    client: KnowledgeRetrievalAppClient = (
        request.app.state.knowledge_retrieval_app_client
    )
    return await client.execution_spec(
        agent_id=agent_id, domain_id=domain_id, auth_context=context
    )


async def _effective_security_level(
    request: Request,
    *,
    requested_level: int,
    execution_spec: dict,
) -> int:
    """按用户、Agent 和请求三者的最小值生成受信检索等级。"""
    context = get_auth_context(request)
    actor_id = context.asserted_user_id or context.client_id
    try:
        user_level = (
            await request.app.state.access_control_service.user_max_security_level(
                user_id=actor_id
            )
        )
    except AccessConfigurationError as exc:
        raise HTTPException(
            403,
            {
                "code": "USER_SECURITY_LEVEL_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc
    resource_context = execution_spec.get("resource_context") or {}
    raw_agent_level = resource_context.get(
        "max_security_level",
        resource_context.get("security_level", 3),
    )
    try:
        agent_level = int(raw_agent_level)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            409,
            {
                "code": "AGENT_SECURITY_LEVEL_INVALID",
                "message": "Agent 安全等级配置无效",
            },
        ) from exc
    if agent_level < 0 or agent_level > 3:
        raise HTTPException(
            409,
            {
                "code": "AGENT_SECURITY_LEVEL_INVALID",
                "message": "Agent 安全等级必须在 0 到 3 之间",
            },
        )
    return min(user_level, agent_level, requested_level)


@router.post("", status_code=202, response_model=AgentRunReceipt)
async def create_run(
    payload: KnowledgeRunCreateRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> AgentRunReceipt:
    spec = await _authorized_spec(request, payload.agent_id)
    effective_level = await _effective_security_level(
        request,
        requested_level=payload.security_level,
        execution_spec=spec,
    )
    result = await _client(request).create_run(
        payload={
            **payload.model_dump(mode="json"),
            "security_level": effective_level,
            "execution_spec": spec,
        },
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return AgentRunReceipt.model_validate(result)


@router.get("/{run_id}", response_model=AgentRunSummary)
async def get_run(
    run_id: UUID, request: Request
) -> AgentRunSummary:
    result = await _client(request).get_run(
        run_id=run_id,
        auth_context=request.state.auth_context,
    )
    return AgentRunSummary.model_validate(result)


@router.get("/{run_id}/result", response_model=AgentArtifact)
async def get_run_result(
    run_id: UUID, request: Request
) -> AgentArtifact:
    result = await _client(request).get_result(
        run_id=run_id,
        auth_context=request.state.auth_context,
    )
    return AgentArtifact.model_validate(result)


@router.get(
    "/{run_id}/references/{citation_label}/preview",
    response_model=DocumentReferencePreview,
)
async def get_document_reference_preview(
    run_id: UUID,
    citation_label: str,
    request: Request,
) -> DocumentReferencePreview:
    reference = await _authorized_document_reference(
        request=request,
        run_id=run_id,
        citation_label=citation_label,
    )
    preview = await _knowledge_client(request).get_bundle_revision_preview(
        domain_id=int(request.state.auth_context.domain_id or "0"),
        collection_id=reference.collection_id,
        bundle_id=reference.bundle_id,
        bundle_revision_id=reference.bundle_revision_id,
        auth_context=request.state.auth_context,
    )
    source_file = next(
        (
            item
            for item in preview.get("files", [])
            if str(item.get("document_version_id"))
            == str(reference.document_version_id)
            and bool(item.get("preview_available"))
        ),
        None,
    )
    if source_file is None:
        raise _reference_not_found()
    mime_type = str(
        source_file.get("detected_mime_type")
        or source_file.get("declared_mime_type")
        or "application/octet-stream"
    ).split(";", 1)[0].strip().lower()
    page_no, page_end, bbox = _document_locator(reference)
    return DocumentReferencePreview(
        citation_label=reference.citation_label,
        title=reference.title,
        mime_type=mime_type,
        preview_type=_preview_type(mime_type),
        page_no=page_no,
        page_end=page_end,
        bbox=bbox,
        content_url=(
            f"{PUBLIC_API_V1}/apps/knowledge-retrieval/runs/{run_id}"
            f"/references/{reference.citation_label}/content"
        ),
    )


@router.get("/{run_id}/references/{citation_label}/content")
async def stream_document_reference_content(
    run_id: UUID,
    citation_label: str,
    request: Request,
    range_header: str | None = Header(default=None, alias="Range"),
) -> StreamingResponse:
    reference = await _authorized_document_reference(
        request=request,
        run_id=run_id,
        citation_label=citation_label,
    )
    upstream = await _knowledge_client(request).stream_source_file(
        domain_id=int(request.state.auth_context.domain_id or "0"),
        collection_id=reference.collection_id,
        bundle_id=reference.bundle_id,
        bundle_revision_id=reference.bundle_revision_id,
        document_version_id=reference.document_version_id,
        range_header=range_header,
        auth_context=request.state.auth_context,
    )
    forwarded_headers = {
        header: upstream.headers[header]
        for header in (
            "accept-ranges",
            "cache-control",
            "content-disposition",
            "content-length",
            "content-range",
            "content-security-policy",
            "x-content-type-options",
        )
        if header in upstream.headers
    }
    return StreamingResponse(
        upstream.body,
        status_code=upstream.status_code,
        media_type=upstream.headers.get(
            "content-type", "application/octet-stream"
        ),
        headers=forwarded_headers,
    )


async def _authorized_document_reference(
    *, request: Request, run_id: UUID, citation_label: str
) -> _DocumentReference:
    artifact = await _client(request).get_result(
        run_id=run_id,
        auth_context=request.state.auth_context,
    )
    payload = artifact.get("payload")
    references = payload.get("references") if isinstance(payload, dict) else None
    if not isinstance(references, list):
        raise _reference_not_found()
    raw_reference = next(
        (
            item
            for item in references
            if isinstance(item, dict)
            and item.get("reference_type") == "DOCUMENT"
            and item.get("citation_label") == citation_label
        ),
        None,
    )
    if raw_reference is None:
        raise _reference_not_found()
    try:
        return _DocumentReference.model_validate(raw_reference)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "DOCUMENT_REFERENCE_INVALID",
                "message": "Run 引用缺少不可变文档定位信息",
            },
        ) from exc


def _document_locator(
    reference: _DocumentReference,
) -> tuple[
    int | None,
    int | None,
    tuple[float, float, float, float] | None,
]:
    if reference.locator_schema_version != "document/v1":
        return None, None, None
    pages = reference.locator.get("pages")
    if not isinstance(pages, list):
        return None, None, None
    page_numbers = [
        int(item["page_no"])
        for item in pages
        if isinstance(item, dict)
        and isinstance(item.get("page_no"), int)
        and int(item["page_no"]) >= 1
    ]
    first = pages[0] if pages and isinstance(pages[0], dict) else {}
    raw_bbox = first.get("bbox")
    bbox = (
        tuple(float(value) for value in raw_bbox)
        if isinstance(raw_bbox, list)
        and len(raw_bbox) == 4
        and all(isinstance(value, (int, float)) for value in raw_bbox)
        else None
    )
    if not page_numbers:
        return None, None, bbox
    return min(page_numbers), max(page_numbers), bbox


def _preview_type(
    mime_type: str,
) -> Literal["PDF", "IMAGE", "TEXT", "DOWNLOAD"]:
    if mime_type == "application/pdf":
        return "PDF"
    if mime_type in {"image/gif", "image/jpeg", "image/png", "image/webp"}:
        return "IMAGE"
    if mime_type == "text/plain":
        return "TEXT"
    return "DOWNLOAD"


def _reference_not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "code": "DOCUMENT_REFERENCE_NOT_FOUND",
            "message": "引用不存在或当前用户无权访问",
        },
    )


@router.post(
    "/{run_id}/cancel",
    status_code=202,
    response_model=AgentRunReceipt,
)
async def cancel_run(
    run_id: UUID,
    payload: CancelRunRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> AgentRunReceipt:
    result = await _client(request).cancel_run(
        run_id=run_id,
        expected_row_version=payload.expected_row_version,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return AgentRunReceipt.model_validate(result)


@router.get("/{run_id}/events")
async def stream_run_events(
    run_id: UUID,
    request: Request,
    last_event_id: str | None = Header(
        default=None, alias="Last-Event-ID"
    ),
) -> StreamingResponse:
    cursor = _parse_cursor(last_event_id)
    summary = await _client(request).get_run(
        run_id=run_id,
        auth_context=request.state.auth_context,
    )
    latest = int(summary["event_cursor"])
    if cursor > latest:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "AGENT_EVENT_CURSOR_INVALID",
                "message": "Last-Event-ID 超过当前 Run 事件游标",
            },
        )
    return StreamingResponse(
        _event_stream(
            run_id=run_id,
            request=request,
            cursor=cursor,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _parse_cursor(value: str | None) -> int:
    if value is None or not value.strip():
        return 0
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_EVENT_CURSOR",
                "message": "Last-Event-ID 必须是非负整数",
            },
        ) from exc
    if parsed < 0:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_EVENT_CURSOR",
                "message": "Last-Event-ID 不能为负数",
            },
        )
    return parsed


async def _event_stream(
    *,
    run_id: UUID,
    request: Request,
    cursor: int,
) -> AsyncIterator[str]:
    settings = request.app.state.main_api_settings
    poll_interval = settings.api.sse_poll_interval_seconds
    heartbeat_seconds = settings.api.sse_heartbeat_seconds
    batch_size = settings.api.sse_batch_size
    auth_context = request.state.auth_context
    last_output_at = monotonic()
    while not await request.is_disconnected():
        events = await _client(request).list_events(
            run_id=run_id,
            after_sequence=cursor,
            limit=batch_size,
            auth_context=auth_context,
        )
        for event in events:
            cursor = int(event["sequence_no"])
            yield _format_sse(event)
            last_output_at = monotonic()
            if event["event_type"] in _TERMINAL_EVENTS:
                yield (
                    "event: done\n"
                    f"data: {json.dumps({'sequence_no': cursor})}\n\n"
                )
                return
        if monotonic() - last_output_at >= heartbeat_seconds:
            yield ": heartbeat\n\n"
            last_output_at = monotonic()
        await asyncio.sleep(poll_interval)


def _format_sse(event: dict) -> str:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return (
        f"id: {event['sequence_no']}\n"
        f"event: {event['event_type']}\n"
        f"data: {payload}\n\n"
    )
