"""AIOps 连续诊断对话内部 API。"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.api.dependencies import get_aiops_auth_context, require_service_scope
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    CreateOpsRunCommand,
    HitlResponse,
)
from platform_core.contracts.aiops.public import HitlResponseItem
from platform_core.identity import uuid7


router = APIRouter(prefix="/internal/v1/aiops/conversations", tags=["AIOps Conversations"])


class _Payload(BaseModel): model_config = ConfigDict(extra="forbid")
class MessagePayload(_Payload):
    agent_id: UUID
    message: str = Field(min_length=1, max_length=32000)
    conversation_id: UUID | None = None
    source_run_id: UUID | None = None
    request_report: bool = False
class EvidenceRequestPayload(_Payload):
    purpose: str = Field(min_length=1, max_length=4000)
    suggested_sql: str | None = Field(default=None, max_length=32000)
class EvidenceTextPayload(_Payload): text: str = Field(min_length=1, max_length=32000)
class EvidenceUploadPayload(_Payload):
    filename: str = Field(min_length=1, max_length=512)
    mime_type: str = Field(min_length=3, max_length=128)
    content_base64: str = Field(min_length=1, max_length=14_000_000)
    text: str | None = Field(default=None, max_length=32000)


def _hitl_response(pending, text: str) -> HitlResponse:
    """把同一输入框中的回贴结果映射为确定性的 HITL 回复。"""
    queries = list((pending.request or {}).get("queries") or [])
    if not queries:
        raise HTTPException(422, {"code": "AIOPS_EVIDENCE_REQUEST_INVALID"})
    outputs: dict[str, str] = {}
    if len(queries) == 1:
        outputs[str(queries[0]["query_id"])] = text
    else:
        markers = []
        for query in queries:
            query_id = str(query["query_id"])
            marker = f"[{query_id}]"
            index = text.find(marker)
            if index < 0:
                raise HTTPException(
                    422,
                    {
                        "code": "AIOPS_EVIDENCE_QUERY_MARKER_REQUIRED",
                        "message": f"多条查询结果请分别以 {marker} 作为标题",
                    },
                )
            markers.append((index, query_id, len(marker)))
        markers.sort()
        for offset, (index, query_id, marker_length) in enumerate(markers):
            end = markers[offset + 1][0] if offset + 1 < len(markers) else len(text)
            outputs[query_id] = text[index + marker_length:end].strip()
    return HitlResponse(
        expected_row_version=pending.row_version,
        responses=tuple(
            HitlResponseItem(
                query_id=str(query["query_id"]),
                status="SUCCEEDED",
                raw_output=outputs[str(query["query_id"])],
            )
            for query in queries
        ),
    )


async def _resume_hitl(*, request, context, domain_id, actor_id, request_id, text):
    runtime = request.app.state.aiops_runtime_service
    pending = await runtime.get_hitl_input(
        hitl_id=request_id,
        domain_id=domain_id,
        actor_id=actor_id,
    )
    await runtime.respond_hitl(
        hitl_id=request_id,
        domain_id=domain_id,
        actor_id=actor_id,
        response=_hitl_response(pending, text),
        idempotency_key=str(uuid7()),
        trace_id=context.trace_id,
    )


def _scope(request: Request, context: AuthContext) -> tuple[int, str]:
    require_service_scope(request, "aiops.run")
    if context.domain_id is None:
        raise HTTPException(403, {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"})
    domain_id = int(context.domain_id)
    if domain_id < 1:
        raise HTTPException(403, {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"})
    return domain_id, context.asserted_user_id or context.client_id


@router.post("", status_code=201)
async def create_or_append(payload: MessagePayload, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    service = request.app.state.conversation_service
    conversation = await service.create_or_append(
        domain_id=domain_id,
        agent_id=payload.agent_id,
        actor_id=actor_id,
        message=payload.message,
        conversation_id=payload.conversation_id,
        source_run_id=payload.source_run_id,
    )
    target_id = await service.target_for(domain_id=domain_id, agent_id=payload.agent_id)
    receipt = await request.app.state.aiops_runtime_service.create_run(CreateOpsRunCommand(
        command_id=uuid7(), idempotency_key=str(uuid7()), domain_id=domain_id,
        actor_id=actor_id, agent_id=payload.agent_id, target_id=target_id,
        trigger_type="CHAT", input=payload.message,
        blueprint_id="diagnosis.root-cause", blueprint_version="1",
        client_metadata={
            "trace_id": context.trace_id,
            "conversation_id": conversation["conversation_id"],
            "source_run_id": conversation.get("source_run_id"),
            "report_requested": payload.request_report,
        },
    ))
    await service.attach_run(domain_id=domain_id, conversation_id=UUID(conversation["conversation_id"]), ops_run_id=receipt.ops_run_id, purpose="QUESTION")
    return {**conversation, "run_id": str(receipt.ops_run_id)}


@router.get("")
async def list_conversations(request: Request, agent_id: UUID | None = None, limit: int = Query(50, ge=1, le=50), context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_service.list(domain_id=domain_id, actor_id=actor_id, agent_id=agent_id, limit=limit)


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: UUID, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_service.get(domain_id=domain_id, conversation_id=conversation_id, actor_id=actor_id)


@router.post("/{conversation_id}/evidence-requests", status_code=201)
async def request_evidence(conversation_id: UUID, payload: EvidenceRequestPayload, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_service.request_evidence(domain_id=domain_id, conversation_id=conversation_id, actor_id=actor_id, purpose=payload.purpose, suggested_sql=payload.suggested_sql)


@router.post("/{conversation_id}/evidence-requests/{request_id}/text")
async def submit_evidence_text(conversation_id: UUID, request_id: UUID, payload: EvidenceTextPayload, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    await _resume_hitl(
        request=request, context=context, domain_id=domain_id,
        actor_id=actor_id, request_id=request_id, text=payload.text,
    )
    return await request.app.state.conversation_service.submit_evidence_text(domain_id=domain_id, conversation_id=conversation_id, request_id=request_id, actor_id=actor_id, text=payload.text)


@router.post("/{conversation_id}/evidence-requests/{request_id}/skip")
async def skip_evidence(conversation_id: UUID, request_id: UUID, payload: EvidenceTextPayload, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_service.submit_evidence_text(domain_id=domain_id, conversation_id=conversation_id, request_id=request_id, actor_id=actor_id, text=payload.text, skipped=True)


@router.post("/{conversation_id}/evidence-requests/{request_id}/uploads")
async def upload_evidence(conversation_id: UUID, request_id: UUID, payload: EvidenceUploadPayload, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    result = await request.app.state.conversation_service.upload_evidence_file(domain_id=domain_id, conversation_id=conversation_id, request_id=request_id, actor_id=actor_id, **payload.model_dump())
    conversation = await request.app.state.conversation_service.get(
        domain_id=domain_id,
        conversation_id=conversation_id,
        actor_id=actor_id,
    )
    extracted = next(
        (
            str(item["payload"].get("text") or "").strip()
            for item in reversed(conversation["messages"])
            if item["message_type"] == "IMAGE_EVIDENCE_PROCESSED"
            and item["payload"].get("processing_id")
            == result.get("image_processing_id")
        ),
        "",
    )
    await _resume_hitl(
        request=request, context=context, domain_id=domain_id,
        actor_id=actor_id, request_id=request_id,
        text=extracted or payload.text or "图片中未提取到可用文字",
    )
    return result
