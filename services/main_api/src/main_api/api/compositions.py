"""模型、知识库、Agent、问数与 Run 的公开组合管理 API。"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from main_api.application import CompositionError, ResourceCompositionService
from platform_core.contracts import (
    AgentCompositionCreate,
    AgentCompositionUpdate,
    CollectionCompositionCreate,
    CollectionModelsCompositionUpdate,
    CompositionReceipt,
    PUBLIC_API_V1,
    ResourceDecommissionPrecheck,
    ResourceDecommissionResult,
    ResourceReferenceGraph,
    RunCompositionView,
    SemanticModelPublicationComposition,
)
from platform_core.security import get_auth_context


router = APIRouter(prefix=f"{PUBLIC_API_V1}/compositions", tags=["Resource Compositions"])


def _service(request: Request) -> ResourceCompositionService:
    return request.app.state.resource_composition_service


def _scope(request: Request) -> tuple[int, str, object]:
    context = get_auth_context(request)
    actor_id = str(context.asserted_user_id or "").strip()
    if context.domain_id is None or not actor_id:
        raise HTTPException(403, {"code": "COMPOSITION_CONTEXT_REQUIRED", "message": "组合管理需要 Domain 和 Actor 上下文"})
    try:
        return int(context.domain_id), actor_id, context
    except ValueError as exc:
        raise HTTPException(422, {"code": "COMPOSITION_DOMAIN_INVALID", "message": "Domain 标识格式无效"}) from exc


def _version(value: str) -> int:
    normalized = value.strip().removeprefix('W/').strip('"')
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise HTTPException(422, {"code": "IF_MATCH_INVALID", "message": "If-Match 必须是正整数版本"}) from exc
    if parsed < 1:
        raise HTTPException(422, {"code": "IF_MATCH_INVALID", "message": "If-Match 必须是正整数版本"})
    return parsed


def _raise(exc: CompositionError) -> None:
    raise HTTPException(exc.status_code, {"code": exc.code, "message": exc.message}) from exc


@router.post("/agents", response_model=CompositionReceipt, status_code=202)
async def create_agent_composition(
    body: AgentCompositionCreate, request: Request,
    idempotency_key: str = Header(min_length=8, max_length=128, alias="Idempotency-Key"),
) -> CompositionReceipt:
    domain_id, actor_id, context = _scope(request)
    try:
        return await _service(request).create_agent(
            body=body, domain_id=domain_id, actor_id=actor_id,
            idempotency_key=idempotency_key, context=context,
        )
    except CompositionError as exc:
        _raise(exc)


@router.patch("/agents/{agent_id}", response_model=CompositionReceipt, status_code=202)
async def update_agent_composition(
    agent_id: UUID, body: AgentCompositionUpdate, request: Request,
    idempotency_key: str = Header(min_length=8, max_length=128, alias="Idempotency-Key"),
    if_match: str = Header(alias="If-Match"),
) -> CompositionReceipt:
    domain_id, actor_id, context = _scope(request)
    try:
        return await _service(request).update_agent(
            agent_id=agent_id, body=body, domain_id=domain_id,
            actor_id=actor_id, idempotency_key=idempotency_key,
            expected_version=_version(if_match), context=context,
        )
    except CompositionError as exc:
        _raise(exc)


@router.post("/collections", response_model=CompositionReceipt, status_code=202)
async def create_collection_composition(
    body: CollectionCompositionCreate, request: Request,
    idempotency_key: str = Header(min_length=8, max_length=128, alias="Idempotency-Key"),
) -> CompositionReceipt:
    domain_id, actor_id, context = _scope(request)
    try:
        return await _service(request).create_collection(
            body=body, domain_id=domain_id, actor_id=actor_id,
            idempotency_key=idempotency_key, context=context,
        )
    except CompositionError as exc:
        _raise(exc)


@router.put("/collections/{collection_id}/models", response_model=CompositionReceipt, status_code=202)
async def update_collection_models_composition(
    collection_id: UUID, body: CollectionModelsCompositionUpdate,
    request: Request,
    idempotency_key: str = Header(min_length=8, max_length=128, alias="Idempotency-Key"),
    if_match: str = Header(alias="If-Match"),
) -> CompositionReceipt:
    domain_id, actor_id, context = _scope(request)
    try:
        return await _service(request).update_collection_models(
            collection_id=collection_id, body=body, domain_id=domain_id,
            actor_id=actor_id, idempotency_key=idempotency_key,
            expected_version=_version(if_match), context=context,
        )
    except CompositionError as exc:
        _raise(exc)


@router.post(
    "/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/publish",
    response_model=CompositionReceipt, status_code=202,
)
async def publish_semantic_model_composition(
    semantic_model_id: UUID, semantic_model_version_id: UUID,
    body: SemanticModelPublicationComposition, request: Request,
    idempotency_key: str = Header(min_length=8, max_length=128, alias="Idempotency-Key"),
    if_match: str = Header(alias="If-Match"),
) -> CompositionReceipt:
    domain_id, actor_id, context = _scope(request)
    try:
        return await _service(request).publish_semantic_model(
            semantic_model_id=semantic_model_id,
            semantic_model_version_id=semantic_model_version_id,
            body=body, domain_id=domain_id, actor_id=actor_id,
            idempotency_key=idempotency_key,
            expected_version=_version(if_match), context=context,
        )
    except CompositionError as exc:
        _raise(exc)


@router.get("/resources/{resource_type}/{resource_id}/references", response_model=ResourceReferenceGraph)
async def resource_references(resource_type: str, resource_id: UUID, request: Request) -> ResourceReferenceGraph:
    domain_id, _, context = _scope(request)
    try:
        return await _service(request).reference_graph(
            resource_type=resource_type, resource_id=resource_id,
            domain_id=domain_id, context=context,
        )
    except CompositionError as exc:
        _raise(exc)


@router.post(
    "/resources/{resource_type}/{resource_id}/decommission-precheck",
    response_model=ResourceDecommissionResult,
    responses={409: {"model": ResourceDecommissionResult}},
)
async def resource_decommission_precheck(
    resource_type: str, resource_id: UUID,
    body: ResourceDecommissionPrecheck, request: Request,
) -> ResourceDecommissionResult | JSONResponse:
    """只返回阻断关系；归属服务的删除或归档命令必须由调用方另行确认。"""
    graph = await resource_references(resource_type, resource_id, request)
    result = ResourceDecommissionResult(
        action=body.action, allowed=not graph.blockers and not graph.partial,
        graph=graph,
    )
    if not result.allowed:
        return JSONResponse(
            status_code=409,
            content=result.model_dump(mode="json"),
        )
    return result


@router.get("/runs/{run_id}", response_model=RunCompositionView)
async def run_composition(run_id: UUID, request: Request) -> RunCompositionView:
    domain_id, actor_id, context = _scope(request)
    try:
        return await _service(request).run_composition(
            run_id=run_id, domain_id=domain_id,
            actor_id=actor_id, context=context,
        )
    except CompositionError as exc:
        _raise(exc)


@router.get("/receipts/{receipt_id}", response_model=CompositionReceipt)
async def composition_receipt(receipt_id: UUID, request: Request) -> CompositionReceipt:
    domain_id, actor_id, _ = _scope(request)
    try:
        return await _service(request).get_receipt(
            receipt_id=receipt_id, domain_id=domain_id, actor_id=actor_id,
        )
    except CompositionError as exc:
        _raise(exc)
