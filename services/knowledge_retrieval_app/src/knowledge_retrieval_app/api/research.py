"""知识检索应用 X Search 内部 API。"""

from datetime import date
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Header, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from knowledge_retrieval_app.api.context import (
    auth_context, idempotency_key, raise_application_error, require_actor, run_response,
)
from knowledge_retrieval_app.application import KnowledgeRetrievalApplicationError, CreateResearchCommand, ResearchRunService


router = APIRouter(prefix="/internal/v1/knowledge-retrieval/x-search/runs", tags=["KnowledgeRetrieval X Search"])


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchCreateRequest(_Request):
    domain_id: int = Field(ge=1)
    model_id: UUID
    agent_id: UUID
    input: str = Field(min_length=1, max_length=32000)
    from_date: date | None = None
    to_date: date | None = None
    allowed_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    excluded_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    enable_image_understanding: bool = False
    enable_video_understanding: bool = False
    search_context_size: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"

    @model_validator(mode="after")
    def validate_filters(self) -> "ResearchCreateRequest":
        if self.allowed_x_handles and self.excluded_x_handles:
            raise ValueError("allowed_x_handles 与 excluded_x_handles 不能同时设置")
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("from_date 不能晚于 to_date")
        return self


def _service(request: Request) -> ResearchRunService:
    return request.app.state.research_service


@router.get("")
async def list_research_runs(
    domain_id: int,
    request: Request,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    actor_id = require_actor(request, domain_id)
    return await _service(request).list(
        domain_id=domain_id, actor_id=actor_id, status=status, limit=limit,
    )


@router.post("")
async def create_research_run(
    payload: ResearchCreateRequest,
    request: Request,
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
):
    context = auth_context(request)
    try:
        view, created = await _service(request).create(CreateResearchCommand(
            actor_id=require_actor(request, payload.domain_id),
            request_id=context.request_id,
            trace_id=context.trace_id,
            idempotency_key=idempotency_key(request, idempotency_key_header),
            **payload.model_dump(),
        ))
        return run_response(view, created)
    except KnowledgeRetrievalApplicationError as exc:
        raise_application_error(exc)


@router.get("/{run_id}")
async def get_research_run(run_id: UUID, domain_id: int, request: Request):
    actor_id = require_actor(request, domain_id)
    try:
        return await _service(request).get(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
    except KnowledgeRetrievalApplicationError as exc:
        raise_application_error(exc)


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_research_run(run_id: UUID, domain_id: int, request: Request):
    actor_id = require_actor(request, domain_id)
    try:
        await _service(request).delete(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
    except KnowledgeRetrievalApplicationError as exc:
        raise_application_error(exc)


@router.get("/{run_id}/events")
async def list_research_events(run_id: UUID, domain_id: int, request: Request):
    actor_id = require_actor(request, domain_id)
    try:
        return await _service(request).list_events(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
    except KnowledgeRetrievalApplicationError as exc:
        raise_application_error(exc)


@router.get("/{run_id}/sources")
async def list_research_sources(run_id: UUID, domain_id: int, request: Request):
    actor_id = require_actor(request, domain_id)
    try:
        return await _service(request).list_sources(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
    except KnowledgeRetrievalApplicationError as exc:
        raise_application_error(exc)
