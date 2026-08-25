"""Discovery 阶段的内部查询端点。"""
from dataclasses import asdict
from time import monotonic
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from platform_core.contracts import (
    INTERNAL_API_V1,
    KNOWLEDGE_DISCOVERY_REVISION_SCOPE_LIMIT,
)
from platform_core.security import require_domain_match

router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/discovery",
    tags=["Knowledge Core Discovery"],
)


class DiscoverySearchRequest(BaseModel):
    domain_id: int = Field(gt=0)
    agent_id: UUID
    query: str = Field(min_length=1, max_length=8000)
    collection_ids: list[UUID] = Field(min_length=1, max_length=128)
    bundle_revision_ids: list[UUID] = Field(
        default_factory=list,
        max_length=KNOWLEDGE_DISCOVERY_REVISION_SCOPE_LIMIT,
    )
    query_vectors: dict[UUID, list[float]] | None = None
    per_channel_limit: int = Field(default=20, ge=1, le=100)
    per_collection_limit: int = Field(default=20, ge=1, le=100)
    max_security_level: int = Field(default=3, ge=0, le=3)
    coverage_mode: Literal["BREADTH", "BALANCED"] = "BALANCED"
    run_id: UUID | None = None
    task_id: UUID | None = None


@router.post("/search")
async def search_discovery(payload: DiscoverySearchRequest, request: Request):
    require_domain_match(request, payload.domain_id)
    started_at = monotonic()
    try:
        scoped_collection_ids = await request.app.state.kc_scope_service.resolve_agent_collections(
            domain_id=payload.domain_id, agent_id=payload.agent_id,
            collection_ids=payload.collection_ids,
        )
        candidates, diagnostics = (
            await request.app.state.kc_discovery_service
            .discover_with_diagnostics(
                collection_ids=scoped_collection_ids,
                bundle_revision_ids=payload.bundle_revision_ids,
                query=payload.query,
                query_vectors=payload.query_vectors,
                per_channel_limit=payload.per_channel_limit,
                per_collection_limit=payload.per_collection_limit,
                max_security_level=payload.max_security_level,
            )
        )
        warnings: list[str] = list(diagnostics.get("warnings") or [])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_DISCOVERY_QUERY", "message": str(exc)}) from exc
    logger.info(
        "KC 对象发现完成 | event=kc.discovery.completed | run_id={} "
        "| task_id={} | trace_id={} | text_hits={} | vector_hits={} "
        "| raw_hits={} | bundle_candidates={} | duration_ms={:.2f}",
        payload.run_id or "-",
        payload.task_id or "-",
        getattr(request.state.auth_context, "trace_id", "-"),
        diagnostics["text_hits"],
        diagnostics["vector_hits"],
        diagnostics["raw_hits"],
        len(candidates),
        (monotonic() - started_at) * 1000,
    )
    return {
        "candidates": [
            {
                key: value
                for key, value in asdict(candidate).items()
                if key != "profile_text"
            }
            for candidate in candidates
        ],
        "diagnostics": diagnostics,
        "warnings": warnings,
    }
