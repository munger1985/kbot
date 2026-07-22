"""V2 Discovery-stage query endpoint."""
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v2/knowledge/discovery", tags=["Knowledge Core Discovery V2"])


class DiscoverySearchRequest(BaseModel):
    domain_id: int = Field(gt=0)
    agent_id: int = Field(gt=0)
    query: str = Field(min_length=1, max_length=8000)
    collection_ids: list[int] = Field(min_length=1, max_length=128)
    query_vectors: dict[int, list[float]] | None = None
    per_channel_limit: int = Field(default=20, ge=1, le=100)
    per_collection_limit: int = Field(default=20, ge=1, le=100)
    max_security_level: int = Field(default=3, ge=0, le=3)


@router.post("/search")
async def search_discovery(payload: DiscoverySearchRequest, request: Request):
    try:
        scoped_collection_ids = await request.app.state.kc_scope_service.resolve_agent_collections(
            domain_id=payload.domain_id, agent_id=payload.agent_id,
            collection_ids=payload.collection_ids,
        )
        candidates = await request.app.state.kc_discovery_service.discover(
            collection_ids=scoped_collection_ids,
            query=payload.query,
            query_vectors=payload.query_vectors,
            per_channel_limit=payload.per_channel_limit,
            per_collection_limit=payload.per_collection_limit,
            max_security_level=payload.max_security_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_DISCOVERY_QUERY", "message": str(exc)}) from exc
    return {"candidates": [asdict(candidate) for candidate in candidates]}
