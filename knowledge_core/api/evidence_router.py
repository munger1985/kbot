"""V2 Evidence query and Citation Pack endpoint."""
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from knowledge_core.application.evidence_retrieval import EvidenceScope

router = APIRouter(prefix="/api/v2/knowledge/retrieval", tags=["Knowledge Core Evidence V2"])


class EvidenceCandidateRequest(BaseModel):
    collection_id: int
    bundle_id: int
    bundle_revision_id: int
    document_version_ids: list[int] = Field(default_factory=list, max_length=128)


class EvidenceSearchRequest(BaseModel):
    domain_id: int = Field(gt=0)
    agent_id: int = Field(gt=0)
    query: str = Field(min_length=1, max_length=8000)
    candidates: list[EvidenceCandidateRequest] = Field(min_length=1, max_length=128)
    query_vectors: dict[int, list[float]] | None = None
    max_evidence: int = Field(default=12, ge=1, le=100)
    context_limit: int = Field(default=4, ge=0, le=20)
    max_security_level: int = Field(default=3, ge=0, le=3)


@router.post("/evidence")
async def search_evidence(payload: EvidenceSearchRequest, request: Request):
    try:
        requested_collections = sorted({item.collection_id for item in payload.candidates})
        scoped_collection_ids = await request.app.state.kc_scope_service.resolve_agent_collections(
            domain_id=payload.domain_id, agent_id=payload.agent_id,
            collection_ids=requested_collections,
        )
        if set(scoped_collection_ids) != set(requested_collections):
            raise ValueError("one or more candidate Collections are disabled")
        citations = await request.app.state.kc_evidence_service.retrieve(
            scopes=[EvidenceScope(
                collection_id=item.collection_id, bundle_id=item.bundle_id,
                bundle_revision_id=item.bundle_revision_id,
                document_version_ids=tuple(item.document_version_ids),
            ) for item in payload.candidates],
            query=payload.query, query_vectors=payload.query_vectors,
            max_evidence=payload.max_evidence, context_limit=payload.context_limit,
            max_security_level=payload.max_security_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_EVIDENCE_QUERY", "message": str(exc)}) from exc
    return {"citations": [asdict(citation) for citation in citations]}
