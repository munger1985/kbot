"""Evidence 查询与 Citation Pack 内部端点。"""
from dataclasses import asdict
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from platform_core.contracts import INTERNAL_API_V1
from platform_core.security import require_domain_match
from knowledge_core.application.evidence_retrieval import EvidenceScope

router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/retrieval",
    tags=["Knowledge Core Evidence"],
)


class EvidenceCandidateRequest(BaseModel):
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    document_version_ids: list[UUID] = Field(default_factory=list, max_length=128)


class EvidenceSearchRequest(BaseModel):
    domain_id: int = Field(gt=0)
    agent_id: UUID
    query: str = Field(min_length=1, max_length=8000)
    candidates: list[EvidenceCandidateRequest] = Field(min_length=1, max_length=128)
    query_vectors: dict[UUID, list[float]] | None = None
    max_evidence: int = Field(default=12, ge=1, le=100)
    context_limit: int = Field(default=4, ge=0, le=20)
    max_security_level: int = Field(default=3, ge=0, le=3)
    do_rerank: bool = False


@router.post("/evidence")
async def search_evidence(payload: EvidenceSearchRequest, request: Request):
    require_domain_match(request, payload.domain_id)
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
        rerank_report = {
            "enabled": False,
            "stage": "EVIDENCE_GROUP",
            "status": "DISABLED",
        }
        warnings: list[str] = []
        if payload.do_rerank and citations:
            citations, rerank_report, warnings = (
                await request.app.state.kc_llm_reranker.rerank_evidence(
                    query=payload.query,
                    citations=citations,
                )
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_EVIDENCE_QUERY", "message": str(exc)}) from exc
    return {
        "citations": [asdict(citation) for citation in citations],
        "rerank": rerank_report,
        "warnings": warnings,
    }
