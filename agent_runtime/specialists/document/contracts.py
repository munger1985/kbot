"""Document Specialist 的稳定 Artifact Schema。"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Citation(_Contract):
    citation_label: str
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    document_id: UUID
    document_version_id: UUID
    evidence_ids: tuple[UUID, ...]
    title: str
    bundle_title: str | None = None
    external_document_id: str | None = None
    document_role: str | None = None
    excerpt: str
    locator: dict[str, Any] = Field(default_factory=dict)
    heading_path: tuple[str, ...] = ()
    relevance_reason: str
    source_hash: str | None = None


class RetrievalCoverage(_Contract):
    candidate_bundle_count: int = Field(ge=0)
    selected_document_count: int = Field(ge=0)
    selected_evidence_count: int = Field(ge=0)
    uncovered_aspects: tuple[str, ...] = ()


class CitationPack(_Contract):
    question: str
    query_plan: dict[str, Any]
    bundle_candidates: tuple[dict[str, Any], ...]
    citations: tuple[Citation, ...]
    coverage: RetrievalCoverage


class DocumentRetrievalResult(_Contract):
    status: str
    citation_pack: CitationPack
    retrieval_report: dict[str, Any]
    coverage_gaps: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    kc_request_ids: tuple[str, ...] = ()
