"""AIOps 检索响应；不公开 App 委托或 Runtime 授权绑定。"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeChannelFailure(BaseModel):
    channel: str
    reason: str


class KnowledgeSearchCoverage(BaseModel):
    model_config = ConfigDict(extra="allow")
    scope: Literal["COMPLETE", "PARTIAL", "NOT_VERIFIED"]
    ranking: Literal["GLOBAL", "LIMITED", "NOT_APPLICABLE"]
    gaps: list[str]
    channel_failures: list[KnowledgeChannelFailure]


class KnowledgeCitation(BaseModel):
    document_version_id: UUID
    parse_run_id: UUID
    index_generation_id: UUID
    quote: str
    locator: dict[str, Any]


class KnowledgePrimaryEvidence(KnowledgeCitation):
    evidence_key: str
    rank: int = Field(ge=1)
    shard: int = Field(ge=1)


class KnowledgeRelatedEvidence(BaseModel):
    evidence_key: str
    document_version_id: UUID
    parse_run_id: UUID
    index_generation_id: UUID
    quote: str | None = None
    locator: dict[str, Any]
    readable: bool


class KnowledgeEvidenceGroup(BaseModel):
    primary: KnowledgePrimaryEvidence
    related: list[KnowledgeRelatedEvidence]
    relation_gaps: list[str]


class AIOpsKnowledgeSearchResult(BaseModel):
    asset_id: UUID
    asset_version_id: UUID
    asset_kind: str
    title: str
    version: int
    authority: str | None = None
    applicability: str
    business_rank: int | None = None
    citation_pack: list[KnowledgeCitation]
    evidence_groups: list[KnowledgeEvidenceGroup] = Field(default_factory=list)
    ranking: dict[str, Any] | None = None
    matched_dimensions: list[str] | None = None
    mismatched_dimensions: list[str] | None = None
    unknown_dimensions: list[str] | None = None
    topics: list[str] | None = None
    reuse_context: dict[str, Any] | None = None
    applicability_decision: dict[str, Any] | None = None


class KnowledgeSearchResponse(BaseModel):
    status: str
    retrieval_conclusion: Literal["EVIDENCE_FOUND", "NO_ELIGIBLE_RESOURCES", "NO_MATCH", "NOT_VERIFIED"] | None = None
    coverage: KnowledgeSearchCoverage
    results: list[AIOpsKnowledgeSearchResult]
    warnings: list[str]
