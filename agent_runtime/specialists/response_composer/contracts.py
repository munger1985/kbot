"""Response Composer 的公开结果 Schema。"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReferenceCard(_Contract):
    reference_type: Literal["DOCUMENT"] = "DOCUMENT"
    citation_label: str
    collection_id: UUID
    bundle_id: UUID
    document_id: UUID
    document_version_id: UUID
    title: str
    locator: dict[str, Any] = Field(default_factory=dict)


class AIOpsReferenceCard(_Contract):
    reference_type: Literal["AIOPS"] = "AIOPS"
    citation_label: str
    ops_run_id: UUID
    delegation_id: UUID
    status: str
    root_cause_grade: str | None = None
    artifact_id: UUID | None = None
    content_hash: str | None = None


class GroundedAnswer(_Contract):
    answer: str
    status: str
    used_citation_labels: tuple[str, ...] = ()
    references: tuple[ReferenceCard | AIOpsReferenceCard, ...] = ()
    warnings: tuple[str, ...] = ()
