"""Response Composer 的公开结果 Schema。"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReferenceCard(_Contract):
    reference_type: str = "DOCUMENT"
    citation_label: str
    collection_id: UUID
    bundle_id: UUID
    document_id: UUID
    document_version_id: UUID
    title: str
    locator: dict[str, Any] = Field(default_factory=dict)


class GroundedAnswer(_Contract):
    answer: str
    status: str
    used_citation_labels: tuple[str, ...] = ()
    references: tuple[ReferenceCard, ...] = ()
    warnings: tuple[str, ...] = ()
