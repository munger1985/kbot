"""Response Composer 的公开结果 Schema。"""

import re
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


_CITATION_PATTERN = re.compile(r"\[([A-Z]\d+)\]")


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReferenceCard(_Contract):
    reference_type: Literal["DOCUMENT"] = "DOCUMENT"
    citation_label: str
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    document_id: UUID
    document_version_id: UUID
    title: str
    locator: dict[str, Any] = Field(default_factory=dict)
    locator_schema_version: str


class AIOpsReferenceCard(_Contract):
    reference_type: Literal["AIOPS"] = "AIOPS"
    citation_label: str
    ops_run_id: UUID
    delegation_id: UUID
    status: str
    resource_url: str
    root_cause_grade: str | None = None
    artifact_id: UUID | None = None
    content_hash: str | None = None


class QueryResultReferenceCard(_Contract):
    reference_type: Literal["QUERY_RESULT"] = "QUERY_RESULT"
    citation_label: str
    query_result_id: UUID
    provider: Literal["MCP", "SEMANTIC"]
    row_count: int = Field(ge=0)


class PresentedAssetReference(_Contract):
    """回答正文中一个 Asset 与其引用证据的内部映射。"""

    primary_citation_label: str = Field(pattern=r"^[A-Z]\d+$")
    supporting_citation_labels: tuple[str, ...] = ()

    @field_validator("supporting_citation_labels")
    @classmethod
    def validate_supporting_labels(cls, value: tuple[str, ...]):
        if any(re.fullmatch(r"[A-Z]\d+", label) is None for label in value):
            raise ValueError("supporting_citation_labels 包含无效引用标签")
        if len(set(value)) != len(value):
            raise ValueError("supporting_citation_labels 不得包含重复标签")
        return value

    @model_validator(mode="after")
    def validate_primary_not_supporting(self):
        if self.primary_citation_label in self.supporting_citation_labels:
            raise ValueError(
                "primary_citation_label 不得重复出现在 supporting 中"
            )
        return self


class GroundedAnswer(_Contract):
    answer: str
    status: str
    used_citation_labels: tuple[str, ...] = ()
    references: tuple[
        ReferenceCard | AIOpsReferenceCard | QueryResultReferenceCard, ...
    ] = ()
    presented_assets: tuple[PresentedAssetReference, ...] = ()
    query_results: tuple[dict[str, Any], ...] = ()
    visualizations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_reference_projection(self):
        """引用卡片只能投影回答正文实际使用的标签。"""
        mentioned = tuple(dict.fromkeys(_CITATION_PATTERN.findall(self.answer)))
        used = tuple(dict.fromkeys(self.used_citation_labels))
        reference_labels = tuple(
            dict.fromkeys(item.citation_label for item in self.references)
        )
        if len(used) != len(self.used_citation_labels):
            raise ValueError("used_citation_labels 不得包含重复标签")
        if len(reference_labels) != len(self.references):
            raise ValueError("references 不得包含重复引用标签")
        if set(used) != set(mentioned):
            raise ValueError("回答正文引用与 used_citation_labels 不一致")
        if set(reference_labels) != set(used):
            raise ValueError("引用列表包含正文未使用的证据")
        presented_labels: list[str] = []
        document_labels = {
            item.citation_label
            for item in self.references
            if isinstance(item, ReferenceCard)
        }
        for asset in self.presented_assets:
            labels = (
                asset.primary_citation_label,
                *asset.supporting_citation_labels,
            )
            if any(label not in used for label in labels):
                raise ValueError("presented_assets 包含正文未使用的引用")
            if any(label not in document_labels for label in labels):
                raise ValueError("presented_assets 只能引用 DOCUMENT 证据")
            presented_labels.extend(labels)
        if len(set(presented_labels)) != len(presented_labels):
            raise ValueError("同一引用不得映射到多个 presented_assets")
        return self
