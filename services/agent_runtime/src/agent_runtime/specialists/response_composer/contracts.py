"""Response Composer 的公开结果 Schema。"""

import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


_CITATION_PATTERN = re.compile(r"\[([A-Z]\d+)\]")


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReferenceCard(_Contract):
    """公开 App 知识引用，不公开存储句柄或旧 KC 身份。"""

    reference_type: Literal["APP_KNOWLEDGE"] = "APP_KNOWLEDGE"
    citation_label: str
    run_id: UUID
    retrieval_ref: UUID
    document_version_id: str
    parse_run_id: str
    evidence_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    owner_app_id: Literal["knowledge_retrieval", "aiops"]
    locators: tuple[dict[str, Any], ...]
    role: Literal["PRIMARY", "CONTEXT", "REFERENCED"]
    title: str = ""


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


class GroundedAnswer(_Contract):
    answer: str
    status: str
    used_citation_labels: tuple[str, ...] = ()
    references: tuple[
        ReferenceCard | AIOpsReferenceCard | QueryResultReferenceCard, ...
    ] = ()
    query_results: tuple[dict[str, Any], ...] = ()
    visualizations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    knowledge_composition: dict[str, Any] | None = None

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
        return self
