"""两种问数 Provider 共用的稳定 Artifact 契约。"""

import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )


class QueryResult(_Contract):
    contract_schema: Literal["QUERY_RESULT.v1"] = Field(
        default="QUERY_RESULT.v1",
        validation_alias="schema",
        serialization_alias="schema",
    )
    query_result_id: UUID
    provider: Literal["MCP", "SEMANTIC"]
    columns: tuple[dict[str, Any], ...]
    rows: tuple[dict[str, Any], ...]
    supporting_columns: tuple[dict[str, Any], ...] = ()
    supporting_rows: tuple[dict[str, Any], ...] = ()
    supporting_query_result_id: UUID | None = None
    row_count: int = Field(ge=0)
    truncated: bool = False
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any]


class KMTopicExpansion(_Contract):
    """KM 主题原语言与英文补充检索词。"""

    source_language: str = Field(min_length=1, max_length=16)
    original_topic: str = Field(min_length=1, max_length=256)
    english_topics: tuple[str, ...] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def validate_english_topics(self) -> "KMTopicExpansion":
        normalized: set[str] = set()
        for topic in self.english_topics:
            value = topic.strip()
            if not value or not re.search(r"[A-Za-z]", value):
                raise ValueError("english_topics 每项必须包含英文字符")
            key = value.casefold()
            if key in normalized:
                raise ValueError("english_topics 不能包含重复词")
            normalized.add(key)
        return self
