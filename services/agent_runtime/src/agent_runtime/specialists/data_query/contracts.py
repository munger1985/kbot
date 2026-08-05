"""两种问数 Provider 共用的稳定 Artifact 契约。"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
    row_count: int = Field(ge=0)
    truncated: bool = False
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any]
