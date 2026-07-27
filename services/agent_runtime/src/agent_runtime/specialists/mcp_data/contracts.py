"""问数和前端图表的稳定 Artifact 契约。"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QueryResult(_Contract):
    query_result_id: UUID
    profile: str
    question: str
    rows: tuple[dict[str, Any], ...]
    row_count: int = Field(ge=0)
    upstream_row_count: int = Field(ge=0)
    truncated: bool = False
    status: str = "READY"


class EChartsResult(_Contract):
    chart_type: str
    title: str | None = None
    option: dict[str, Any]
    query_result_id: UUID
