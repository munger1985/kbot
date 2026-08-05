"""可视化 Artifact 契约。"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class EChartsResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chart_type: str
    title: str | None = None
    option: dict[str, Any]
    query_result_id: UUID
