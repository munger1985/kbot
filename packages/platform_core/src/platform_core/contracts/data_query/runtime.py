"""Data Query Run 的内部稳定命令与回执。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, model_validator

from .query_plan import DataQueryPlanV1, _PlanContract


class CreateDataQueryRun(_PlanContract):
    idempotency_key: str = Field(min_length=1, max_length=128)
    original_question: str = Field(min_length=1, max_length=8000)
    standalone_query: str = Field(min_length=1, max_length=8000)
    plan: DataQueryPlanV1
    agent_id: UUID
    parent_agent_run_id: UUID | None = None
    parent_agent_task_id: UUID | None = None
    deadline_at: datetime | None = None

    @model_validator(mode="after")
    def validate_deadline_timezone(self) -> "CreateDataQueryRun":
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at 必须包含时区")
        return self


class DataQueryRunReceipt(_PlanContract):
    data_query_run_id: UUID
    status: str
    event_sequence_no: int = Field(ge=1)
    idempotent_replay: bool = False


class DataQueryRunView(_PlanContract):
    data_query_run_id: UUID
    status: str
    error_code: str | None = None
    result_available: bool


class DataQueryResultView(_PlanContract):
    data_query_run_id: UUID
    columns: tuple[dict[str, object], ...]
    preview_rows: tuple[dict[str, object], ...]
    row_count: int = Field(ge=0)
    observed_row_count: int = Field(ge=0)
    truncated: bool
    provenance: dict[str, str]


class PlanningSemanticModel(_PlanContract):
    semantic_model_id: UUID
    semantic_model_version: int = Field(ge=1)
    display_name: str
    datasets: tuple[dict[str, object], ...]
    dimensions: tuple[dict[str, object], ...]
    measures: tuple[dict[str, object], ...]
    max_rows: int = Field(ge=1)


class DataQueryPlanningContext(_PlanContract):
    agent_id: UUID
    models: tuple[PlanningSemanticModel, ...]
