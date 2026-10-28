"""DBA 调查 Tool 执行结果 Artifact 契约。"""

from __future__ import annotations

from typing import Literal
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator
from platform_core.contracts.aiops.investigation import DiagnosticSampling, DiagnosticObservationSchedule

from aiops_agent.contracts.artifacts.database import EvidenceGap
from platform_core.contracts.aiops.playbooks import PresentationPreference
from platform_core.contracts.aiops.types import MeasurementSemantics
from platform_core.contracts.aiops.executor import DatabaseObservation


class ToolOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    tool_id: str
    tool_version: str
    status: Literal["SUCCEEDED", "GAP", "SKIPPED"]
    observation: DatabaseObservation | None = None
    gap: EvidenceGap | None = None


class DiagnosticSamplingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested: DiagnosticSampling
    completed_count: int = Field(ge=0, le=4, strict=True)
    started_at: datetime
    finished_at: datetime
    stop_reason: Literal["COMPLETE", "QUERY_GAP", "BUDGET_EXHAUSTED", "RESULT_LIMIT", "INVALID_SAMPLE_TIME"]

    @model_validator(mode="after")
    def validate_sampling_summary(self):
        """摘要必须保留真实次数与带时区的观察区间。"""
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None or self.finished_at < self.started_at:
            raise ValueError("追加采样观察区间无效")
        if self.completed_count > self.requested.count:
            raise ValueError("追加采样实际次数不能超过请求次数")
        if self.stop_reason == "COMPLETE" and self.completed_count != self.requested.count:
            raise ValueError("追加采样未完成全部时点，不能标记完成")
        return self


class DiagnosticObservationSeriesSummary(DiagnosticSamplingSummary):
    """跨进程恢复的未来观察摘要，不声明过去窗口已覆盖。"""

    requested: DiagnosticObservationSchedule


class DbaToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DBA_TOOL_RESULT.v1"] = "DBA_TOOL_RESULT.v1"
    source_type: Literal["PLAYBOOK", "TOOL"]
    source_id: str
    source_version: str
    definition_hash: str
    output_schema: str
    measurement_semantics: MeasurementSemantics
    presentation_kind: PresentationPreference
    status: Literal["SUCCEEDED", "PARTIAL", "FAILED"]
    tool_outcomes: tuple[ToolOutcome, ...]
    sampling: DiagnosticSamplingSummary | None = None
    observation_series: DiagnosticObservationSeriesSummary | None = None

    @model_validator(mode="after")
    def validate_sampled_outcomes(self):
        """正式证据重放时也复核采样摘要，不接受伪造的完成数。"""
        paged = [item.observation for item in self.tool_outcomes if item.observation is not None and item.observation.pagination is not None]
        if paged:
            if len(self.tool_outcomes) != 1 or len(paged) != 1 or self.sampling is not None or self.observation_series is not None:
                raise ValueError("同次游标分页只产生一个观察，不能包装成采样或多个独立来源")
            if self.status != ("PARTIAL" if paged[0].truncated else "SUCCEEDED"):
                raise ValueError("分页截断结果不能包装成完整成功")
        if self.sampling is not None and self.observation_series is not None:
            raise ValueError("不能同时声明短时与持久化采样摘要")
        series = self.sampling or self.observation_series
        if series is not None:
            completed = sum(item.status == "SUCCEEDED" and item.observation is not None for item in self.tool_outcomes)
            if completed != series.completed_count:
                raise ValueError("追加采样摘要与原始观测次数不一致")
            expected = (
                "SUCCEEDED" if series.stop_reason == "COMPLETE"
                else "PARTIAL" if completed else "FAILED"
            )
            if self.status != expected:
                raise ValueError("追加采样状态与原始观测不一致")
            request_ids = [item.observation.executor_request_id for item in self.tool_outcomes if item.observation is not None]
            if len(set(request_ids)) != len(request_ids):
                raise ValueError("追加采样不能复用同一次执行结果")
        return self


def is_turn_evidence_outcome(
    result: DbaToolResult,
    outcome: ToolOutcome,
) -> bool:
    """判断工具结果是否应作为当前 Turn 的可引用证据。"""
    return (
        outcome.observation is not None
        and (
            outcome.tool_id != "db.instance.identity"
            or result.source_type == "TOOL"
        )
    )
