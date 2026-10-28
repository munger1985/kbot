"""Conversation Turn 证据充分性和自然回答 Artifact 契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_serializer

from aiops_agent.contracts.investigation import DiagnosisAnswerDraft, InvestigationRecord
from aiops_agent.contracts.measurements import EvidenceProvenance, DerivedMeasurement
from aiops_agent.contracts.calculation import CalculationEvaluation
from aiops_agent.contracts.series_calculation import SeriesComputation
from aiops_agent.contracts.evidence import SourcePaginationObservation
from aiops_agent.contracts.physical_identity import SourceIdentityWitness
from aiops_agent.contracts.sql_plan import SqlPlanDocument

from platform_core.contracts.aiops import (
    AnswerBlockType,
    MeasurementSemantics,
    SufficiencyStatus,
    InvestigationAssessment,
)
from platform_core.contracts.aiops.playbooks import PresentationPreference
from platform_core.contracts.aiops.executor import ObjectAccessObservation, DiagnosticPaginationObservation
from platform_core.contracts.visualization import ChartSpec


class TurnEvidenceFact(BaseModel):
    """可独立引用的一次受控 Tool 观测。"""

    model_config = ConfigDict(extra="forbid")

    evidence_ref: str = Field(min_length=1, max_length=512)
    artifact_id: str = Field(min_length=1, max_length=64)
    source_id: str = Field(min_length=1, max_length=128)
    step_id: str = Field(min_length=1, max_length=128)
    tool_id: str = Field(min_length=1, max_length=128)
    trust_level: Literal[
        "SOURCE_VERIFIED", "USER_PROVIDED", "MODEL_INFERENCE"
    ] = "SOURCE_VERIFIED"
    measurement_semantics: MeasurementSemantics
    presentation_kind: PresentationPreference
    captured_at: str
    columns: tuple[dict[str, Any], ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int = Field(ge=0)
    truncated: bool = False
    followup_run_id: str | None = None
    warnings: tuple[str, ...] = ()
    visualizations: tuple[ChartSpec, ...] = ()
    provenance: EvidenceProvenance | None = None
    sql_plan: SqlPlanDocument | None = None
    object_access: tuple[ObjectAccessObservation, ...] = Field(default=(), max_length=1)
    pagination: DiagnosticPaginationObservation | None = None
    source_pagination: SourcePaginationObservation | None = None
    derivations: tuple[DerivedMeasurement, ...] = ()
    calculation_graphs: tuple[CalculationEvaluation, ...] = Field(default=(), max_length=128)
    series_computations: tuple[SeriesComputation, ...] = Field(default=(), max_length=128)
    source_identity_witnesses: tuple[SourceIdentityWitness, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_series_budget(self):
        if self.source_identity_witnesses and (
            self.tool_id != "metric.query_range" or self.source_id != "monitoring.overview"
            or self.trust_level != "SOURCE_VERIFIED" or any(
                not item.input_ref.startswith(f"artifact:{self.artifact_id}#/observations/")
                or self.provenance is None or str(item.target_id) != self.provenance.target_id
                for item in self.source_identity_witnesses
            )
        ):
            raise ValueError("监控身份见证必须保留受控来源、目标及原始序列引用")
        if self.tool_id == "evidence.sql_plan.controlled_estimate" or self.sql_plan is not None and self.sql_plan.origin == "CONTROLLED_ESTIMATE":
            if (self.sql_plan is None or self.sql_plan.origin != "CONTROLLED_ESTIMATE"
                or self.tool_id != "evidence.sql_plan.controlled_estimate" or self.source_id != "sql.plan_capture"
                or self.trust_level != "SOURCE_VERIFIED" or self.measurement_semantics != "NOT_APPLICABLE"
                or self.provenance is None or self.followup_run_id is not None
                or self.sql_plan.observed_at != self.captured_at or self.row_count != len(self.sql_plan.nodes)
                or self.row_count != len(self.rows) or self.truncated
                or self.sql_plan.physical_identity != dict(self.provenance.physical_identity)):
                raise ValueError("受控估算必须保留独立来源、完整节点和采集时点，不得伪装为实际运行")
        if self.sql_plan is not None and self.sql_plan.origin == "USER_UPLOAD" and (
            self.trust_level != "USER_PROVIDED" or self.tool_id != "artifact.sql_plan.import"
            or self.measurement_semantics != "NOT_APPLICABLE" or self.provenance is not None
            or self.row_count != len(self.sql_plan.nodes) or self.row_count != len(self.rows) or self.truncated
        ):
            raise ValueError("导入计划必须保留人工来源和完整安全投影，不能自证数据库来源")
        if self.sql_plan is not None and self.sql_plan.origin == "DATABASE_CURSOR" and (
            self.trust_level != "SOURCE_VERIFIED" or self.tool_id != "db.oracle.statement.plan"
            or self.measurement_semantics != "CURRENT_ACTIVITY" or self.provenance is None
            or self.sql_plan.observed_at != self.captured_at or self.row_count != len(self.sql_plan.nodes)
            or self.row_count != len(self.rows) or self.truncated
            or self.sql_plan.physical_identity != dict(self.provenance.physical_identity)
        ):
            raise ValueError("缓存计划必须与受控工具、行数、采集时间和原生来源一致")
        if self.source_pagination is not None and (
            self.pagination is not None or sum(self.source_pagination.returned_page_rows) != self.row_count
            or self.row_count != len(self.rows)
            or self.truncated != (self.source_pagination.stop_reason != "EXHAUSTED")
        ):
            raise ValueError("来源分页事实与原始行数或截断状态不一致")
        if self.pagination is not None and (
            sum(self.pagination.returned_page_rows) != self.row_count or self.row_count != len(self.rows)
            or self.truncated != (self.pagination.stop_reason != "EXHAUSTED")
        ):
            raise ValueError("证据分页范围与原始结果不一致")
        if sum(len(item.samples) for item in self.series_computations) > 8192:
            raise ValueError("监控计算总输入超出预算")
        indexes = [item.row_index for item in self.series_computations]
        if len(indexes) != len(set(indexes)):
            raise ValueError("同一摘要行不能绑定多份监控计算")
        return self

    @field_serializer("rows", when_used="json")
    def exact_json_rows(self, rows):
        from decimal import Decimal
        def exact(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, Decimal) or isinstance(value, int) and abs(value) > 9007199254740991:
                return str(value)
            if isinstance(value, (list, tuple)):
                return [exact(item) for item in value]
            if isinstance(value, dict):
                return {key: exact(item) for key, item in value.items()}
            return value
        return exact(rows)


class TurnEvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=128)
    step_id: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=1, max_length=128)
    detail: str = Field(min_length=1, max_length=2000)
    retryable: bool = False
    evidence_ref: str | None = Field(default=None, min_length=1, max_length=512)
    object_access: tuple[ObjectAccessObservation, ...] = Field(default=(), max_length=1)


class DbaSufficiencyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DBA_SUFFICIENCY.v1"] = "DBA_SUFFICIENCY.v1"
    status: SufficiencyStatus
    evidence: tuple[TurnEvidenceFact, ...] = ()
    gaps: tuple[TurnEvidenceGap, ...] = ()
    reasons: tuple[str, ...] = ()
    clarification_question: str | None = Field(default=None, max_length=2000)
    investigation: InvestigationAssessment | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "DbaSufficiencyAssessment":
        if (
            self.status == SufficiencyStatus.NEEDS_CLARIFICATION
            and not self.clarification_question
        ):
            raise ValueError("NEEDS_CLARIFICATION 必须包含澄清问题")
        return self


class DbaAnswerDraft(BaseModel):
    """模型只生成自然语言正文，不得生成或改写事实表格。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DBA_ANSWER_DRAFT.v1"] = "DBA_ANSWER_DRAFT.v1"
    markdown: str = Field(min_length=1, max_length=32000)
    evidence_refs: tuple[str, ...] = ()


class TurnAnswerBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_type: AnswerBlockType
    schema_version: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any]
    evidence_refs: tuple[str, ...] = ()


class AIOpsTurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["AIOPS_TURN_RESULT.v1"] = "AIOPS_TURN_RESULT.v1"
    status: Literal["COMPLETED", "PARTIAL", "WAITING_USER"]
    sufficiency_status: SufficiencyStatus
    blocks: tuple[TurnAnswerBlock, ...]
    evidence: tuple[TurnEvidenceFact, ...] = ()
    evidence_gaps: tuple[TurnEvidenceGap, ...] = ()
    assessment_reasons: tuple[str, ...] = ()
    answer_streamed: bool = False
    model_receipt: dict[str, Any] | None = None
    investigation_record: InvestigationRecord | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "AIOpsTurnResult":
        if not self.blocks:
            raise ValueError("Turn 回答至少包含一个展示块")
        if self.investigation_record is not None:
            self.investigation_record.validate_observations(self.evidence)
            self.investigation_record.validate_access_gaps(self.evidence_gaps)
        return self


class DbaAnswerProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["answer.delta", "thinking.delta"]
    event_key: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any]
