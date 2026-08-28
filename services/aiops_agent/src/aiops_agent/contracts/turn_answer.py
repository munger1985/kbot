"""Conversation Turn 证据充分性和自然回答 Artifact 契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from platform_core.contracts.aiops import (
    AnswerBlockType,
    MeasurementSemantics,
    SufficiencyStatus,
    InvestigationAssessment,
)
from platform_core.contracts.aiops.skills import PresentationPreference


class TurnEvidenceFact(BaseModel):
    """可独立引用的一次受控 Tool 观测。"""

    model_config = ConfigDict(extra="forbid")

    evidence_ref: str = Field(min_length=1, max_length=512)
    artifact_id: str = Field(min_length=1, max_length=64)
    skill_id: str = Field(min_length=1, max_length=128)
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
    warnings: tuple[str, ...] = ()


class TurnEvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(min_length=1, max_length=128)
    step_id: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=1, max_length=128)
    detail: str = Field(min_length=1, max_length=2000)
    retryable: bool = False


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
    answer_streamed: bool = False
    model_receipt: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "AIOpsTurnResult":
        if not self.blocks:
            raise ValueError("Turn 回答至少包含一个展示块")
        return self


class DbaAnswerProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["answer.delta", "thinking.delta"]
    event_key: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any]
