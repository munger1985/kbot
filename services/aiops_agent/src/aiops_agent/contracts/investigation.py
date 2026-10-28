"""问题驱动诊断的严格判断与不可变调查记录，不赋予执行权限。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.contracts.sql_diagnosis import EvidenceFieldReference, SqlRecommendation


class InvestigationClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=r"^c[0-9]+$")
    kind: Literal["FACT", "CAUSE", "RISK", "PREDICTION", "EXCLUDED", "INTERPRETATION"]
    subject: str = Field(min_length=1, max_length=1000)
    statement: str = Field(min_length=1, max_length=3000)
    level: Literal["OBSERVED", "SUPPORTED", "HYPOTHESIS", "UNKNOWN"]
    applicable_time: str = Field(min_length=1, max_length=1000)
    evidence_fields: tuple[EvidenceFieldReference, ...] = Field(default=(), max_length=24)
    counterevidence_fields: tuple[EvidenceFieldReference, ...] = Field(default=(), max_length=24)
    reasoning: str = Field(min_length=1, max_length=3000)
    alternatives: tuple[str, ...] = Field(default=(), max_length=8)
    limitations: tuple[str, ...] = Field(default=(), max_length=12)
    revision_state: Literal["ACTIVE", "REFUTED", "SUPERSEDED"] = "ACTIVE"

    @model_validator(mode="after")
    def validate_strength(self):
        if self.level in {"OBSERVED", "SUPPORTED"} and not self.evidence_fields:
            raise ValueError("已观测或有支持的主张必须定位原始行与字段")
        if self.kind != "FACT" and self.level == "OBSERVED":
            raise ValueError("只有直接事实可以标记为已观测，原因和预测不是物理观测")
        if self.kind != "FACT" and not self.limitations:
            raise ValueError("解释、原因、风险、预测及排除必须说明适用限制")
        if self.counterevidence_fields and self.level == "OBSERVED":
            raise ValueError("存在反证的主张不能直接作为已确认事实")
        return self


class DiagnosisAnswerDraft(BaseModel):
    """所有普通诊断先提交逐项判断，再生成用户正文。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DBA_DIAGNOSTIC_ASSESSMENT.v1"] = "DBA_DIAGNOSTIC_ASSESSMENT.v1"
    claims: tuple[InvestigationClaim, ...] = Field(min_length=1, max_length=16)
    recommendations: tuple[SqlRecommendation, ...] = Field(default=(), max_length=12)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=12)
    goal_satisfied: bool

    @model_validator(mode="after")
    def validate_links(self):
        ids = {item.claim_id for item in self.claims}
        if len(ids) != len(self.claims):
            raise ValueError("主张 ID 不能重复")
        action_ids = {item.recommendation_id for item in self.recommendations}
        if len(action_ids) != len(self.recommendations):
            raise ValueError("建议 ID 不能重复")
        active_ids = {item.claim_id for item in self.claims if item.revision_state == "ACTIVE"}
        for action in self.recommendations:
            if not set(action.conclusion_ids) <= active_ids:
                raise ValueError("建议只能引用当前有效主张")
            if action.intent in {"MITIGATE", "REPAIR"} and not action.requires_approval:
                raise ValueError("处置建议不能绕过独立审批")
        if self.goal_satisfied and not any(
            item.level in {"OBSERVED", "SUPPORTED"} and item.revision_state == "ACTIVE"
            for item in self.claims
        ):
            raise ValueError("仅有假设或未知时不能声明目标已回答")
        return self


class InvestigationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str
    artifact_id: str
    tool_id: str
    source_id: str
    captured_at: str
    measurement_semantics: str
    trust_level: Literal["SOURCE_VERIFIED", "USER_PROVIDED", "MODEL_INFERENCE"]
    row_count: int = Field(ge=0)
    truncated: bool


class InvestigationGoal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_id: str
    status: Literal["SATISFIED", "PARTIAL", "BLOCKED", "NOT_APPLICABLE"]
    claim_refs: tuple[str, ...] = ()
    blocking_gaps: tuple[str, ...] = ()


class InvestigationRecord(BaseModel):
    """冻结于 Turn 结果 Artifact；正文、报告与案例共享相同修订。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["DBA_INVESTIGATION_RECORD.v1"] = "DBA_INVESTIGATION_RECORD.v1"
    record_id: str
    revision: int = Field(default=1, ge=1)
    previous_revision: str | None = None
    target_id: str
    run_id: str
    task_id: str
    original_request: str
    task_kind: str
    generated_at: str
    observations: tuple[InvestigationObservation, ...]
    claims: tuple[InvestigationClaim, ...]
    recommendations: tuple[SqlRecommendation, ...]
    goal_assessments: tuple[InvestigationGoal, ...]
    unresolved_questions: tuple[str, ...] = ()
    stop_reason: Literal["GOAL_ANSWERED", "EVIDENCE_EXHAUSTED", "BUDGET_EXHAUSTED", "PARTIAL_ANSWER", "SAFETY_STOP", "NEEDS_AUTHORIZATION"]
    execution_refs: tuple[str, ...] = ()
    verification_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_record(self):
        draft = DiagnosisAnswerDraft(
            claims=self.claims, recommendations=self.recommendations,
            unresolved_questions=self.unresolved_questions,
            goal_satisfied=any(goal.status == "SATISFIED" for goal in self.goal_assessments),
        )
        refs = {item.evidence_ref for item in self.observations}
        if len(refs) != len(self.observations):
            raise ValueError("调查观测引用不能重复")
        for claim in draft.claims:
            if any(item.evidence_ref not in refs for item in (*claim.evidence_fields, *claim.counterevidence_fields)):
                raise ValueError("主张必须引用本记录冻结的观测")
        claim_ids = {item.claim_id for item in self.claims}
        if any(not set(goal.claim_refs) <= claim_ids for goal in self.goal_assessments):
            raise ValueError("目标完成状态引用了不存在的主张")
        if (self.revision == 1) != (self.previous_revision is None):
            raise ValueError("调查修订必须关联前一不可变版本")
        return self

    def validate_observations(self, evidence):
        expected = {item.evidence_ref: item for item in self.observations}
        actual = {item.evidence_ref: item for item in evidence}
        if len(actual) != len(evidence) or set(expected) != set(actual):
            raise ValueError("调查记录与结果证据集合不一致")
        for ref, observation in expected.items():
            fact = actual[ref]
            for key in InvestigationObservation.model_fields:
                if str(getattr(observation, key)) != str(getattr(fact, key)):
                    raise ValueError("调查记录与原观测的身份、时间、口径或信任标记不一致")
