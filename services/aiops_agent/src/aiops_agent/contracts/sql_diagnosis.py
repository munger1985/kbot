"""SQL 诊断的逐项判断、证据字段及条件化建议合同。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.contracts.diagnosis_state import ClaimRevision, InvestigationGoal


class EvidenceFieldReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str = Field(min_length=1, max_length=512)
    row_index: int = Field(ge=0)
    field: str = Field(min_length=1, max_length=128)


class SqlConclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conclusion_id: str = Field(pattern=r"^c[0-9]+$")
    statement: str = Field(min_length=1, max_length=3000)
    level: Literal["OBSERVED", "SUPPORTED", "HYPOTHESIS", "UNKNOWN"]
    scope: Literal["SUBJECT", "CONTEXT"]
    applicable_time: str = Field(min_length=1, max_length=1000)
    evidence_fields: tuple[EvidenceFieldReference, ...] = Field(default=(), max_length=24)
    reasoning: str = Field(min_length=1, max_length=3000)
    alternatives: tuple[str, ...] = Field(default=(), max_length=8)
    limitations: tuple[str, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_support(self):
        if self.level in {"OBSERVED", "SUPPORTED"} and not self.evidence_fields:
            raise ValueError("事实或有依据的解释必须引用具体观察字段")
        if self.level == "SUPPORTED" and not self.limitations:
            raise ValueError("原因解释必须说明适用边界")
        return self


class RecommendationParameter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    value: str | None = Field(default=None, max_length=2000)
    source: Literal["OBSERVATION", "USER_INPUT", "DOCUMENT", "UNRESOLVED"]
    evidence_field: EvidenceFieldReference | None = None

    @model_validator(mode="after")
    def validate_origin(self):
        if self.source == "OBSERVATION" and (self.evidence_field is None or self.value is None):
            raise ValueError("观测参数必须有原始字段引用及值")
        if self.source == "UNRESOLVED" and self.value is not None:
            raise ValueError("未知参数不能填入猜测值")
        return self


class SqlRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recommendation_id: str = Field(pattern=r"^r[0-9]+$")
    conclusion_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    action: str = Field(min_length=1, max_length=3000)
    intent: Literal["MITIGATE", "REPAIR", "VERIFY", "OBSERVE"]
    preconditions: tuple[str, ...] = Field(min_length=1, max_length=12)
    risks: tuple[str, ...] = Field(min_length=1, max_length=12)
    validation: tuple[str, ...] = Field(min_length=1, max_length=12)
    rollback_or_stop: str = Field(min_length=1, max_length=2000)
    requires_approval: bool
    draft_sql: str | None = Field(default=None, max_length=12000)
    parameters: tuple[RecommendationParameter, ...] = Field(default=(), max_length=24)
    missing_parameters: tuple[str, ...] = Field(default=(), max_length=24)

    @model_validator(mode="after")
    def validate_draft(self):
        if self.draft_sql and not self.requires_approval:
            raise ValueError("SQL 草案须明确待验证与审批，不能自动执行")
        return self


class SqlDiagnosisDraft(BaseModel):
    """模型可以提出新原因，但不得越过证据与执行边界。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["SQL_DIAGNOSTIC_ASSESSMENT.v1"] = "SQL_DIAGNOSTIC_ASSESSMENT.v1"
    summary: str = Field(min_length=1, max_length=2000)
    conclusions: tuple[SqlConclusion, ...] = Field(min_length=1, max_length=12)
    recommendations: tuple[SqlRecommendation, ...] = Field(min_length=1, max_length=12)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=12)
    goal_satisfied: bool
    goal_assessments: tuple[InvestigationGoal, ...] = Field(default=(), max_length=16)
    continues_investigation: bool = False
    claim_revisions: tuple[ClaimRevision, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_links(self):
        known = {item.conclusion_id for item in self.conclusions}
        if len(known) != len(self.conclusions):
            raise ValueError("诊断结论标识不能重复")
        if len({item.recommendation_id for item in self.recommendations}) != len(self.recommendations):
            raise ValueError("建议标识不能重复")
        for item in self.recommendations:
            if not set(item.conclusion_ids) <= known:
                raise ValueError("建议必须关联本轮诊断结论")
            if item.intent in {"MITIGATE", "REPAIR"} and not item.requires_approval:
                raise ValueError("缓解和修复措施必须另行确认与审批")
        return self
