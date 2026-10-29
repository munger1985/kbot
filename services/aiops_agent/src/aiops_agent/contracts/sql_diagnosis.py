"""SQL 诊断的逐项判断、证据字段及条件化建议合同。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

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
    scope: Literal["SUBJECT", "CONTEXT", "COMPARISON"]
    subject_ids: tuple[str, ...] = Field(default=(), max_length=20)
    applicable_time: str = Field(min_length=1, max_length=1000)
    evidence_fields: tuple[EvidenceFieldReference, ...] = Field(default=(), max_length=24)
    reasoning: str = Field(min_length=1, max_length=3000)
    alternatives: tuple[str, ...] = Field(default=(), max_length=8)
    limitations: tuple[str, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_support(self):
        if len(self.subject_ids) != len(set(self.subject_ids)):
            raise ValueError("SQL 诊断统计主体不能重复")
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
    preconditions: tuple[str, ...] = Field(default=(), max_length=12)
    risks: tuple[str, ...] = Field(default=(), max_length=12)
    validation: tuple[str, ...] = Field(min_length=1, max_length=12)
    rollback_or_stop: str = Field(default="", max_length=2000)
    requires_approval: bool
    draft_sql: str | None = Field(default=None, max_length=12000)
    parameters: tuple[RecommendationParameter, ...] = Field(default=(), max_length=24)
    missing_parameters: tuple[str, ...] = Field(default=(), max_length=24)

    @model_validator(mode="after")
    def validate_draft(self):
        if self.draft_sql and not self.requires_approval:
            raise ValueError("SQL 草案须明确待验证与审批，不能自动执行")
        if (self.intent in {"MITIGATE", "REPAIR"} or self.draft_sql) and not (
            self.preconditions and self.risks and self.rollback_or_stop
        ):
            raise ValueError("变更建议必须保留适用条件、风险与回退方案")
        return self


class OracleSqlEvidenceRequest(BaseModel):
    """仅声明诊断所需证据，不由模型提供游标、执行标识或 SQL。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["XPLAN", "SQL_MONITOR", "ASH"]
    subject_id: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=400)
    begin_time: AwareDatetime | None = None
    end_time: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_window(self):
        if (self.begin_time is None) != (self.end_time is None):
            raise ValueError("ASH 显式时间必须成对提供")
        if self.begin_time is not None and (
            self.kind != "ASH" or not 0 < (self.end_time - self.begin_time).total_seconds() <= 86400
        ):
            raise ValueError("只有 ASH 可以提供带时区且不超过24小时的有序窗口")
        return self


class SqlDiagnosisDraft(BaseModel):
    """模型可以提出新原因，但不得越过证据与执行边界。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["SQL_DIAGNOSTIC_ASSESSMENT.v1"] = "SQL_DIAGNOSTIC_ASSESSMENT.v1"
    summary: str = Field(min_length=1, max_length=2000)
    verdict: Literal["PROBLEM_FOUND", "NO_OBVIOUS_ISSUE", "INSUFFICIENT_EVIDENCE"] | None = None
    conclusions: tuple[SqlConclusion, ...] = Field(min_length=1, max_length=12)
    recommendations: tuple[SqlRecommendation, ...] = Field(min_length=1, max_length=12)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=12)
    goal_satisfied: bool
    goal_assessments: tuple[InvestigationGoal, ...] = Field(default=(), max_length=16)
    continues_investigation: bool = False
    claim_revisions: tuple[ClaimRevision, ...] = Field(default=(), max_length=16)
    next_evidence: tuple[Annotated[str, Field(min_length=1, max_length=2000)], ...] = Field(default=(), max_length=3)
    oracle_evidence_requests: tuple[OracleSqlEvidenceRequest, ...] = Field(default=(), max_length=2)

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
        if self.verdict == "NO_OBVIOUS_ISSUE" and (
            not any(item.scope == "SUBJECT" and item.level in {"OBSERVED", "SUPPORTED"} for item in self.conclusions)
            or any(item.intent in {"MITIGATE", "REPAIR"} for item in self.recommendations)
        ):
            raise ValueError("无明显问题必须有主体直接判断，不能同时要求修复未经证实的问题")
        if self.goal_satisfied and (self.next_evidence or self.oracle_evidence_requests):
            raise ValueError("已完成目标不能同时要求阻断交付的补证")
        if any(not item.strip() for item in self.next_evidence):
            raise ValueError("关键补证必须说明需要验证的内容")
        return self


class SqlReviewConclusion(SqlConclusion):
    """交互检查点保留机制与引用，不复制报告级长篇说明。"""

    statement: str = Field(min_length=1, max_length=1000)
    reasoning: str = Field(min_length=1, max_length=1000)
    alternatives: tuple[str, ...] = Field(default=(), max_length=3)
    limitations: tuple[str, ...] = Field(default=(), max_length=3)


class SqlReviewRecommendation(SqlRecommendation):
    action: str = Field(min_length=1, max_length=1000)
    preconditions: tuple[str, ...] = Field(default=(), max_length=3)
    risks: tuple[str, ...] = Field(default=(), max_length=3)
    validation: tuple[str, ...] = Field(min_length=1, max_length=3)


class SqlDiagnosisReview(SqlDiagnosisDraft):
    """交互诊断明确交付判断，不强制生成完整实施报告。"""

    verdict: Literal["PROBLEM_FOUND", "NO_OBVIOUS_ISSUE", "INSUFFICIENT_EVIDENCE"]
    summary: str = Field(min_length=1, max_length=1000)
    conclusions: tuple[SqlReviewConclusion, ...] = Field(min_length=1, max_length=12)
    recommendations: tuple[SqlReviewRecommendation, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_problem_verdict(self):
        if self.verdict == "PROBLEM_FOUND" and not any(
            item.scope in {"SUBJECT", "COMPARISON"} and item.level in {"OBSERVED", "SUPPORTED"}
            for item in self.conclusions
        ):
            raise ValueError("只有背景或待验证假设时不能声明已发现目标 SQL 问题")
        return self


class SqlDiagnosticCheckpoint(BaseModel):
    """阶段判断绑定原证据与任务范围，供回答和预算收敛复用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    draft: SqlDiagnosisDraft
    scope_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_receipt: dict[str, Any] | None = None
