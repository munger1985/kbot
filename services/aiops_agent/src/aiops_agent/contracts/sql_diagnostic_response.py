"""交互推理只生成判断；时间、编号和报告结构由已核验来源重建。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.contracts.diagnosis_state import ClaimRevision, InvestigationGoal
from aiops_agent.contracts.sql_diagnosis import EvidenceFieldReference, SqlReviewRecommendation, OracleSqlEvidenceRequest


class SqlDiagnosticFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conclusion_id: str = Field(pattern=r"^c[0-9]+$")
    statement: str = Field(min_length=1, max_length=400)
    level: Literal["OBSERVED", "SUPPORTED", "HYPOTHESIS", "UNKNOWN"]
    scope: Literal["SUBJECT", "CONTEXT", "COMPARISON"] = "SUBJECT"
    subject_ids: tuple[str, ...] = Field(default=(), max_length=20)
    evidence_fields: tuple[EvidenceFieldReference, ...] = Field(default=(), max_length=8)
    reasoning: str = Field(min_length=1, max_length=400)
    limitations: tuple[Annotated[str, Field(max_length=200)], ...] = Field(default=(), max_length=2)


class SqlPlanReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str
    node_ids: tuple[Annotated[int, Field(ge=0, le=127)], ...] = Field(min_length=1, max_length=4)
    conclusion_id: str = Field(pattern=r"^c[0-9]+$")
    mechanism: str = Field(min_length=1, max_length=300)
    impact: str = Field(min_length=1, max_length=200)


class SqlDiagnosticResponse(BaseModel):
    """模型不再复制原始统计、全部节点、时间血缘或整套报告模板。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Literal["PROBLEM_FOUND", "NO_OBVIOUS_ISSUE", "INSUFFICIENT_EVIDENCE"]
    summary: str = Field(min_length=1, max_length=400)
    findings: tuple[SqlDiagnosticFinding, ...] = Field(min_length=1, max_length=12)
    plan_reviews: tuple[SqlPlanReview, ...] = Field(default=(), max_length=8)
    recommendations: tuple[SqlReviewRecommendation, ...] = Field(min_length=1, max_length=6)
    goal_satisfied: bool
    goal_assessments: tuple[InvestigationGoal, ...] = Field(default=(), max_length=16)
    next_evidence: tuple[Annotated[str, Field(min_length=1, max_length=500)], ...] = Field(default=(), max_length=2)
    oracle_evidence_requests: tuple[OracleSqlEvidenceRequest, ...] = Field(default=(), max_length=2)
    continues_investigation: bool = False
    claim_revisions: tuple[ClaimRevision, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_plan_findings(self):
        ids = {finding.conclusion_id for finding in self.findings}
        reviewed = [item.conclusion_id for item in self.plan_reviews]
        if len(reviewed) != len(set(reviewed)):
            raise ValueError("每条 finding 只关联一项关键计划解释；相关节点合并在同一 plan_review，不逐节点复制结论")
        if not set(reviewed) <= ids:
            raise ValueError("计划解释必须关联当前 findings 的 conclusion_id")
        return self


class SingleSqlDiagnosticResponse(SqlDiagnosticResponse):
    """单主体交互交付关键机制，不生成比较调查或完整报告的篇幅。"""

    findings: tuple[SqlDiagnosticFinding, ...] = Field(min_length=1, max_length=4)
    plan_reviews: tuple[SqlPlanReview, ...] = Field(default=(), max_length=4)
    recommendations: tuple[SqlReviewRecommendation, ...] = Field(min_length=1, max_length=3)


class SingleSqlPlanDiagnosticResponse(SingleSqlDiagnosticResponse):
    plan_reviews: tuple[SqlPlanReview, ...] = Field(min_length=1, max_length=4)


class SqlPlanDiagnosticResponse(SqlDiagnosticResponse):
    plan_reviews: tuple[SqlPlanReview, ...] = Field(min_length=1, max_length=8)
