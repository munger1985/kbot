"""问题 SQL 的逐条 Agent 初步研判，身份和引用由服务端核验。"""

from pydantic import BaseModel, ConfigDict, Field


class InspectionSqlCause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    possible_cause: str = Field(min_length=1)
    supporting_metrics: tuple[str, ...] = Field(min_length=1)
    verification: str = Field(min_length=1)


class InspectionSqlAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    validation: str = Field(min_length=1)
    requires_approval: bool


class InspectionSqlReviewDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    problem_id: str
    analysis: str = Field(min_length=1)
    possible_causes: tuple[InspectionSqlCause, ...] = Field(min_length=1)
    recommendations: tuple[InspectionSqlAdvice, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
