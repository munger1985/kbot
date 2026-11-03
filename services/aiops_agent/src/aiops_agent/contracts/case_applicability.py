"""案例检索适用性判断合同；不能替代当前现场验证或变更授权。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CaseConditionJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str = Field(min_length=1, max_length=128)
    quote: str = Field(min_length=1, max_length=4000)
    state: Literal["MET", "NOT_MET", "UNKNOWN"]
    fact_ids: tuple[str, ...] = Field(default=(), max_length=16)
    explanation: str = Field(min_length=1, max_length=800)


class CaseSemanticJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_version_id: str = Field(min_length=1, max_length=128)
    conditions: tuple[CaseConditionJudgment, ...] = Field(default=(), max_length=96)


class CaseApplicabilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["OPS_CASE_APPLICABILITY_ASSESSMENT.v1"] = "OPS_CASE_APPLICABILITY_ASSESSMENT.v1"
    context_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[CaseSemanticJudgment, ...] = Field(max_length=20)
