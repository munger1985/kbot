"""普通诊断与 SQL 专项共享的修订和逐目标状态。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClaimRevision(BaseModel):
    """显式引用前一版本的判断，不以措辞相似自动合并。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prior_claim_id: str = Field(pattern=r"^c[0-9]+$")
    prior_artifact_id: str | None = None
    current_claim_id: str = Field(pattern=r"^c[0-9]+$")
    relation: Literal["SUPPORTS", "REFUTES", "SUPERSEDES"]
    reason: str = Field(min_length=1, max_length=2000)


class InvestigationGoal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_id: str
    status: Literal["SATISFIED", "PARTIAL", "BLOCKED", "NOT_APPLICABLE"]
    claim_refs: tuple[str, ...] = ()
    blocking_gaps: tuple[str, ...] = ()
    explanation: str = Field(default="", max_length=2000)
