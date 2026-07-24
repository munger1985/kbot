"""处理前后对比的不可变计划与确定性结果契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from platform_core.contracts.aiops.types import UtcDatetime


class _ComparisonContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonPlan(_ComparisonContract):
    schema_version: Literal["COMPARISON_PLAN.v1"] = "COMPARISON_PLAN.v1"
    proposal_id: str
    source_run_id: str
    solution_group_key: str
    action_template_id: str
    action_template_version: str
    baseline_start: UtcDatetime
    baseline_end: UtcDatetime
    settle_delay_seconds: int = Field(ge=0)
    after_window_seconds: int = Field(ge=1)
    primary_signals: tuple[str, ...]
    guardrail_signals: tuple[str, ...] = ()
    required_tool_refs: tuple[str, ...]
    baseline_evidence_refs: tuple[str, ...]
    result_rule_version: Literal["action-effect.v1"] = "action-effect.v1"


class ComparisonResult(_ComparisonContract):
    schema_version: Literal["COMPARISON_RESULT.v1"] = (
        "COMPARISON_RESULT.v1"
    )
    comparison_plan_artifact_id: str
    verification_artifact_id: str
    proposal_id: str
    source_run_id: str
    source_result_artifact_id: str
    baseline_start: UtcDatetime
    baseline_end: UtcDatetime
    after_start: UtcDatetime
    after_end: UtcDatetime
    primary_signals: dict[str, bool | None]
    guardrail_signals: dict[str, bool | None] = Field(default_factory=dict)
    gap_codes: tuple[str, ...] = ()
    evidence_hashes: tuple[str, ...] = ()
    result: Literal[
        "IMPROVED", "UNCHANGED", "DEGRADED", "INCONCLUSIVE"
    ]
    rationale_codes: tuple[str, ...]
    causal_limitations: tuple[str, ...] = (
        "时间相关性不能单独证明因果关系",
    )
