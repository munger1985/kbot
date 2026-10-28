"""处理前后对比的不可变计划与确定性结果契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from platform_core.contracts.aiops.types import UtcDatetime
from platform_core.contracts.aiops.effect import FrozenEffectPlan


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
    result_rule_version: Literal["action-effect.v1", "business-effect.v1"] = "action-effect.v1"
    business_effect: FrozenEffectPlan | None = None

    @model_validator(mode="after")
    def validate_effect_rule(self):
        if (self.business_effect is not None) != (self.result_rule_version == "business-effect.v1"):
            raise ValueError("动作复核与业务效果规则版本不得混用")
        if self.business_effect is not None:
            frozen = self.business_effect
            if (
                frozen.settle_delay_seconds != self.settle_delay_seconds
                or frozen.after_window_seconds != self.after_window_seconds
                or self.primary_signals != tuple(item.key for item in frozen.signals if item.role == "PRIMARY")
                or self.guardrail_signals != tuple(item.key for item in frozen.signals if item.role == "GUARDRAIL")
                or self.baseline_evidence_refs != (f"artifact:{frozen.baseline_artifact_id}",)
            ):
                raise ValueError("业务效果的窗口、条件或基线与冻结计划不一致")
        return self


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
        "RESOLVED",
        "IMPROVED",
        "UNCHANGED",
        "DEGRADED",
        "INCONCLUSIVE",
    ]
    rationale_codes: tuple[str, ...]
    causal_limitations: tuple[str, ...] = (
        "时间相关性不能单独证明因果关系",
    )
    direct_effect: Literal["ACHIEVED", "NOT_ACHIEVED", "ADVERSE", "UNKNOWN"] = "UNKNOWN"
    measured_effect: Literal["IMPROVED", "UNCHANGED", "DEGRADED", "INCONCLUSIVE"] = "INCONCLUSIVE"
    symptom_recovered: bool | None = None
    guardrails_passed: bool | None = None
    attribution_strength: Literal["SUPPORTED", "LIMITED", "UNKNOWN"] = "UNKNOWN"
    measurements: dict = Field(default_factory=dict)
    load_signals: dict[str, bool | None] = Field(default_factory=dict)
    measurement_artifact_refs: tuple[dict[str, str], ...] = ()
    execution_source: Literal["SOURCE_VERIFIED", "USER_PROVIDED", "UNKNOWN"] = "UNKNOWN"
    control_analysis: dict = Field(default_factory=dict)
