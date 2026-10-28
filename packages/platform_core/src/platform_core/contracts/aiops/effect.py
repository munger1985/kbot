"""执行前冻结的业务效果规则；不接受客户端填写观测值或验证等级。"""

from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from .types import AIOpsContract, ArtifactRef, UUIDv7, UtcDatetime


class EffectSignalRule(AIOpsContract):
    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
    role: Literal["PRIMARY", "GUARDRAIL", "LOAD", "SYMPTOM"]
    metric: str = Field(min_length=1, max_length=256)
    statement_identity: dict[str, str | bool | None] = Field(default_factory=dict, max_length=6)
    comparison: Literal["DECREASE_BY", "INCREASE_BY", "STABLE_WITHIN", "AT_MOST", "AT_LEAST"]
    threshold: str = Field(pattern=r"^-?\d{1,30}(?:\.\d{1,18})?$", max_length=50)

    @model_validator(mode="after")
    def validate_rule(self):
        value = Decimal(self.threshold)
        if self.comparison in {"DECREASE_BY", "INCREASE_BY", "STABLE_WITHIN"} and not 0 <= value <= 1:
            raise ValueError("相对阈值必须介于 0 与 1 之间")
        if self.role == "LOAD" and self.comparison != "STABLE_WITHIN":
            raise ValueError("业务负载条件必须使用稳定范围比较")
        if self.role == "PRIMARY" and self.comparison not in {"DECREASE_BY", "INCREASE_BY"}:
            raise ValueError("主要效果必须声明改善方向和幅度")
        if self.role == "PRIMARY" and value <= 0:
            raise ValueError("主要效果改善幅度必须大于零")
        return self


class EffectPlanCommand(AIOpsContract):
    expected_row_version: int = Field(ge=1)
    baseline_artifact_id: UUIDv7
    signals: tuple[EffectSignalRule, ...] = Field(min_length=3, max_length=32)
    settle_delay_seconds: int = Field(default=60, ge=0, le=86400)
    after_window_seconds: int = Field(default=900, ge=60, le=604800)
    max_baseline_age_seconds: int = Field(default=3600, ge=60, le=604800)
    auto_evaluate: bool = True
    stop_conditions: tuple[str, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_signals(self):
        if len({item.key for item in self.signals}) != len(self.signals):
            raise ValueError("效果规则名称不得重复")
        if not {"PRIMARY", "GUARDRAIL", "LOAD"} <= {item.role for item in self.signals}:
            raise ValueError("必须冻结主要效果、护栏和负载可比条件")
        if any(not item.strip() or len(item) > 1000 for item in self.stop_conditions):
            raise ValueError("停止条件不能为空或超长")
        return self


class FrozenEffectPlan(EffectPlanCommand):
    """基线与时间由服务端冻结，不能通过任意字典绕过效果条件校验。"""

    baseline_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    frozen_at: UtcDatetime
    collection_interval_seconds: int | None = Field(default=None, ge=60, le=3600)

    @model_validator(mode="after")
    def validate_collection(self):
        if self.auto_evaluate and self.collection_interval_seconds is None:
            raise ValueError("自动效果验证必须冻结已启用的采集周期")
        return self


class EffectEvaluationCommand(AIOpsContract):
    expected_row_version: int = Field(ge=1)
    after_artifact_id: UUIDv7
    verification_artifact_id: UUIDv7


class EffectVerificationView(AIOpsContract):
    proposal_id: UUIDv7
    plan_artifact: ArtifactRef | None = None
    frozen_plan: dict = Field(default_factory=dict)
    comparison_artifact: ArtifactRef | None = None
    result: dict = Field(default_factory=dict)
    report_id: UUIDv7 | None = None
    evidence_candidates: tuple[dict, ...] = ()
    followup: dict = Field(default_factory=dict)
