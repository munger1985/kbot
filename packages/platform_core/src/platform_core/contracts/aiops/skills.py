"""AIOps DBA 意图和 Skill 计划的版本化契约。"""

from __future__ import annotations

from pydantic import Field, model_validator

from .conversation import DbaIntent, MeasurementSemantics
from .types import AIOpsContract, JsonObject


INTENT_PLAN_SCHEMA_VERSION = "DBA_INTENT_PLAN.v1"
SKILL_PLAN_SCHEMA_VERSION = "DBA_SKILL_PLAN.v1"


class IntentCandidate(AIOpsContract):
    intent: DbaIntent
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1000)


class DbaIntentPlan(AIOpsContract):
    schema_version: str = INTENT_PLAN_SCHEMA_VERSION
    primary_intent: DbaIntent
    candidates: tuple[IntentCandidate, ...]
    primary_domain: str = Field(min_length=1, max_length=48)
    subject: str | None = Field(default=None, max_length=64)
    time_window: JsonObject | None = None
    requested_limit: int | None = Field(default=None, ge=1, le=1000)
    requested_order: tuple[str, ...] = ()
    presentation_preference: str | None = Field(default=None, max_length=32)
    clarification_question: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_candidates(self) -> "DbaIntentPlan":
        if not self.candidates:
            raise ValueError("意图计划至少包含一个候选意图")
        if self.primary_intent not in {item.intent for item in self.candidates}:
            raise ValueError("主意图必须出现在候选意图中")
        return self


class SkillPlanItem(AIOpsContract):
    ordinal: int = Field(ge=1)
    skill_id: str = Field(min_length=1, max_length=128)
    skill_version: str = Field(min_length=1, max_length=64)
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(min_length=1, max_length=1000)
    evidence_question: str = Field(min_length=1, max_length=1000)
    measurement_semantics: MeasurementSemantics
    input: JsonObject = Field(default_factory=dict)
    depends_on: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_dependencies(self) -> "SkillPlanItem":
        if any(value >= self.ordinal for value in self.depends_on):
            raise ValueError("Skill 只能依赖顺序号更小的前置 Skill")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("Skill 前置依赖不能重复")
        return self


class DbaSkillPlan(AIOpsContract):
    schema_version: str = SKILL_PLAN_SCHEMA_VERSION
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: tuple[SkillPlanItem, ...]

    @model_validator(mode="after")
    def validate_ordinals(self) -> "DbaSkillPlan":
        ordinals = tuple(item.ordinal for item in self.items)
        if ordinals != tuple(range(1, len(ordinals) + 1)):
            raise ValueError("Skill 顺序号必须从1开始连续递增")
        return self
