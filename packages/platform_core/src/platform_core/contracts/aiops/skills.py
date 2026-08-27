"""AIOps DBA 意图和 Skill 计划的版本化契约。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .conversation import DbaIntent, MeasurementSemantics
from .types import AIOpsContract, DatabaseType, JsonObject, Sha256Digest


INTENT_PLAN_SCHEMA_VERSION = "DBA_INTENT_PLAN.v1"
SKILL_PLAN_SCHEMA_VERSION = "DBA_SKILL_PLAN.v1"
SKILL_MANIFEST_SCHEMA_VERSION = "DBA_SKILL_MANIFEST.v1"
CAPABILITY_SNAPSHOT_SCHEMA_VERSION = "DBA_CAPABILITY_SNAPSHOT.v1"


class DbaDomain(StrEnum):
    SQL_PERFORMANCE = "SQL_PERFORMANCE"
    SESSION_AND_LOCK = "SESSION_AND_LOCK"
    INSTANCE_PERFORMANCE = "INSTANCE_PERFORMANCE"
    STORAGE_AND_CAPACITY = "STORAGE_AND_CAPACITY"
    BACKUP_AND_RECOVERY = "BACKUP_AND_RECOVERY"
    HIGH_AVAILABILITY = "HIGH_AVAILABILITY"
    REPLICATION = "REPLICATION"
    CONFIGURATION = "CONFIGURATION"
    SECURITY_AND_PRIVILEGE = "SECURITY_AND_PRIVILEGE"
    CONNECTION_AND_NETWORK = "CONNECTION_AND_NETWORK"
    MAINTENANCE = "MAINTENANCE"
    PATCH_AND_UPGRADE = "PATCH_AND_UPGRADE"
    DATA_INTEGRITY = "DATA_INTEGRITY"
    ALERT_AND_LOG = "ALERT_AND_LOG"
    HOST_AND_OS = "HOST_AND_OS"


class PresentationPreference(StrEnum):
    MARKDOWN = "MARKDOWN"
    TABLE = "TABLE"
    CHART = "CHART"
    TABLE_AND_CHART = "TABLE_AND_CHART"


class TimeWindow(AIOpsContract):
    mode: Literal["CURRENT", "RECENT", "ABSOLUTE", "SINCE_STARTUP"]
    duration_seconds: int | None = Field(default=None, ge=1, le=31_536_000)
    start_at: str | None = None
    end_at: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "TimeWindow":
        if self.mode == "RECENT" and self.duration_seconds is None:
            raise ValueError("RECENT 时间窗口必须包含 duration_seconds")
        if self.mode == "ABSOLUTE" and not (self.start_at and self.end_at):
            raise ValueError("ABSOLUTE 时间窗口必须包含起止时间")
        return self


class IntentCandidate(AIOpsContract):
    intent: DbaIntent
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1000)


class DbaIntentPlan(AIOpsContract):
    schema_version: str = INTENT_PLAN_SCHEMA_VERSION
    primary_intent: DbaIntent
    candidates: tuple[IntentCandidate, ...]
    primary_domain: DbaDomain
    subject: str | None = Field(default=None, max_length=64)
    time_window: TimeWindow | None = None
    requested_limit: int | None = Field(default=None, ge=1, le=1000)
    requested_order: tuple[str, ...] = ()
    presentation_preference: PresentationPreference | None = None
    clarification_question: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_candidates(self) -> "DbaIntentPlan":
        if not self.candidates:
            raise ValueError("意图计划至少包含一个候选意图")
        if self.primary_intent not in {item.intent for item in self.candidates}:
            raise ValueError("主意图必须出现在候选意图中")
        if len({item.intent for item in self.candidates}) != len(self.candidates):
            raise ValueError("候选意图不能重复")
        return self


class SkillVersionRange(AIOpsContract):
    minimum: str | None = Field(default=None, pattern=r"^[0-9]+$")
    maximum: str | None = Field(default=None, pattern=r"^[0-9]+$")

    @model_validator(mode="after")
    def validate_range(self) -> "SkillVersionRange":
        if (
            self.minimum is not None
            and self.maximum is not None
            and int(self.minimum) > int(self.maximum)
        ):
            raise ValueError("Skill 数据库版本范围无效")
        return self


class SkillLimits(AIOpsContract):
    max_rows: int = Field(ge=1, le=5000)
    timeout_seconds: int = Field(ge=1, le=600)
    max_attempts: int = Field(default=2, ge=1, le=5)
    cost_units: int = Field(default=1, ge=1, le=1000)


class SkillToolStep(AIOpsContract):
    step_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
    tool_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
    tool_version: str = Field(
        default="1.0.0", pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$"
    )
    depends_on: tuple[str, ...] = ()
    input: JsonObject = Field(default_factory=dict)


class SkillFreshnessPolicy(AIOpsContract):
    max_age_seconds: int | None = Field(default=None, ge=1, le=31_536_000)
    requires_observed_at: bool = True


class DbaSkillManifest(AIOpsContract):
    schema_version: str = SKILL_MANIFEST_SCHEMA_VERSION
    skill_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    database_types: tuple[DatabaseType, ...]
    version_range: SkillVersionRange = Field(default_factory=SkillVersionRange)
    supported_intents: tuple[DbaIntent, ...]
    domains: tuple[DbaDomain, ...]
    subjects: tuple[str, ...] = ()
    required_source_capabilities: tuple[str, ...] = ()
    optional_source_capabilities: tuple[str, ...] = ()
    required_target_capabilities: tuple[str, ...] = ()
    required_privileges: tuple[str, ...] = ()
    required_entitlements: tuple[str, ...] = ()
    input_schema: str = Field(min_length=1, max_length=128)
    defaults: JsonObject = Field(default_factory=dict)
    limits: SkillLimits
    tool_dag: tuple[SkillToolStep, ...]
    output_schema: str = Field(min_length=1, max_length=128)
    measurement_semantics: MeasurementSemantics
    presentation_kind: PresentationPreference
    fallback_skills: tuple[str, ...] = ()
    manual_evidence_template: str | None = Field(default=None, max_length=128)
    sensitive_fields: tuple[str, ...] = ()
    redaction_rules: tuple[str, ...] = ()
    freshness: SkillFreshnessPolicy = Field(default_factory=SkillFreshnessPolicy)

    @model_validator(mode="after")
    def validate_manifest(self) -> "DbaSkillManifest":
        if not self.database_types or not self.supported_intents or not self.domains:
            raise ValueError("Skill 必须声明数据库类型、意图和专业领域")
        if len(set(self.database_types)) != len(self.database_types):
            raise ValueError("Skill 数据库类型不能重复")
        step_ids: set[str] = set()
        for step in self.tool_dag:
            if step.step_id in step_ids:
                raise ValueError("Skill Tool Step ID 不能重复")
            if any(value not in step_ids for value in step.depends_on):
                raise ValueError("Skill Tool 只能依赖已经声明的前置步骤")
            step_ids.add(step.step_id)
        return self


class SourceCapabilitySnapshot(AIOpsContract):
    source_id: str = Field(min_length=1, max_length=64)
    source_type: str = Field(min_length=1, max_length=64)
    enabled: bool
    reachable: bool
    capabilities: tuple[str, ...] = ()


class DbaCapabilitySnapshot(AIOpsContract):
    schema_version: str = CAPABILITY_SNAPSHOT_SCHEMA_VERSION
    agent_id: str = Field(min_length=1, max_length=64)
    agent_version_id: str = Field(min_length=1, max_length=64)
    target_id: str = Field(min_length=1, max_length=64)
    database_type: DatabaseType
    database_version: str | None = Field(default=None, max_length=64)
    target_enabled: bool
    target_reachable: bool
    target_capabilities: tuple[str, ...] = ()
    privileges: tuple[str, ...] = ()
    entitlements: tuple[str, ...] = ()
    source_snapshots: tuple[SourceCapabilitySnapshot, ...] = ()

    @property
    def available_source_capabilities(self) -> frozenset[str]:
        return frozenset(
            capability
            for source in self.source_snapshots
            if source.enabled and source.reachable
            for capability in source.capabilities
        )


class SkillPlanItem(AIOpsContract):
    ordinal: int = Field(ge=1)
    skill_id: str = Field(min_length=1, max_length=128)
    skill_version: str = Field(min_length=1, max_length=64)
    manifest_hash: Sha256Digest
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
    catalog_hash: Sha256Digest
    items: tuple[SkillPlanItem, ...]

    @model_validator(mode="after")
    def validate_ordinals(self) -> "DbaSkillPlan":
        ordinals = tuple(item.ordinal for item in self.items)
        if ordinals != tuple(range(1, len(ordinals) + 1)):
            raise ValueError("Skill 顺序号必须从1开始连续递增")
        return self
