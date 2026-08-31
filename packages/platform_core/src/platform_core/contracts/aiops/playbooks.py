"""AIOps DBA Playbook 与能力快照的版本化契约。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .conversation import DbaIntent, MeasurementSemantics
from .types import AIOpsContract, DatabaseType, JsonObject, Sha256Digest


PLAYBOOK_PLAN_SCHEMA_VERSION = "DBA_PLAYBOOK_PLAN.v1"
PLAYBOOK_MANIFEST_SCHEMA_VERSION = "DBA_PLAYBOOK_MANIFEST.v1"
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


class PlaybookVersionRange(AIOpsContract):
    minimum: str | None = Field(default=None, pattern=r"^[0-9]+$")
    maximum: str | None = Field(default=None, pattern=r"^[0-9]+$")

    @model_validator(mode="after")
    def validate_range(self) -> "PlaybookVersionRange":
        if (
            self.minimum is not None
            and self.maximum is not None
            and int(self.minimum) > int(self.maximum)
        ):
            raise ValueError("Playbook 数据库版本范围无效")
        return self


class PlaybookLimits(AIOpsContract):
    max_rows: int = Field(ge=1, le=5000)
    timeout_seconds: int = Field(ge=1, le=600)
    max_attempts: int = Field(default=2, ge=1, le=5)
    cost_units: int = Field(default=1, ge=1, le=1000)


class PlaybookToolStep(AIOpsContract):
    step_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
    tool_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
    tool_version: str = Field(
        default="1.0.0", pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$"
    )
    depends_on: tuple[str, ...] = ()
    input: JsonObject = Field(default_factory=dict)


class PlaybookFreshnessPolicy(AIOpsContract):
    max_age_seconds: int | None = Field(default=None, ge=1, le=31_536_000)
    requires_observed_at: bool = True


class DbaPlaybookManifest(AIOpsContract):
    schema_version: str = PLAYBOOK_MANIFEST_SCHEMA_VERSION
    playbook_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    database_types: tuple[DatabaseType, ...]
    version_range: PlaybookVersionRange = Field(
        default_factory=PlaybookVersionRange
    )
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
    limits: PlaybookLimits
    tool_dag: tuple[PlaybookToolStep, ...]
    output_schema: str = Field(min_length=1, max_length=128)
    measurement_semantics: MeasurementSemantics
    presentation_kind: PresentationPreference
    fallback_playbooks: tuple[str, ...] = ()
    manual_evidence_template: str | None = Field(default=None, max_length=128)
    sensitive_fields: tuple[str, ...] = ()
    redaction_rules: tuple[str, ...] = ()
    freshness: PlaybookFreshnessPolicy = Field(
        default_factory=PlaybookFreshnessPolicy
    )

    @model_validator(mode="after")
    def validate_manifest(self) -> "DbaPlaybookManifest":
        if not self.database_types or not self.supported_intents or not self.domains:
            raise ValueError("Playbook 必须声明数据库类型、意图和专业领域")
        if len(set(self.database_types)) != len(self.database_types):
            raise ValueError("Playbook 数据库类型不能重复")
        step_ids: set[str] = set()
        for step in self.tool_dag:
            if step.step_id in step_ids:
                raise ValueError("Playbook Tool Step ID 不能重复")
            if any(value not in step_ids for value in step.depends_on):
                raise ValueError("Playbook Tool 只能依赖已经声明的前置步骤")
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
        """返回已授权Source能力；瞬时健康不作为规划准入条件。"""
        return frozenset(
            capability
            for source in self.source_snapshots
            if source.enabled
            for capability in source.capabilities
        )


class PlaybookPlanItem(AIOpsContract):
    ordinal: int = Field(ge=1)
    playbook_id: str = Field(min_length=1, max_length=128)
    playbook_version: str = Field(min_length=1, max_length=64)
    manifest_hash: Sha256Digest
    reason: str = Field(min_length=1, max_length=1000)
    evidence_question: str = Field(min_length=1, max_length=1000)
    measurement_semantics: MeasurementSemantics
    input: JsonObject = Field(default_factory=dict)
    depends_on: tuple[int, ...] = ()
    action_id: str | None = Field(
        default=None, pattern=r"^a[0-9]+$"
    )
    selected_tool_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9_.-]{0,127}$"
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "PlaybookPlanItem":
        if any(value >= self.ordinal for value in self.depends_on):
            raise ValueError("Playbook 只能依赖顺序号更小的前置 Playbook")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("Playbook 前置依赖不能重复")
        return self


class DbaPlaybookPlan(AIOpsContract):
    schema_version: str = PLAYBOOK_PLAN_SCHEMA_VERSION
    catalog_hash: Sha256Digest
    items: tuple[PlaybookPlanItem, ...]

    @model_validator(mode="after")
    def validate_ordinals(self) -> "DbaPlaybookPlan":
        ordinals = tuple(item.ordinal for item in self.items)
        if ordinals != tuple(range(1, len(ordinals) + 1)):
            raise ValueError("Playbook 顺序号必须从1开始连续递增")
        return self
