"""AIOps 配置资源的公开与内部共享 Wire 契约。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, HttpUrl, model_validator

from .workload import ActivitySamplerPolicy, WorkloadPolicy
from .types import (
    AIOpsContract,
    CursorPage,
    DatabaseType,
    JsonObject,
    PUBLIC_SCHEMA_VERSION,
    UUIDv7,
    UtcDatetime,
)


TargetStatus = Literal["ENABLED", "DISABLED"]
ConnectivityStatus = Literal[
    "UNKNOWN", "CHECKING", "CONNECTED", "DEGRADED", "MISCONFIGURED", "UNREACHABLE"
]
OracleContainerScope = Literal["CDB_ROOT", "PDB", "NON_CDB"]
ObservedStatus = Literal["UNKNOWN", "UP", "DOWN", "DEGRADED"]
HealthStatus = Literal["UNKNOWN", "HEALTHY", "DEGRADED", "UNREACHABLE"]
BindingStatus = Literal["ACTIVE", "REVOKED"]
SourceStatus = Literal["ENABLED", "DISABLED"]
SourceBindingStatus = Literal["ACTIVE", "DISABLED"]
PolicyStatus = Literal["DRAFT", "ACTIVE", "RETIRED"]
InspectionPlanStatus = Literal["ACTIVE", "PAUSED", "DISABLED"]
NotificationStage = Literal[
    "SITUATION_DETECTED",
    "DIAGNOSIS_STARTED",
    "REPORT_READY",
    "SITUATION_RECOVERED",
]
SupportedDatabaseVersion = Literal["19c", "26ai", "8.4", "16"]
SUPPORTED_DATABASE_VERSIONS: dict[DatabaseType, tuple[SupportedDatabaseVersion, ...]] = {
    DatabaseType.ORACLE: ("19c", "26ai"),
    DatabaseType.MYSQL: ("8.4",),
    DatabaseType.POSTGRESQL: ("16",),
}
# 多个已验证版本可复用同一档案；仅当系统视图合同变化时新增SQL档案。
DATABASE_SQL_PROFILES: dict[
    tuple[DatabaseType, SupportedDatabaseVersion], str
] = {
    (DatabaseType.ORACLE, "19c"): "oracle_19c",
    (DatabaseType.ORACLE, "26ai"): "oracle_26ai",
    (DatabaseType.MYSQL, "8.4"): "mysql_8_4",
    (DatabaseType.POSTGRESQL, "16"): "postgresql_16",
}


def validate_supported_database_version(
    db_type: DatabaseType | str,
    version_code: str,
) -> None:
    """只允许已经过端到端验证的数据库版本组合。"""
    database_type = DatabaseType(db_type)
    if version_code not in SUPPORTED_DATABASE_VERSIONS[database_type]:
        supported = "、".join(SUPPORTED_DATABASE_VERSIONS[database_type])
        raise ValueError(f"{database_type.value}仅支持已验证版本：{supported}")
    if (database_type, version_code) not in DATABASE_SQL_PROFILES:
        raise ValueError("数据库版本缺少固定SQL档案")


class SecretRefStatus(AIOpsContract):
    """只暴露 Secret 引用是否配置及不可逆指纹。"""

    configured: bool
    provider: str | None = None
    fingerprint: str | None = None


class DatabaseCredentialInput(AIOpsContract):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=4096)


class DatabaseCredentialStatus(AIOpsContract):
    configured: bool
    credential_id: UUIDv7 | None = None
    key_version: str | None = None
    updated_at: UtcDatetime | None = None


class TargetEndpoint(AIOpsContract):
    host: str = Field(
        min_length=1,
        max_length=253,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$",
    )
    port: int = Field(ge=1, le=65535)
    service: str | None = Field(default=None, min_length=1, max_length=256)
    database: str | None = Field(default=None, min_length=1, max_length=256)
    tls_enabled: bool = True
    tls_profile_ref: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )

    @model_validator(mode="after")
    def validate_tls_profile(self) -> "TargetEndpoint":
        if self.tls_profile_ref is not None and not self.tls_enabled:
            raise ValueError("TLS Profile只能在启用TLS时配置")
        return self


class TargetCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str = Field(min_length=1, max_length=256)
    db_type: DatabaseType
    version_code: SupportedDatabaseVersion
    environment: Literal["PROD", "STG", "DEV"]
    db_role: Literal["PRIMARY", "STANDBY", "UNKNOWN"] = "UNKNOWN"
    oracle_container_scope: OracleContainerScope | None = None
    oracle_pdb_name: str | None = Field(default=None, min_length=1, max_length=128)
    endpoint: TargetEndpoint | None = None
    readonly_connection_enabled: bool = False
    controlled_change_enabled: bool = False
    diagnostic_credential: DatabaseCredentialInput | None = None
    execution_credential: DatabaseCredentialInput | None = None
    importance_level: int = Field(default=3, ge=1, le=5)
    capabilities: JsonObject = Field(default_factory=dict)
    workload_snapshot_policy: WorkloadPolicy = Field(
        default_factory=WorkloadPolicy
    )
    activity_sampler_policy: ActivitySamplerPolicy = Field(
        default_factory=ActivitySamplerPolicy
    )

    @model_validator(mode="after")
    def validate_database_endpoint(self) -> "TargetCreate":
        validate_supported_database_version(self.db_type, self.version_code)
        if self.readonly_connection_enabled and (
            self.endpoint is None or self.diagnostic_credential is None
        ):
            raise ValueError("启用只读数据库连接时必须配置 Endpoint 和诊断凭据")
        if self.controlled_change_enabled and (
            not self.readonly_connection_enabled
            or self.endpoint is None
            or self.execution_credential is None
        ):
            raise ValueError("允许受控变更时必须启用只读连接并配置执行凭据")
        if not self.readonly_connection_enabled and (
            self.endpoint is not None
            or self.diagnostic_credential is not None
            or self.execution_credential is not None
            or self.controlled_change_enabled
        ):
            raise ValueError("仅监控 Target 不能携带数据库连接或执行凭据")
        if self.db_type not in {
            DatabaseType.MYSQL,
            DatabaseType.POSTGRESQL,
        } and (
            self.workload_snapshot_policy.enabled
            or self.activity_sampler_policy.enabled
        ):
            raise ValueError(
                "只有MySQL或PostgreSQL Target可以启用工作负载快照或活动采样"
            )
        if self.endpoint is None:
            if self.oracle_container_scope is not None or self.oracle_pdb_name is not None:
                raise ValueError("仅 Oracle 直连 Target 可以声明容器范围")
            return self
        if self.db_type == DatabaseType.ORACLE:
            if self.endpoint.tls_profile_ref is not None:
                raise ValueError("当前只有PostgreSQL Target支持受控TLS Profile")
            if not self.endpoint.service or self.endpoint.database:
                raise ValueError("Oracle Endpoint 必须只设置 service")
            _validate_oracle_container_expectation(
                self.oracle_container_scope,
                self.oracle_pdb_name,
            )
        elif self.db_type in {DatabaseType.MYSQL, DatabaseType.POSTGRESQL}:
            if not self.endpoint.database or self.endpoint.service:
                raise ValueError("MySQL/PostgreSQL Endpoint 必须只设置 database")
            if (
                self.db_type != DatabaseType.POSTGRESQL
                and self.endpoint.tls_profile_ref is not None
            ):
                raise ValueError("当前只有PostgreSQL Target支持受控TLS Profile")
        elif self.oracle_container_scope is not None or self.oracle_pdb_name is not None:
            raise ValueError("非 Oracle Target 不能声明 Oracle 容器范围")
        return self


class TargetConnectionTest(AIOpsContract):
    db_type: DatabaseType
    version_code: SupportedDatabaseVersion
    endpoint: TargetEndpoint
    diagnostic_credential: DatabaseCredentialInput
    oracle_container_scope: OracleContainerScope | None = None
    oracle_pdb_name: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_database_endpoint(self) -> "TargetConnectionTest":
        validate_supported_database_version(self.db_type, self.version_code)
        if self.db_type == DatabaseType.ORACLE:
            if self.endpoint.tls_profile_ref is not None:
                raise ValueError("当前只有PostgreSQL Target支持受控TLS Profile")
            if not self.endpoint.service or self.endpoint.database:
                raise ValueError("Oracle Endpoint 必须只设置 service")
            _validate_oracle_container_expectation(
                self.oracle_container_scope,
                self.oracle_pdb_name,
            )
        elif not self.endpoint.database or self.endpoint.service:
            raise ValueError("MySQL/PostgreSQL Endpoint 必须只设置 database")
        elif (
            self.db_type != DatabaseType.POSTGRESQL
            and self.endpoint.tls_profile_ref is not None
        ):
            raise ValueError("当前只有PostgreSQL Target支持受控TLS Profile")
        elif self.oracle_container_scope is not None or self.oracle_pdb_name is not None:
            raise ValueError("非 Oracle Target 不能声明 Oracle 容器范围")
        return self


class TargetConnectionTestResult(AIOpsContract):
    ok: bool
    database_version: str | None = None
    supported_version_code: SupportedDatabaseVersion | None = None
    server_uuid: str | None = Field(default=None, max_length=128)
    server_started_at: UtcDatetime | None = None
    capability_probe_version: str | None = Field(default=None, max_length=64)
    discovered_capabilities: tuple[str, ...] = ()
    discovered_privileges: tuple[str, ...] = ()
    capability_details: JsonObject = Field(default_factory=dict)
    oracle_container_scope: OracleContainerScope | None = None
    oracle_container_name: str | None = None
    oracle_container_number: int | None = Field(default=None, ge=0)
    oracle_database_name: str | None = None
    error_code: str | None = Field(default=None, max_length=128)


class TargetPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    version_code: SupportedDatabaseVersion | None = None
    environment: Literal["PROD", "STG", "DEV"] | None = None
    db_role: Literal["PRIMARY", "STANDBY", "UNKNOWN"] | None = None
    oracle_container_scope: OracleContainerScope | None = None
    oracle_pdb_name: str | None = Field(default=None, min_length=1, max_length=128)
    endpoint: TargetEndpoint | None = None
    readonly_connection_enabled: bool | None = None
    controlled_change_enabled: bool | None = None
    importance_level: int | None = Field(default=None, ge=1, le=5)
    capabilities: JsonObject | None = None
    workload_snapshot_policy: WorkloadPolicy | None = None
    activity_sampler_policy: ActivitySamplerPolicy | None = None


class TargetSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    target_id: UUIDv7
    display_name: str
    db_type: DatabaseType
    environment: str
    importance_level: int = Field(ge=1, le=5)
    status: TargetStatus
    connectivity_status: ConnectivityStatus
    observed_status: ObservedStatus
    readonly_connection_enabled: bool
    controlled_change_enabled: bool
    connectivity_check_pending: bool
    diagnostic_credential_configured: bool
    execution_credential_configured: bool
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class TargetDetail(TargetSummary):
    version_code: str | None = None
    db_role: str
    oracle_container_scope: OracleContainerScope | None = None
    oracle_pdb_name: str | None = None
    observed_oracle_container_scope: OracleContainerScope | None = None
    observed_oracle_container_name: str | None = None
    observed_oracle_container_number: int | None = Field(default=None, ge=0)
    observed_oracle_database_name: str | None = None
    endpoint: TargetEndpoint | None = None
    diagnostic_credential: DatabaseCredentialStatus
    execution_credential: DatabaseCredentialStatus
    capabilities: JsonObject
    workload_snapshot_policy: WorkloadPolicy
    activity_sampler_policy: ActivitySamplerPolicy
    activity_sampler_status: Literal["DISABLED", "READY", "DEGRADED"]
    activity_sampler_disabled_reason: str | None = None
    workload_next_run_at: UtcDatetime | None = None
    workload_consecutive_failures: int = Field(ge=0)
    workload_last_collected_at: UtcDatetime | None = None
    workload_last_error_code: str | None = None
    activity_next_sample_at: UtcDatetime | None = None
    activity_consecutive_failures: int = Field(ge=0)
    activity_daily_bytes: int = Field(ge=0)
    activity_last_sampled_at: UtcDatetime | None = None
    connectivity_version: int = Field(ge=1)
    last_observed_at: UtcDatetime | None = None
    last_connectivity_check_at: UtcDatetime | None = None
    last_connectivity_success_at: UtcDatetime | None = None
    last_error_code: str | None = None
    created_at: UtcDatetime
    created_by: str
    updated_by: str


class TargetPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[TargetSummary, ...] = ()


TargetFactType = Literal["ASM_DISKGROUP", "DATAFILE_PATH", "TABLESPACE_PLACEMENT"]
TargetFactSource = Literal["MANUAL_CONFIRMED", "DISCOVERED"]
TargetFactStatus = Literal["ACTIVE", "RETIRED"]
RecoveryAssuranceLevel = Literal[
    "BACKUP_METADATA",
    "RESTORE_VALIDATE",
    "DATABASE_OPEN",
    "APPLICATION_VALIDATED",
]
RecoveryDrillResult = Literal["PASS", "FAIL", "INCONCLUSIVE"]
RecoveryDrillStatus = Literal["SUBMITTED", "VERIFIED", "REJECTED"]
RecoveryBackupSourceType = Literal[
    "ORACLE_RMAN",
    "POSTGRESQL_BASEBACKUP",
    "POSTGRESQL_PGBACKREST",
    "POSTGRESQL_BARMAN",
    "POSTGRESQL_WALG",
    "MYSQL_XTRABACKUP",
    "MYSQL_ENTERPRISE_BACKUP",
    "MYSQL_LOGICAL_DUMP",
    "FILESYSTEM_SNAPSHOT",
    "STORAGE_SNAPSHOT",
    "CLOUD_MANAGED_BACKUP",
    "THIRD_PARTY_BACKUP",
]


class OracleRecoveryMarker(AIOpsContract):
    kind: Literal["ORACLE"] = "ORACLE"
    scn: int | None = Field(default=None, ge=0)
    resetlogs_id: int | None = Field(default=None, ge=0)
    incarnation: int | None = Field(default=None, ge=0)


class PostgreSQLRecoveryMarker(AIOpsContract):
    kind: Literal["POSTGRESQL"] = "POSTGRESQL"
    timeline_id: int | None = Field(default=None, ge=1)
    lsn: str | None = Field(
        default=None,
        pattern=r"^[0-9A-Fa-f]+/[0-9A-Fa-f]+$",
    )


class MySQLRecoveryMarker(AIOpsContract):
    kind: Literal["MYSQL"] = "MYSQL"
    gtid_executed: str | None = Field(default=None, max_length=4000)
    binlog_file: str | None = Field(default=None, max_length=256)
    binlog_position: int | None = Field(default=None, ge=0)


RecoveryMarker = Annotated[
    OracleRecoveryMarker | PostgreSQLRecoveryMarker | MySQLRecoveryMarker,
    Field(discriminator="kind"),
]


class TargetRecoveryProfileUpsert(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    rpo_seconds: int = Field(ge=0, le=31_536_000)
    rto_seconds: int = Field(ge=60, le=31_536_000)
    required_drill_interval_days: int = Field(ge=1, le=3650)
    required_assurance_level: RecoveryAssuranceLevel
    required_backup_source_types: tuple[RecoveryBackupSourceType, ...] = Field(
        min_length=1, max_length=16
    )
    rto_clock_basis: Literal["SERVICE_UNAVAILABLE_TO_VALIDATED"] = (
        "SERVICE_UNAVAILABLE_TO_VALIDATED"
    )
    source_note: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_source_types(self) -> "TargetRecoveryProfileUpsert":
        if len(set(self.required_backup_source_types)) != len(
            self.required_backup_source_types
        ):
            raise ValueError("必需备份来源类型不能重复")
        return self


class TargetRecoveryProfileView(TargetRecoveryProfileUpsert):
    recovery_profile_id: UUIDv7
    target_id: UUIDv7
    version_no: int = Field(ge=1)
    status: Literal["ACTIVE", "RETIRED"]
    effective_at: UtcDatetime
    retired_at: UtcDatetime | None = None
    confirmed_by: str
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class RecoveryDrillEvidence(AIOpsContract):
    evidence_kind: Literal[
        "RMAN_LOG",
        "DATABASE_LOG",
        "BACKUP_TOOL_LOG",
        "IDENTITY_CHECK",
        "DATABASE_CHECK",
        "APPLICATION_CHECK",
        "EXTERNAL_REPORT",
    ]
    reference: str = Field(min_length=1, max_length=2048)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class RecoveryDrillCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    scenario: Literal[
        "FULL_INSTANCE",
        "POINT_IN_TIME",
        "DATABASE_OR_PDB",
        "DATAFILE",
        "CONTROLFILE_OR_SYSTEM",
        "LOGICAL_RESTORE",
    ]
    assurance_level: RecoveryAssuranceLevel
    backup_source_type: RecoveryBackupSourceType
    environment: Literal["ISOLATED", "STG", "DEV", "PROD"]
    result: RecoveryDrillResult
    simulated_failure_at: UtcDatetime
    recovered_through_at: UtcDatetime | None = None
    service_validated_at: UtcDatetime | None = None
    recovery_marker: RecoveryMarker
    evidence: tuple[RecoveryDrillEvidence, ...] = Field(default=(), max_length=32)
    notes: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_drill_semantics(self) -> "RecoveryDrillCreate":
        source_db_type = {
            "ORACLE_RMAN": "ORACLE",
            "POSTGRESQL_BASEBACKUP": "POSTGRESQL",
            "POSTGRESQL_PGBACKREST": "POSTGRESQL",
            "POSTGRESQL_BARMAN": "POSTGRESQL",
            "POSTGRESQL_WALG": "POSTGRESQL",
            "MYSQL_XTRABACKUP": "MYSQL",
            "MYSQL_ENTERPRISE_BACKUP": "MYSQL",
            "MYSQL_LOGICAL_DUMP": "MYSQL",
        }.get(self.backup_source_type)
        if source_db_type is not None and source_db_type != self.recovery_marker.kind:
            raise ValueError("恢复坐标与备份来源数据库类型不匹配")
        if self.result == "PASS" and self.assurance_level in {
            "DATABASE_OPEN",
            "APPLICATION_VALIDATED",
        } and self.service_validated_at is None:
            raise ValueError("数据库已恢复或业务已验证时必须填写验证完成时间")
        if (
            self.recovered_through_at is not None
            and self.recovered_through_at > self.simulated_failure_at
        ):
            raise ValueError("恢复数据时间不能晚于模拟故障时间")
        if (
            self.service_validated_at is not None
            and self.service_validated_at < self.simulated_failure_at
        ):
            raise ValueError("服务验证时间不能早于模拟故障时间")
        return self


class RecoveryDrillReview(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    decision: Literal["VERIFY", "REJECT"]
    review_note: str | None = Field(default=None, max_length=2000)


class RecoveryDrillView(RecoveryDrillCreate):
    drill_id: UUIDv7
    target_id: UUIDv7
    recovery_profile_id: UUIDv7 | None = None
    recovery_profile_version: int | None = Field(default=None, ge=1)
    db_type: DatabaseType
    status: RecoveryDrillStatus
    achieved_rpo_seconds: int | None = Field(default=None, ge=0)
    achieved_rto_seconds: int | None = Field(default=None, ge=0)
    source_trust_level: Literal["USER_PROVIDED", "SOURCE_VERIFIED"]
    reviewed_by: str | None = None
    reviewed_at: UtcDatetime | None = None
    review_note: str | None = None
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    created_by: str
    updated_at: UtcDatetime
    updated_by: str


class RecoveryDrillPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[RecoveryDrillView, ...] = ()


class TargetFactCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    fact_type: TargetFactType
    fact_key: str = Field(min_length=1, max_length=256)
    fact_value: JsonObject


class TargetFactView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    fact_id: UUIDv7
    target_id: UUIDv7
    fact_type: TargetFactType
    fact_key: str
    fact_value: JsonObject
    source: TargetFactSource
    status: TargetFactStatus
    confirmed_by: str | None = None
    confirmed_at: UtcDatetime | None = None
    retired_by: str | None = None
    retired_at: UtcDatetime | None = None
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime
    created_by: str
    updated_by: str


class TargetFactPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[TargetFactView, ...] = ()


def _validate_oracle_container_expectation(
    scope: OracleContainerScope | None,
    pdb_name: str | None,
) -> None:
    if scope is None:
        raise ValueError("Oracle 直连 Target 必须声明容器范围")
    if scope == "PDB" and pdb_name is None:
        raise ValueError("Oracle PDB Target 必须填写 PDB Name")
    if scope != "PDB" and pdb_name is not None:
        raise ValueError("仅 Oracle PDB Target 可以填写 PDB Name")


class NotificationSubscriptionUpsert(AIOpsContract):
    """当前用户针对一个 Target 的站内主动分享订阅。"""

    schema_version: str = PUBLIC_SCHEMA_VERSION
    minimum_severity: Literal["INFO", "WARNING", "HIGH", "CRITICAL"] = "HIGH"
    stages: tuple[NotificationStage, ...] = (
        "SITUATION_DETECTED",
        "DIAGNOSIS_STARTED",
        "REPORT_READY",
        "SITUATION_RECOVERED",
    )

    @model_validator(mode="after")
    def validate_stages(self) -> "NotificationSubscriptionUpsert":
        if not self.stages or len(set(self.stages)) != len(self.stages):
            raise ValueError("主动分享阶段不能为空或重复")
        return self


class NotificationSubscriptionView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    subscription_id: UUIDv7
    target_id: UUIDv7
    recipient_actor_id: str
    channel: Literal["IN_APP"] = "IN_APP"
    minimum_severity: Literal["INFO", "WARNING", "HIGH", "CRITICAL"]
    stages: tuple[NotificationStage, ...]
    status: Literal["ACTIVE", "DISABLED"]
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class NotificationSubscriptionList(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[NotificationSubscriptionView, ...] = ()


class AgentBindingCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    agent_id: UUIDv7
    allow_mutation: bool = False
    policy_id: UUIDv7 | None = None
    allowed_actions: tuple[str, ...] = ()
    change_window: JsonObject | None = None
    max_daily_executions: int | None = Field(default=None, ge=0)


class AgentBindingPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    allow_mutation: bool | None = None
    policy_id: UUIDv7 | None = None
    allowed_actions: tuple[str, ...] | None = None
    change_window: JsonObject | None = None
    max_daily_executions: int | None = Field(default=None, ge=0)


class AgentBindingView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    binding_id: UUIDv7
    target_id: UUIDv7
    agent_id: UUIDv7
    allow_mutation: bool
    policy_id: UUIDv7 | None = None
    allowed_actions: tuple[str, ...] = ()
    change_window: JsonObject | None = None
    max_daily_executions: int | None = None
    status: BindingStatus
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class DiagnosticSourceCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str = Field(min_length=1, max_length=256)
    source_type: str = Field(pattern=r"^[A-Z][A-Z0-9_.-]{1,63}$")
    endpoint: HttpUrl | None = None
    credentials: dict[str, str] | None = Field(
        default=None, json_schema_extra={"writeOnly": True}
    )
    webhook_credentials: dict[str, str] | None = Field(
        default=None, json_schema_extra={"writeOnly": True}
    )
    config: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_endpoint(self) -> "DiagnosticSourceCreate":
        if self.endpoint is not None and (
            self.endpoint.username
            or self.endpoint.password
            or self.endpoint.query
            or self.endpoint.fragment
        ):
            raise ValueError("Diagnostic Source Endpoint 不允许凭证、Query 或 Fragment")
        if self.source_type == "ALERTMANAGER":
            if self.endpoint is None and self.webhook_credentials is None:
                raise ValueError(
                    "Alertmanager 必须配置 Endpoint 或 Webhook 凭据"
                )
        elif self.endpoint is None:
            raise ValueError(
                f"{self.source_type} Diagnostic Source 必须配置 Endpoint"
            )
        if (
            self.source_type not in {"ALERTMANAGER", "ZABBIX"}
            and self.webhook_credentials is not None
        ):
            raise ValueError(
                "只有 Alertmanager 或 Zabbix 可以配置 Webhook 凭据"
            )
        return self


class DiagnosticSourcePatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    endpoint: HttpUrl | None = None
    credentials: dict[str, str] | None = Field(
        default=None, json_schema_extra={"writeOnly": True}
    )
    webhook_credentials: dict[str, str] | None = Field(
        default=None, json_schema_extra={"writeOnly": True}
    )
    config: JsonObject | None = None

    @model_validator(mode="after")
    def validate_endpoint(self) -> "DiagnosticSourcePatch":
        if self.endpoint is not None and (
            self.endpoint.username
            or self.endpoint.password
            or self.endpoint.query
            or self.endpoint.fragment
        ):
            raise ValueError("Diagnostic Source Endpoint 不允许凭证、Query 或 Fragment")
        return self


class DiagnosticSourceConnectionTestResult(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    ok: bool
    error_code: str | None = None
    discovered_capabilities: tuple[str, ...] = ()


class DiagnosticSourceSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    display_name: str
    source_type: str
    adapter_id: str
    adapter_version: str
    status: SourceStatus
    connectivity_status: ConnectivityStatus
    connectivity_check_pending: bool
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class DiagnosticSourceDetail(DiagnosticSourceSummary):
    endpoint: str | None = None
    secret: SecretRefStatus
    webhook_secret: SecretRefStatus
    tls_profile: SecretRefStatus
    declared_capabilities: JsonObject
    discovered_capabilities: JsonObject
    config: JsonObject
    webhook_configured: bool
    connectivity_version: int = Field(ge=1)
    last_connectivity_check_at: UtcDatetime | None = None
    last_connectivity_success_at: UtcDatetime | None = None
    last_error_code: str | None = None
    created_at: UtcDatetime
    created_by: str
    updated_by: str


class DiagnosticSourcePage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[DiagnosticSourceSummary, ...] = ()


class SourceBindingCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    source_locator_key: str = Field(min_length=1, max_length=512)
    source_locator: JsonObject
    role: Literal["PRIMARY", "SUPPLEMENTARY", "FALLBACK"] = "PRIMARY"
    priority: int = Field(default=100, ge=0)
    capability_scope: JsonObject | None = None
    mapping_overrides: JsonObject | None = None
    query_budget: JsonObject | None = None


class SourceBindingPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_locator_key: str | None = Field(default=None, min_length=1, max_length=512)
    source_locator: JsonObject | None = None
    host_candidate_ref: str | None = Field(
        default=None, min_length=1, max_length=8192
    )
    role: Literal["PRIMARY", "SUPPLEMENTARY", "FALLBACK"] | None = None
    priority: int | None = Field(default=None, ge=0)
    capability_scope: JsonObject | None = None
    mapping_overrides: JsonObject | None = None
    query_budget: JsonObject | None = None


class SourceBindingView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    binding_id: UUIDv7
    target_id: UUIDv7
    source_id: UUIDv7
    source_locator_key: str
    source_locator: JsonObject
    role: str
    priority: int
    capability_scope: JsonObject | None = None
    mapping_overrides: JsonObject | None = None
    query_budget: JsonObject | None = None
    status: SourceBindingStatus
    health_status: HealthStatus
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class InstanceDiscoveryRequest(AIOpsContract):
    """受控的监控实例发现请求，不接受 Provider 查询或 Locator。"""

    schema_version: str = PUBLIC_SCHEMA_VERSION
    db_types: tuple[DatabaseType, ...] = ()
    page_size: int = Field(default=50, ge=1, le=100)
    cursor: str | None = Field(default=None, min_length=1, max_length=8192)

    @model_validator(mode="after")
    def validate_db_types(self) -> "InstanceDiscoveryRequest":
        if len(set(self.db_types)) != len(self.db_types):
            raise ValueError("数据库类型过滤条件不能重复")
        return self


class InstanceDiscoveryCandidate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    candidate_ref: str = Field(min_length=1, max_length=8192)
    display_name: str = Field(min_length=1, max_length=256)
    locator_hint: str = Field(min_length=1, max_length=256)
    db_type: DatabaseType
    mapping_status: Literal["UNMAPPED", "MAPPED"]
    mapped_target_id: UUIDv7 | None = None


class HostDiscoveryCandidate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    candidate_ref: str = Field(min_length=1, max_length=8192)
    display_name: str = Field(min_length=1, max_length=256)
    locator_hint: str = Field(min_length=1, max_length=256)


class InstanceDiscoveryPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    items: tuple[InstanceDiscoveryCandidate, ...] = ()
    host_items: tuple[HostDiscoveryCandidate, ...] = ()


class InstanceMappingItem(AIOpsContract):
    candidate_ref: str = Field(min_length=1, max_length=8192)
    host_candidate_ref: str | None = Field(
        default=None, min_length=1, max_length=8192
    )
    target_id: UUIDv7


class InstanceMappingRequest(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    mappings: tuple[InstanceMappingItem, ...] = Field(
        min_length=1, max_length=100
    )

    @model_validator(mode="after")
    def validate_unique_mappings(self) -> "InstanceMappingRequest":
        target_ids = [item.target_id for item in self.mappings]
        candidate_refs = [item.candidate_ref for item in self.mappings]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("批量映射中的 Target 不能重复")
        if len(set(candidate_refs)) != len(candidate_refs):
            raise ValueError("批量映射中的候选实例不能重复")
        return self


class InstanceMappingView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    binding_id: UUIDv7
    target_id: UUIDv7
    source_id: UUIDv7
    locator_hint: str = Field(min_length=1, max_length=256)
    status: SourceBindingStatus
    health_status: HealthStatus
    row_version: int = Field(ge=1)


class InstanceMappingResult(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    items: tuple[InstanceMappingView, ...]


class PolicyCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    policy_key: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    rules: JsonObject


class PolicySummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    policy_id: UUIDv7
    policy_key: str
    version_no: int = Field(ge=1)
    display_name: str
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: PolicyStatus
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class PolicyDetail(PolicySummary):
    rules: JsonObject
    effective_at: UtcDatetime | None = None
    retired_at: UtcDatetime | None = None
    created_at: UtcDatetime
    created_by: str
    updated_by: str


class PolicyPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[PolicySummary, ...] = ()


class InspectionCheckCatalogItem(AIOpsContract):
    check_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=128)
    availability: Literal["READY", "PLANNED"]
    tool_id: str | None = Field(default=None, min_length=1, max_length=128)
    playbook_id: str | None = Field(default=None, min_length=1, max_length=128)
    finding_types: tuple[str, ...] = ()
    default_for: tuple[Literal["DAILY", "WEEKLY"], ...] = Field(min_length=1)
    trend_required: bool
    supported_db_types: tuple[DatabaseType, ...] = ("ORACLE",)


class InspectionCheckCatalogGroup(AIOpsContract):
    group_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,63}$")
    display_name: str = Field(min_length=1, max_length=64)
    checks: tuple[InspectionCheckCatalogItem, ...] = Field(min_length=1)


class InspectionCheckCatalogView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    groups: tuple[InspectionCheckCatalogGroup, ...] = Field(min_length=1)


class InspectionTemplateCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str = Field(min_length=1, max_length=256)
    selected_check_ids: tuple[str, ...] = Field(min_length=1)


class InspectionTemplateVersionCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    selected_check_ids: tuple[str, ...] = Field(min_length=1)


class InspectionTemplateSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    inspection_template_id: UUIDv7
    display_name: str
    status: str
    version_no: int = Field(ge=1)
    selected_check_ids: tuple[str, ...] = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class InspectionPlanCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str = Field(min_length=1, max_length=256)
    agent_id: UUIDv7
    schedule_type: Literal["DAILY", "WEEKLY", "CRON"]
    cron_expression: str = Field(min_length=9, max_length=256)
    timezone: str = Field(min_length=1, max_length=64)
    inspection_template_id: UUIDv7
    timeout_seconds: int = Field(ge=1, le=86400)
    overlap_policy: Literal["SKIP", "QUEUE"] = "SKIP"
    misfire_policy: Literal["SKIP", "LATEST_ONLY"] = "LATEST_ONLY"


class InspectionPlanPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    agent_id: UUIDv7 | None = None
    cron_expression: str | None = Field(default=None, min_length=9, max_length=256)
    timezone: str | None = Field(default=None, min_length=1, max_length=64)
    inspection_template_id: UUIDv7 | None = None
    timeout_seconds: int | None = Field(default=None, ge=1, le=86400)
    overlap_policy: Literal["SKIP", "QUEUE"] | None = None
    misfire_policy: Literal["SKIP", "LATEST_ONLY"] | None = None


class InspectionPlanSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    plan_id: UUIDv7
    display_name: str
    agent_id: UUIDv7
    schedule_type: str
    timezone: str
    status: InspectionPlanStatus
    next_run_at: UtcDatetime | None = None
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class InspectionPlanDetail(InspectionPlanSummary):
    cron_expression: str
    inspection_template_id: UUIDv7
    inspection_template_version_id: UUIDv7
    inspection_template_name: str
    inspection_template_version: int = Field(ge=1)
    selected_check_ids: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: int
    overlap_policy: str
    misfire_policy: str
    agent_target_count: int = Field(ge=0)
    created_at: UtcDatetime
    created_by: str
    updated_by: str


class InspectionPlanPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[InspectionPlanSummary, ...] = ()


class WebhookKeyRotation(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    webhook_key: str = Field(min_length=32, max_length=256)
    previous_key_expires_at: UtcDatetime | None = None
    created_at: UtcDatetime


class ConfigListQuery(AIOpsContract):
    """内部 Client 对列表过滤条件的规范表达。"""

    status: str | None = Field(default=None, max_length=32)
    cursor: str | None = Field(default=None, max_length=2048)
    limit: int = Field(default=50, ge=1, le=200)


class ConfigCommandReceipt(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    resource_id: UUIDv7
    status: str
    row_version: int = Field(ge=1)
    accepted_at: UtcDatetime
