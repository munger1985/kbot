"""数据保护配置公共契约；不复用诊断身份或人工演练记录。"""

from datetime import date
from re import fullmatch
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .types import AIOpsContract, DatabaseType, UUIDv7, UtcDatetime

ProtectionTool = Literal["RMAN", "PGBACKREST", "XTRABACKUP"]
ProtectionBackupKind = Literal["FULL", "INCREMENTAL"]
ProtectionImportance = Literal["CRITICAL", "HIGH", "MEDIUM", "LOWER", "LOW"]
ProtectionTopology = Literal["UNKNOWN", "SINGLE", "ASYNC_REPLICA", "SYNC_REPLICA"]
ProtectionKey = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")]


class ProtectionSchedule(AIOpsContract):
    timezone: str = Field(min_length=1, max_length=96)
    cron: str | None = Field(default=None, max_length=128)
    once_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_trigger(self):
        if (self.cron is None) == (self.once_at is None):
            raise ValueError("必须选择一次性时间或 Cron，不能同时设置")
        return self


class ProtectionBackupRotation(AIOpsContract):
    timezone: str = Field(min_length=1, max_length=96)
    start_date: date
    cycle_days: int = Field(strict=True, ge=2, le=365)
    full_time: str = Field(pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")
    incremental_time: str = Field(pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")


class ProtectionPlanSettings(AIOpsContract):
    rto_seconds: int | None = Field(default=None, gt=0, le=31536000)
    rpo_seconds: int | None = Field(default=None, ge=0, le=31536000)
    importance: ProtectionImportance | None = None
    topology: ProtectionTopology = "UNKNOWN"
    tool: ProtectionTool
    storage_key: ProtectionKey | None = None
    backup_directory: str | None = Field(default=None, max_length=512, pattern=r"^(/[a-zA-Z0-9_.-]+)+$")
    backup_parallelism: int = Field(default=1, strict=True, ge=1, le=32)
    backup_schedule: ProtectionSchedule | None = None
    backup_incremental_schedule: ProtectionSchedule | None = None
    backup_rotation: ProtectionBackupRotation | None = None
    restore_template_key: ProtectionKey | None = None
    drill_destination_key: ProtectionKey | None = None
    drill_schedule: ProtectionSchedule | None = None
    timeout_seconds: int = Field(default=7200, ge=60, le=172800)
    schedule_window_seconds: int = Field(default=900, ge=60, le=86400)

    @field_validator("backup_directory")
    @classmethod
    def validate_backup_directory(cls, value: str | None) -> str | None:
        if value is not None and (
            not fullmatch(r"(/[a-zA-Z0-9_.-]+)+", value) or any(part in {".", ".."} for part in value.split("/"))
        ):
            raise ValueError("备份目录必须是规范绝对路径，不允许路径跳转或命令字符")
        return value

    @model_validator(mode="after")
    def validate_backup_schedule(self):
        full, incremental = self.backup_schedule, self.backup_incremental_schedule
        if self.backup_rotation and (full or incremental):
            raise ValueError("按天轮转不能与备份 Cron 同时设置")
        if any(schedule and schedule.once_at for schedule in (full, incremental)):
            raise ValueError("备份计划须为周期任务")
        if incremental and (full is None or full.timezone != incremental.timezone):
            raise ValueError("混合备份须在同一时区配置全量和增量 Cron")
        return self

    @property
    def backup_scheduled(self) -> bool:
        return self.backup_schedule is not None or self.backup_rotation is not None

    @property
    def incremental_enabled(self) -> bool:
        return self.backup_incremental_schedule is not None or self.backup_rotation is not None


class ProtectionIdentityInput(AIOpsContract):
    model_config = ConfigDict(hide_input_in_errors=True)
    username: str = Field(min_length=1, max_length=256, repr=False)
    password: str = Field(min_length=1, max_length=4096, repr=False)


class ProtectionBackupIdentityInput(AIOpsContract):
    model_config = ConfigDict(hide_input_in_errors=True)
    auth_mode: Literal["DATABASE_PASSWORD", "ORACLE_OS"] = "DATABASE_PASSWORD"
    username: str = Field(min_length=1, max_length=256, repr=False)
    password: str | None = Field(default=None, min_length=1, max_length=4096, repr=False)

    @model_validator(mode="after")
    def check_auth_mode(self):
        if self.auth_mode == "ORACLE_OS":
            if self.password is not None or not fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.-]{0,63}", self.username):
                raise ValueError("本机 Oracle OS 认证只填写现有的操作系统用户名")
        elif self.password is None:
            raise ValueError("数据库专用身份需要密码")
        return self


class ProtectionPlanSave(AIOpsContract):
    expected_revision: int = Field(ge=0)
    settings: ProtectionPlanSettings
    backup_identity: ProtectionBackupIdentityInput | None = Field(default=None, repr=False)
    restore_identity: ProtectionIdentityInput | None = Field(default=None, repr=False)


class ProtectionBackupScheduleChange(AIOpsContract):
    expected_revision: int = Field(ge=1, strict=True)
    enabled: bool = Field(strict=True)


class ProtectionDrillScheduleChange(AIOpsContract):
    expected_revision: int = Field(ge=1, strict=True)
    enabled: bool = Field(strict=True)


class ProtectionCheck(AIOpsContract):
    area: Literal["DATABASE", "HOST", "STORAGE", "NETWORK_TOOL", "VERIFY"]
    operation: Literal["BACKUP", "DRILL"]
    status: Literal["MISSING", "UNVERIFIED", "PASS"]
    guidance: str


class ProtectionAdvice(AIOpsContract):
    rto_seconds: int | None = None
    rpo_seconds: int | None = None
    explanation: str


class ProtectionRunSummary(AIOpsContract):
    run_id: UUIDv7
    operation: Literal["BACKUP", "DRILL"]
    backup_kind: ProtectionBackupKind | None = None
    message: str | None = None
    scheduled_at: UtcDatetime
    completed_at: UtcDatetime | None = None
    status: Literal["RUNNING", "SUCCEEDED", "FAILED", "REJECTED", "SKIPPED", "UNKNOWN"]
    report_id: UUIDv7 | None = None


class ProtectionPlanView(AIOpsContract):
    resource_id: UUIDv7
    db_type: DatabaseType
    db_version: str | None = None
    db_role: str
    status: Literal["NOT_ENROLLED", "DRAFT", "SCHEDULED"]
    revision: int = Field(ge=0)
    settings: ProtectionPlanSettings
    has_backup_identity: bool = False
    has_restore_identity: bool = False
    readiness: tuple[ProtectionCheck, ...]
    advice: ProtectionAdvice
    backup_template_key: str | None = None
    backup_template_version: str | None = None
    backup_tool_requirement: str | None = None
    next_backup_at: UtcDatetime | None = None
    next_drill_at: UtcDatetime | None = None
    drill_preflight_status: Literal["PENDING", "REJECTED"] | None = None
    recent_runs: tuple[ProtectionRunSummary, ...] = ()
