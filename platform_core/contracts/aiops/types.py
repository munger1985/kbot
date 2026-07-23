"""AIOps Wire 层共享类型；不包含领域状态迁移规则。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


PUBLIC_SCHEMA_VERSION = "aiops.public.v1"
INTERNAL_SCHEMA_VERSION = "aiops.internal.v1"
EXECUTOR_SCHEMA_VERSION = "aiops.executor.v1"
EVENT_SCHEMA_VERSION = "aiops.event.v1"


def _require_uuid7(value: UUID) -> UUID:
    if value.version != 7:
        raise ValueError("AIOps 资源 ID 必须是 UUIDv7")
    return value


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("时间必须包含 UTC 时区")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError("时间必须使用 UTC")
    return value


UUIDv7 = Annotated[UUID, AfterValidator(_require_uuid7)]
UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SecretRef = Annotated[
    str,
    Field(min_length=8, max_length=2048, pattern=r"^[a-z][a-z0-9+.-]*://.+"),
]


class AIOpsContract(BaseModel):
    """所有 AIOps Wire DTO 的严格基类。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        use_enum_values=True,
    )


class DatabaseType(StrEnum):
    ORACLE = "ORACLE"
    MYSQL = "MYSQL"


class ExecutionMode(StrEnum):
    ADVISORY = "ADVISORY"
    AGENT_EXECUTE = "AGENT_EXECUTE"


class ResourceStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    REVOKED = "REVOKED"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"


class TriggerType(StrEnum):
    CHAT = "CHAT"
    API = "API"
    ROOT = "ROOT"
    ALERT = "ALERT"
    SCHEDULE = "SCHEDULE"


class OpsRunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_INPUT = "WAITING_INPUT"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class RootCauseGrade(StrEnum):
    CONFIRMED = "CONFIRMED"
    PROBABLE = "PROBABLE"
    POSSIBLE = "POSSIBLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class HitlType(StrEnum):
    DATA_REQUIRED = "DATA_REQUIRED"
    MANUAL_DIAGNOSTIC_SQL = "MANUAL_DIAGNOSTIC_SQL"
    CHANGE_APPROVAL = "CHANGE_APPROVAL"


class HitlStatus(StrEnum):
    PENDING = "PENDING"
    ANSWERED = "ANSWERED"
    SKIPPED = "SKIPPED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class ProposalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExecutionStatus(StrEnum):
    CREATED = "CREATED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class ReportType(StrEnum):
    INCIDENT = "INCIDENT"
    PERFORMANCE = "PERFORMANCE"
    INSPECTION_DAILY = "INSPECTION_DAILY"
    INSPECTION_WEEKLY = "INSPECTION_WEEKLY"
    COMPARISON = "COMPARISON"


class ReportStatus(StrEnum):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    SUPERSEDED = "SUPERSEDED"


class CommandType(StrEnum):
    CANCEL_RUN = "CANCEL_RUN"
    ANSWER_HITL = "ANSWER_HITL"
    CANCEL_HITL = "CANCEL_HITL"
    APPROVE_PROPOSAL = "APPROVE_PROPOSAL"
    REJECT_PROPOSAL = "REJECT_PROPOSAL"
    RECORD_MANUAL_RESULT = "RECORD_MANUAL_RESULT"


class ResultFormat(StrEnum):
    JSON = "JSON"
    CSV = "CSV"
    TEXT = "TEXT"


class ResultStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ArtifactRef(AIOpsContract):
    artifact_id: UUIDv7
    artifact_type: str = Field(min_length=1, max_length=64)
    schema_version: str = Field(min_length=1, max_length=64)
    content_hash: Sha256Digest


class ResourceRef(AIOpsContract):
    resource_type: str = Field(min_length=1, max_length=64)
    resource_id: UUIDv7


class CursorPage(AIOpsContract):
    next_cursor: str | None = Field(default=None, max_length=2048)
    has_more: bool = False


JsonObject = dict[str, Any]
