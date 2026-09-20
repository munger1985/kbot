"""AIOps Finding Card 契约；字段由 Compiler 从证据列映射，不经模型编写。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .types import AIOpsContract, JsonObject


class FindingType(StrEnum):
    LOCK_WAIT = "LOCK_WAIT"
    LONG_SESSION = "LONG_SESSION"
    DG_LAG = "DG_LAG"
    WAIT_CLASS = "WAIT_CLASS"
    TABLESPACE = "TABLESPACE"
    SQL_STATS_STALE = "SQL_STATS_STALE"
    EXACHECK_FAIL = "EXACHECK_FAIL"
    EXACHECK_WARNING = "EXACHECK_WARNING"
    REPLICATION_LAG = "REPLICATION_LAG"
    DEAD_TUPLES = "DEAD_TUPLES"
    AUTOVACUUM = "AUTOVACUUM"
    IDLE_SESSION = "IDLE_SESSION"
    CONNECTION_USAGE = "CONNECTION_USAGE"
    QUERY_RATE = "QUERY_RATE"
    INVALID_OBJECT = "INVALID_OBJECT"
    ARCHIVE_HEADROOM = "ARCHIVE_HEADROOM"
    BACKUP_FAILED = "BACKUP_FAILED"
    LONG_TRANSACTION = "LONG_TRANSACTION"
    TOP_SQL = "TOP_SQL"


class FindingSeverity(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class FindingConfirmation(StrEnum):
    CONFIRMED = "CONFIRMED"
    LIKELY = "LIKELY"
    POSSIBLE = "POSSIBLE"
    UNKNOWN = "UNKNOWN"


class FindingObjectRef(AIOpsContract):
    target_id: str | None = Field(default=None, max_length=64)
    instance_id: int | None = None
    object_kind: str = Field(min_length=1, max_length=64)
    object_name: str | None = Field(default=None, max_length=256)
    sql_id: str | None = Field(default=None, max_length=32)


class FindingThreshold(AIOpsContract):
    metric: str = Field(min_length=1, max_length=64)
    current: Any = None
    operator: str = Field(min_length=1, max_length=16)
    limit: Any = None


class FindingCard(AIOpsContract):
    schema_version: Literal["AIOPS_FINDING.v1"] = "AIOPS_FINDING.v1"
    finding_id: str = Field(min_length=1, max_length=128)
    finding_type: FindingType
    severity: FindingSeverity
    confirmation: FindingConfirmation
    object_ref: FindingObjectRef
    fields: JsonObject = Field(default_factory=dict)
    threshold: FindingThreshold | None = None
    impact: str = Field(min_length=1, max_length=500)
    evidence_refs: tuple[str, ...] = ()
    playbook_id: str | None = Field(default=None, max_length=128)


class FindingColumnGap(AIOpsContract):
    finding_type: FindingType
    source_tool_id: str = Field(min_length=1, max_length=128)
    column: str = Field(min_length=1, max_length=64)
    code: str = Field(min_length=1, max_length=64)
    detail: str = Field(min_length=1, max_length=500)
    evidence_ref: str | None = Field(default=None, max_length=512)


class FindingCompilation(AIOpsContract):
    schema_version: Literal["AIOPS_FINDING_CARDS_BLOCK.v1"] = (
        "AIOPS_FINDING_CARDS_BLOCK.v1"
    )
    findings: tuple[FindingCard, ...] = ()
    empty_reasons: tuple[str, ...] = ()
    gaps: tuple[FindingColumnGap, ...] = ()
