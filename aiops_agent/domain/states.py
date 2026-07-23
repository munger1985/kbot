"""AIOps 领域状态枚举；Wire 与 Persistence 通过 Mapper 显式转换。"""

from enum import StrEnum


class DomainOpsRunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_INPUT = "WAITING_INPUT"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class DomainHitlStatus(StrEnum):
    PENDING = "PENDING"
    ANSWERED = "ANSWERED"
    SKIPPED = "SKIPPED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class DomainProposalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DomainExecutionStatus(StrEnum):
    CREATED = "CREATED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
