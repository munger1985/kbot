"""只读查询生命周期不是 SQL 诊断证据，也不构成新的执行授权。"""

from typing import Literal

from pydantic import Field, model_validator

from .types import AIOpsContract, UUIDv7, UtcDatetime


class ReadonlyTerminationObservation(AIOpsContract):
    db_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    status: Literal["CANCEL_REQUESTED", "TERMINATION_CONFIRMED", "TERMINATION_UNKNOWN"]
    method: Literal["NONE", "POSTGRESQL_PROTOCOL_CLOSE", "MYSQL_KILL_OWN_QUERY", "ORACLE_CANCEL"]
    observed_at: UtcDatetime

    @model_validator(mode="after")
    def validate_termination(self):
        methods = {"POSTGRESQL": "POSTGRESQL_PROTOCOL_CLOSE", "MYSQL": "MYSQL_KILL_OWN_QUERY", "ORACLE": "ORACLE_CANCEL"}
        if self.method != "NONE" and self.method != methods[self.db_type]:
            raise ValueError("终止方式必须与目标数据库一致")
        if self.status == "TERMINATION_CONFIRMED" and (self.db_type == "ORACLE" or self.method == "NONE"):
            raise ValueError("取消请求或本地断连不能作为数据库终止确认")
        if self.status == "CANCEL_REQUESTED" and self.method != "ORACLE_CANCEL":
            raise ValueError("仅 Oracle 同步取消接口采用请求已发送口径")
        return self


class ReadonlyTerminationReceipt(AIOpsContract):
    lease_id: UUIDv7
    artifact_id: UUIDv7
    recorded_at: UtcDatetime


class ReadonlyLifecycleItem(AIOpsContract):
    lease_id: UUIDv7
    acquired_at: UtcDatetime
    execution_deadline_at: UtcDatetime
    status: Literal["IN_PROGRESS", "RELEASED", "CANCEL_REQUESTED", "TERMINATION_CONFIRMED", "TERMINATION_UNKNOWN"]
    observation: ReadonlyTerminationObservation | None = None
    artifact_id: UUIDv7 | None = None
    recorded_at: UtcDatetime | None = None


class ReadonlyLifecyclePage(AIOpsContract):
    items: tuple[ReadonlyLifecycleItem, ...] = Field(default=(), max_length=50)
    truncated: bool = False
