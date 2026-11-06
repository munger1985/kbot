"""外部投递的共同合同；厂商接受不等于已读或业务完成。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Literal, Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

_ID_NAMESPACE = UUID("5a782d47-dd39-5d42-9834-944d06734b49")


class DeliveryStatus(StrEnum):
    PENDING = "PENDING"
    SENDING = "SENDING"
    ACCEPTED = "ACCEPTED"
    RETRY_WAIT = "RETRY_WAIT"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"
    DEAD = "DEAD"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeliveryScope(StrictModel):
    domain_id: str = Field(min_length=1, max_length=64)
    tenant_id: UUID | None = None


class ConnectionCreate(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    provider: Literal["FEISHU"] = "FEISHU"
    app_id: str = Field(pattern=r"^cli_[a-zA-Z0-9]{4,100}$")
    app_secret: SecretStr = Field(min_length=1, max_length=8192)
    enabled: bool = False


class ConnectionUpdate(StrictModel):
    expected_row_version: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=128)
    enabled: bool
    app_secret: SecretStr | None = Field(default=None, min_length=1, max_length=8192)


class BindingCreate(StrictModel):
    connection_id: UUID
    actor_id: str | None = Field(default=None, min_length=1, max_length=256)
    target_type: Literal["open_id", "chat_id"] = "open_id"
    external_id: str = Field(min_length=4, max_length=256)

    @model_validator(mode="after")
    def check_target(self):
        prefix = "ou_" if self.target_type == "open_id" else "oc_"
        if not self.external_id.startswith(prefix):
            raise ValueError("NOTIFICATION_EXTERNAL_ID_INVALID")
        if (self.target_type == "open_id") != (self.actor_id is not None):
            raise ValueError("NOTIFICATION_BINDING_ACTOR_INVALID")
        return self


class BindingConfirmation(StrictModel):
    code: str = Field(pattern=r"^[0-9]{6}$")
    expected_row_version: int = Field(ge=1)


class ExternalPolicyCreate(StrictModel):
    connection_id: UUID
    event_types: list[str] = Field(min_length=1, max_length=50)
    binding_ids: list[UUID] = Field(default_factory=list, max_length=100)
    actor_ids: list[str] = Field(default_factory=list, max_length=100)
    enabled: bool = False

    @field_validator("event_types")
    @classmethod
    def events(cls, values):
        if len(set(values)) != len(values) or any(value not in EXTERNAL_EVENTS for value in values):
            raise ValueError("NOTIFICATION_EXTERNAL_EVENT_INVALID")
        return values

    @model_validator(mode="after")
    def targets(self):
        if not self.binding_ids and not self.actor_ids:
            raise ValueError("NOTIFICATION_POLICY_TARGET_REQUIRED")
        if any(not value.strip() or len(value) > 256 for value in self.actor_ids):
            raise ValueError("NOTIFICATION_POLICY_ACTOR_INVALID")
        return self


class RetryRequest(StrictModel):
    expected_row_version: int = Field(ge=1)
    acknowledge_duplicate_risk: bool = False


# 外发文字全部来自版本化目录，不插入业务 summary、名称或任意 safe_payload。
EXTERNAL_EVENTS = {
    "aiops.situation.detected": ("数据库异常已确认", "有告警需要查看。"),
    "aiops.situation.escalated": ("数据库异常等级已升级", "请及时查看原告警。"),
    "aiops.situation.recovered": ("数据库异常已恢复", "请查看恢复状态。"),
    "aiops.proposal.review_required": ("变更方案等待审批", "请登录原任务核对并处理。"),
    "aiops.proposal.approved": ("变更方案已批准", "批准不表示执行完成，请查看原任务。"),
    "aiops.proposal.rejected": ("变更方案已拒绝", "请查看原任务处理结果。"),
    "aiops.proposal.expired": ("变更方案已过期", "此方案已不能审批。"),
    "aiops.proposal.invalidated": ("变更方案已失效", "此方案已不能审批。"),
    "aiops.diagnostic.input_required": ("诊断等待人工处理", "请登录原任务完成授权或补充证据。"),
    "aiops.diagnostic.input_resolved": ("人工交互已处理", "请查看原任务。"),
    "aiops.diagnostic.input_expired": ("人工交互已过期", "请查看原任务当前状态。"),
    "aiops.diagnostic.input_cancelled": ("人工交互已取消", "请查看原任务当前状态。"),
    "aiops.execution.started": ("变更开始执行", "开始不表示执行成功。"),
    "aiops.execution.succeeded": ("变更执行成功，效果待验证", "请查看执行回执并核验效果。"),
    "aiops.execution.failed": ("变更执行失败", "请查看原执行记录。"),
    "aiops.execution.cancelled": ("变更执行已取消", "请查看原执行记录。"),
    "aiops.execution.unknown": ("变更执行结果未知", "请核验实际结果，勿直接重复执行。"),
    "aiops.execution.timed_out": ("变更执行开始前已超时", "请查看原执行记录。"),
    "aiops.proposal.manual_result_recorded": ("人工执行结果已登记", "请查看原任务登记结果。"),
    "aiops.report.ready": ("报告已生成", "请登录系统查看报告。"),
}
ACTIONABLE_EVENTS = frozenset({"aiops.proposal.review_required", "aiops.diagnostic.input_required"})
GROUP_EVENTS = frozenset(key for key in EXTERNAL_EVENTS if key.startswith("aiops.situation."))


def stable_id(*parts: Any) -> UUID:
    """用结构化身份派生主键，避免分隔符碰撞及先查再插竞态。"""
    return uuid5(_ID_NAMESPACE, json.dumps([str(part) for part in parts], separators=(",", ":")))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def render_card(*, event_type: str, url: str, severity: str | None = None, group: bool = False) -> dict:
    title, message = EXTERNAL_EVENTS[event_type]
    if group:
        if event_type not in GROUP_EVENTS:
            raise ValueError("NOTIFICATION_GROUP_EVENT_DENIED")
        title, message = "有告警事项待查看", "请登录系统核对当前权限及告警。"
    elif severity in {"INFO", "WARNING", "HIGH", "CRITICAL"} and event_type in GROUP_EVENTS:
        message += f" 原始等级：{severity}。"
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"},
        "elements": [
            {"tag": "div", "text": {"tag": "plain_text", "content": message}},
            {"tag": "action", "actions": [{"tag": "button",
                "text": {"tag": "plain_text", "content": "查看详情／进入处理"},
                "type": "primary", "url": url}]},
        ],
    }


@dataclass(frozen=True, slots=True)
class SendResult:
    status: DeliveryStatus
    error_code: str | None = None
    message_id: str | None = None
    request_id: str | None = None
    retry_after_seconds: float | None = None
    elapsed_ms: int | None = None


class IMNotificationSender(Protocol):
    async def send(self, *, connection: dict, secret: str, target_type: str,
                   external_id: str, content: dict, provider_uuid: str) -> SendResult: ...


def can_retry_uncertain(first_sent_at: datetime | None, now: datetime, *, window_seconds: int = 3300) -> bool:
    """厂商只承诺一小时去重，保留五分钟安全余量且不接受未来时间。"""
    return first_sent_at is not None and timedelta(0) <= now - first_sent_at < timedelta(seconds=window_seconds)
