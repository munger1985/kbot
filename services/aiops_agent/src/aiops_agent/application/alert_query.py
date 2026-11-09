"""聊天告警查询的显式筛选合同，不把历史监控恢复当成人工处置。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from aiops_agent.domain.evidence.events import SignalSeverity
from platform_core.contracts.aiops import TimeRangeV1


TERMINAL_SIGNAL_STATUSES = ("RESOLVED", "CLOSED")


class AlertQueryParameters(BaseModel):
    """路由器只能按用户意图选择条件，时间范围使用 TurnPlan.time_range。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["CURRENT", "HISTORY"] = Field(default="CURRENT",
        description="当前健康或当前告警用 CURRENT；明确查历史或恢复记录用 HISTORY")
    status: Literal[
        "OPEN", "ACKNOWLEDGED", "INVESTIGATING", "DIAGNOSED", "MITIGATING",
        "OBSERVING", "SUPPRESSED", "RESOLVED", "CLOSED",
    ] | None = None
    severity: SignalSeverity | None = None
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_current_status(self) -> "AlertQueryParameters":
        if self.mode == "CURRENT" and self.status in TERMINAL_SIGNAL_STATUSES:
            raise ValueError("已恢复或已关闭告警必须使用 HISTORY 模式")
        return self

    def time_bounds(self, time_range: TimeRangeV1 | None, *, now: datetime) -> tuple[datetime, datetime]:
        """默认当前查一小时，历史查一天；显式时间范围优先，不得退回无界查询。"""
        if time_range is not None and time_range.start_at is not None:
            return time_range.start_at, time_range.end_at
        seconds = (time_range.relative_window_seconds if time_range is not None else None)
        seconds = seconds or (3600 if self.mode == "CURRENT" else 86400)
        end = now.astimezone(UTC)
        return end - timedelta(seconds=seconds), end


def signal_status_label(status: str) -> str:
    """仅用于监控信号；不改变工作项和人工效果验证的 RESOLVED 语义。"""
    return {
        "OPEN": "活动告警", "ACKNOWLEDGED": "已确认",
        "INVESTIGATING": "调查中", "DIAGNOSED": "已诊断",
        "MITIGATING": "缓解中", "OBSERVING": "观察中", "SUPPRESSED": "已抑制",
        "RESOLVED": "告警已恢复", "CLOSED": "告警已关闭",
    }.get(status, status)
