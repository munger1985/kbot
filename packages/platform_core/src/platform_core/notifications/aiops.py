"""AIOps 通知的跨平台事实、严重程度与待办身份合同。"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AIOpsNotificationFacts(BaseModel):
    """业务版本与输出任务版本分开，禁止用通知时间猜测交互身份。"""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["INFO", "WARNING", "HIGH", "CRITICAL"] | None = None
    hitl_id: UUID | None = None
    aggregate_version: int | None = Field(default=None, ge=1)
    source_status: str | None = Field(default=None, max_length=40)


def work_item_identity(*, resource_type: str, resource_id: str, facts: dict) -> tuple[str, str]:
    """一个 HITL 对应一个不可重开的待办；不同输入请求不得合并。"""
    hitl_id = facts.get("hitl_id")
    if resource_type == "aiops_run" and hitl_id is not None:
        return "aiops_hitl", str(UUID(str(hitl_id)))
    return resource_type, str(resource_id)


def work_item_source_key(*, resource_type: str, resource_id: str, facts: dict) -> str:
    kind, identity = work_item_identity(resource_type=resource_type, resource_id=resource_id, facts=facts)
    return f"{kind}:{identity}"


def presentation_severity(facts: dict, default: str) -> str:
    """通知紧迫度保留 CRITICAL，HIGH 同样进入高优先级展示；原级别另行保留。"""
    severity = AIOpsNotificationFacts.model_validate({
        key: facts[key] for key in ("severity",) if key in facts
    }).severity
    return "CRITICAL" if severity in {"HIGH", "CRITICAL"} else severity or default
