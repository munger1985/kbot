"""KBot 通知事件目录；事件类型与投影动作由代码控制。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


ProjectionStatus = Literal[
    "RUNNING", "WAITING_USER", "SUCCEEDED", "PARTIAL", "FAILED"
]


@dataclass(frozen=True, slots=True)
class NotificationEventDefinition:
    event_type: str
    producer_service: str
    category: Literal["ACTION_REQUIRED", "TASK", "RESULT", "SYSTEM"]
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    title: str
    notify: bool = True
    operation_status: ProjectionStatus | None = None
    work_item_action: str | None = None
    resolve_work_item: bool = False
    retention_days: int = 90
    allowed_channels: tuple[str, ...] = ("IN_APP",)


def _event(
    event_type: str,
    producer: str,
    category: Literal["ACTION_REQUIRED", "TASK", "RESULT", "SYSTEM"],
    severity: Literal["INFO", "WARNING", "CRITICAL"],
    title: str,
    **projection,
) -> NotificationEventDefinition:
    return NotificationEventDefinition(
        event_type=event_type,
        producer_service=producer,
        category=category,
        severity=severity,
        title=title,
        **projection,
    )


_EVENTS = (
    _event("agent.run.input_required", "agent-runtime", "ACTION_REQUIRED", "WARNING", "Agent 运行等待输入", operation_status="WAITING_USER", work_item_action="agent.run.input"),
    _event("agent.run.completed", "agent-runtime", "RESULT", "INFO", "Agent 运行完成", operation_status="SUCCEEDED", work_item_action="agent.run.input", resolve_work_item=True),
    _event("agent.run.failed", "agent-runtime", "RESULT", "WARNING", "Agent 运行失败", operation_status="FAILED", work_item_action="agent.run.input", resolve_work_item=True),
    _event("knowledge.ingestion.completed", "knowledge-core", "RESULT", "INFO", "知识处理完成", operation_status="SUCCEEDED"),
    _event("knowledge.ingestion.partial", "knowledge-core", "ACTION_REQUIRED", "WARNING", "知识处理部分完成", operation_status="PARTIAL", work_item_action="knowledge.ingestion.repair"),
    _event("knowledge.ingestion.failed", "knowledge-core", "RESULT", "WARNING", "知识处理失败", operation_status="FAILED", work_item_action="knowledge.ingestion.repair", resolve_work_item=True),
    _event("knowledge.collection.purge_completed", "knowledge-core", "RESULT", "INFO", "知识库删除完成", operation_status="SUCCEEDED"),
    _event("knowledge.collection.purge_failed", "knowledge-core", "RESULT", "WARNING", "知识库删除失败", operation_status="FAILED"),
    _event("data_query.schema.selection_required", "data-query", "ACTION_REQUIRED", "WARNING", "请选择数据采集范围", operation_status="WAITING_USER", work_item_action="data_query.schema.select"),
    _event("data_query.schema.capture_completed", "data-query", "RESULT", "INFO", "数据库结构采集完成", operation_status="SUCCEEDED", work_item_action="data_query.schema.select", resolve_work_item=True),
    _event("data_query.schema.capture_failed", "data-query", "RESULT", "WARNING", "数据库结构采集失败", operation_status="FAILED"),
    _event("data_query.semantic_model.generation_completed", "data-query", "RESULT", "INFO", "语义模型生成完成", operation_status="SUCCEEDED"),
    _event("data_query.semantic_model.generation_failed", "data-query", "RESULT", "WARNING", "语义模型生成失败", operation_status="FAILED"),
    _event("data_query.validation.completed", "data-query", "RESULT", "INFO", "语义模型验证完成", operation_status="SUCCEEDED"),
    _event("data_query.validation.failed", "data-query", "RESULT", "WARNING", "语义模型验证失败", operation_status="FAILED"),
    _event("data_query.run.completed", "data-query", "RESULT", "INFO", "数据查询完成", operation_status="SUCCEEDED"),
    _event("data_query.run.failed", "data-query", "RESULT", "WARNING", "数据查询失败", operation_status="FAILED"),
    _event("model.catalog.archive_blocked", "model-serving", "ACTION_REQUIRED", "WARNING", "模型归档被引用阻止"),
    _event("model.catalog.delete_blocked", "model-serving", "ACTION_REQUIRED", "WARNING", "模型删除被引用阻止"),
    _event("model.runtime.reload_failed", "model-serving", "SYSTEM", "CRITICAL", "模型运行时重新加载失败"),
)

EVENT_TYPES = MappingProxyType({item.event_type: item for item in _EVENTS})


def event_definition(event_type: str) -> NotificationEventDefinition:
    try:
        return EVENT_TYPES[event_type]
    except KeyError as exc:
        raise ValueError("NOTIFICATION_EVENT_TYPE_UNKNOWN") from exc
