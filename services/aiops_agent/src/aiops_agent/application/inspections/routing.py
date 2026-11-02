"""计划巡检的确定性执行路由，不把固定取证任务交给自然语言意图分类。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from aiops_agent.application.errors import validation_failed
from platform_core.contracts.aiops import (
    AIOpsCapability,
    ScopeType,
    TimeRangeV1,
    TurnPlanStepV1,
    TurnPlanV1,
)


def is_scheduled_inspection(context: dict[str, Any]) -> bool:
    """仅识别内部调度执行合同，不根据问题中的巡检关键词推断。"""
    execution = dict(context.get("execution_context") or {})
    return (
        execution.get("trigger_type") == "SCHEDULE"
        and execution.get("workflow_kind") == "INSPECTION"
    )


@dataclass(frozen=True)
class ScheduledInspectionPlanReceipt:
    """冻结巡检执行来源，避免将确定性调度伪装成模型或功能入口决策。"""

    inspection_fire_id: str
    target_id: str
    template_id: str
    template_hash: str

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "schema_version": "aiops.scheduled-inspection-plan-receipt.v1",
            "purpose": "aiops_turn_router",
            "route": "SCHEDULED_INSPECTION",
            "inspection_fire_id": self.inspection_fire_id,
            "target_id": self.target_id,
            "template_id": self.template_id,
            "template_hash": self.template_hash,
        }


def scheduled_inspection_plan(
    context: dict[str, Any],
) -> tuple[TurnPlanV1, ScheduledInspectionPlanReceipt]:
    """固定巡检必须执行授权的主动诊断，仍由诊断内核完成取证和报告。"""
    if AIOpsCapability.ACTIVE_DIAGNOSIS.value not in {
        str(item) for item in context["capabilities"]
    }:
        raise validation_failed("当前 Agent 版本未授权执行巡检主动诊断")
    execution = dict(context["execution_context"])
    inspection = dict(execution.get("inspection") or {})
    steps = inspection.get("evidence_steps") or ()
    if not steps or not all(
        isinstance(step, dict) and step.get("tool_id") for step in steps
    ):
        raise validation_failed("巡检缺少冻结的固定取证步骤")
    try:
        fire_id = str(UUID(str(execution.get("inspection_fire_id") or "")))
        target_id = str(UUID(str(context.get("focused_target_id") or "")))
        if not execution.get("observation_start") or not execution.get("observation_end"):
            raise ValueError("巡检缺少观测窗口")
        time_range = TimeRangeV1(
            start_at=execution["observation_start"],
            end_at=execution["observation_end"],
        )
    except ValueError as exc:
        raise validation_failed("巡检批次、目标或观测窗口无效") from exc
    plan = TurnPlanV1(
        turn_id=context["turn_id"],
        revision_no=context["plan_revision"],
        requested_capabilities=(AIOpsCapability.ACTIVE_DIAGNOSIS,),
        requires_active_diagnosis=True,
        time_range=time_range,
        steps=(TurnPlanStepV1(
            step_id="s1",
            capability=AIOpsCapability.ACTIVE_DIAGNOSIS,
            objective="按冻结巡检模板执行固定只读取证并生成巡检报告",
            scope_type=ScopeType.RESOURCE,
        ),),
        policy_snapshot_hash=context["policy_snapshot_hash"],
    )
    return plan, ScheduledInspectionPlanReceipt(
        inspection_fire_id=fire_id,
        target_id=target_id,
        template_id=str(inspection.get("template_id") or ""),
        template_hash=str(inspection.get("template_hash") or ""),
    )


def scheduled_inspection_systems(context: dict[str, Any]) -> list[dict[str, Any]]:
    """从当前授权资源图重新校验冻结目标，不能扩大到其他数据库或请求澄清。"""
    target_id = str(context["focused_target_id"])
    inspection = dict(context["execution_context"].get("inspection") or {})
    business_system_id = str(inspection.get("business_system_id") or "")
    if not business_system_id:
        raise validation_failed("巡检缺少冻结的业务系统归属")
    matches = [
        (system, resource)
        for system in context["systems"]
        if str(system["business_system_id"]) == business_system_id
        for resource in system["resources"]
        if str(resource.get("target_id")) == target_id
    ]
    if not matches:
        raise validation_failed("巡检目标已不属于 Agent 当前授权资源范围")
    # 同一数据库可以被多个系统引用，只使用计划所属系统的资源上下文。
    system, resource = matches[0]
    if not resource.get("target_ready"):
        raise validation_failed("巡检目标未启用只读取证，不能执行数据库巡检")
    return [{**system, "resources": [resource]}]
