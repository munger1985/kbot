"""按冻结 Manifest 顺序调用受控数据库 Tool 的 Skill Handler。"""

from __future__ import annotations

from dataclasses import replace

from aiops_agent.contracts.skill_execution import (
    DbaSkillResult,
    SkillToolOutcome,
)
from aiops_agent.contracts.artifacts.database import DatabaseDiagnosticResult

from .handlers import TaskExecutionContext


class DbaSkillInvocationHandler:
    """只消费规划时冻结的 Tool 快照，禁止运行时重新选 Tool。"""

    def __init__(self, *, database_handler) -> None:
        self._database_handler = database_handler

    async def execute(self, context: TaskExecutionContext) -> DbaSkillResult:
        execution = context.plan_snapshot["skill_execution"]
        invocation = dict(execution["invocations"][context.task_key])
        database = dict(execution["database"])
        database_snapshot = {
            **database,
            "catalog_hash": execution["diagnostic_catalog_hash"],
            "capability_snapshot_hash": execution[
                "capability_snapshot_hash"
            ],
            "automatic_access_enabled": True,
            "initial_gaps": [],
            "tools": list(invocation["tools"]),
        }
        outcomes: list[SkillToolOutcome] = []
        completed_steps: set[str] = set()
        input_artifacts = list(context.input_artifacts)
        for tool in invocation["tools"]:
            dependencies = set(tool.get("depends_on", ()))
            if not dependencies <= completed_steps:
                outcomes.append(
                    SkillToolOutcome(
                        step_id=tool["step_id"],
                        tool_id=tool["tool_id"],
                        tool_version=tool["tool_version"],
                        status="SKIPPED",
                    )
                )
                continue
            tool_context = replace(
                context,
                task_key=f"diagnostic:{tool['tool_id']}",
                plan_snapshot={
                    **context.plan_snapshot,
                    "database_diagnostics": database_snapshot,
                },
                input_artifacts=tuple(input_artifacts),
            )
            result: DatabaseDiagnosticResult = (
                await self._database_handler.execute(tool_context)
            )
            outcomes.append(
                SkillToolOutcome(
                    step_id=tool["step_id"],
                    tool_id=tool["tool_id"],
                    tool_version=tool["tool_version"],
                    status=result.status,
                    observation=result.observation,
                    gap=result.gap,
                )
            )
            if result.status == "SUCCEEDED":
                completed_steps.add(tool["step_id"])
                input_artifacts.append(
                    {
                        "schema_version": result.schema_version,
                        "payload": result.model_dump(mode="json"),
                    }
                )
        succeeded = sum(
            item.status == "SUCCEEDED" for item in outcomes
        )
        status = (
            "SUCCEEDED"
            if succeeded == len(outcomes)
            else "PARTIAL"
            if succeeded
            else "FAILED"
        )
        return DbaSkillResult(
            skill_id=invocation["skill_id"],
            skill_version=invocation["skill_version"],
            manifest_hash=invocation["manifest_hash"],
            output_schema=invocation["output_schema"],
            measurement_semantics=invocation["measurement_semantics"],
            status=status,
            tool_outcomes=tuple(outcomes),
        )
