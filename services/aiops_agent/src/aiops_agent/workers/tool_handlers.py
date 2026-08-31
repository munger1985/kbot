"""按冻结 Manifest 顺序调用受控数据库 Tool 的 Handler。"""

from __future__ import annotations

from dataclasses import replace

from aiops_agent.contracts.tool_execution import (
    DbaToolResult,
    ToolOutcome,
)
from aiops_agent.contracts.artifacts.database import DatabaseDiagnosticResult

from .handlers import TaskExecutionContext


class DbaPlaybookInvocationHandler:
    """只消费规划时冻结的 Tool 快照，禁止运行时重新选 Tool。"""

    def __init__(self, *, database_handler) -> None:
        self._database_handler = database_handler

    async def execute(self, context: TaskExecutionContext) -> DbaToolResult:
        execution = context.plan_snapshot["investigation_execution"]
        invocation = dict(execution["invocations"][context.task_key])
        database = dict(execution["database"])
        database_snapshot = {
            **database,
            "catalog_hash": execution["diagnostic_catalog_hash"],
            "capability_snapshot_hash": execution[
                "capability_snapshot_hash"
            ],
            "tools": list(invocation["tools"]),
        }
        outcomes: list[ToolOutcome] = []
        attempted_steps: set[str] = set()
        input_artifacts = list(context.input_artifacts)
        for tool in invocation["tools"]:
            dependencies = set(tool.get("depends_on", ()))
            if not dependencies <= attempted_steps:
                outcomes.append(
                    ToolOutcome(
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
                ToolOutcome(
                    step_id=tool["step_id"],
                    tool_id=tool["tool_id"],
                    tool_version=tool["tool_version"],
                    status=result.status,
                    observation=result.observation,
                    gap=result.gap,
                )
            )
            attempted_steps.add(tool["step_id"])
            if result.status == "SUCCEEDED":
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
        return DbaToolResult(
            source_type="PLAYBOOK",
            source_id=invocation["playbook_id"],
            source_version=invocation["playbook_version"],
            definition_hash=invocation["manifest_hash"],
            output_schema=invocation["output_schema"],
            measurement_semantics=invocation["measurement_semantics"],
            presentation_kind=invocation["presentation_kind"],
            status=status,
            tool_outcomes=tuple(outcomes),
        )


class DbaDiagnosticToolHandler:
    """直接执行调查计划中的原子数据库Tool，不要求Playbook父对象。"""

    def __init__(self, *, database_handler) -> None:
        self._database_handler = database_handler

    async def execute(self, context: TaskExecutionContext) -> DbaToolResult:
        execution = context.plan_snapshot["investigation_execution"]
        invocation = dict(execution["direct_invocations"][context.task_key])
        tool = dict(invocation["tool"])
        database_snapshot = {
            **dict(execution["database"]),
            "catalog_hash": execution["diagnostic_catalog_hash"],
            "capability_snapshot_hash": execution[
                "capability_snapshot_hash"
            ],
            "tools": [tool],
        }
        result = await self._database_handler.execute(
            replace(
                context,
                task_key=f"diagnostic:{tool['tool_id']}",
                plan_snapshot={
                    **context.plan_snapshot,
                    "database_diagnostics": database_snapshot,
                },
            )
        )
        return DbaToolResult(
            source_type="TOOL",
            source_id=tool["tool_id"],
            source_version=tool["tool_version"],
            definition_hash=invocation["catalog_hash"],
            output_schema="DBA_TOOL_RESULT.v1",
            measurement_semantics=invocation["measurement_semantics"],
            presentation_kind=invocation["presentation_kind"],
            status=("SUCCEEDED" if result.status == "SUCCEEDED" else "FAILED"),
            tool_outcomes=(
                ToolOutcome(
                    step_id=tool["step_id"],
                    tool_id=tool["tool_id"],
                    tool_version=tool["tool_version"],
                    status=result.status,
                    observation=result.observation,
                    gap=result.gap,
                ),
            ),
        )
