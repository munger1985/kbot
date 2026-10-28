"""按冻结 Manifest 顺序调用受控数据库 Tool 的 Handler。"""

from __future__ import annotations

from dataclasses import replace
import asyncio
from datetime import datetime, timedelta, timezone

from aiops_agent.application.top_sql_window import TOP_SQL_REFERENCE_CONDITION
from aiops_agent.contracts.tool_execution import (
    DbaToolResult,
    ToolOutcome,
    DiagnosticSamplingSummary,
)
from aiops_agent.contracts.artifacts.database import DatabaseDiagnosticResult, EvidenceGap
from platform_core.contracts.aiops.investigation import DiagnosticSampling

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
        if invocation.get("execution_condition") == TOP_SQL_REFERENCE_CONDITION:
            windows = [
                dict(item.get("payload") or {}).get("result")
                for item in context.input_artifacts
                if item.get("schema_version") == "TOP_SQL_WINDOW_RESULT.v1"
            ]
            if len(windows) != 1 or not isinstance(windows[0], dict):
                raise ValueError("累计参考缺少唯一的窗口结果，不能判断执行条件")
            if windows[0].get("covered_intervals"):
                return DbaToolResult(
                    source_type="TOOL", source_id=tool["tool_id"],
                    source_version=tool["tool_version"], definition_hash=invocation["catalog_hash"],
                    output_schema="DBA_TOOL_RESULT.v1", measurement_semantics=invocation["measurement_semantics"],
                    presentation_kind=invocation["presentation_kind"], status="SUCCEEDED",
                    tool_outcomes=(ToolOutcome(
                        step_id=tool["step_id"], tool_id=tool["tool_id"],
                        tool_version=tool["tool_version"], status="SKIPPED",
                    ),),
                )
        database_snapshot = {
            **dict(execution["database"]),
            "catalog_hash": execution["diagnostic_catalog_hash"],
            "capability_snapshot_hash": execution[
                "capability_snapshot_hash"
            ],
            "tools": [tool],
        }
        if invocation.get("sampling") is not None:
            return await self._sample(context, invocation, tool, database_snapshot)
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

    async def _sample(self, context, invocation, tool, database_snapshot):
        """有限次真实查询；缺口即停，每次独立领取当前租约下的 Grant。"""
        sampling = DiagnosticSampling.model_validate(invocation["sampling"])
        started = datetime.now(timezone.utc)
        bounds = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in (
            context.lease_until, context.deadline_at,
        ) if value]
        if not context.lease_until or not bounds or any(item.tzinfo is None for item in bounds):
            raise ValueError("追加采样必须具有带时区的任务租约期限")
        cutoff = min(*bounds, started + timedelta(seconds=60)) - timedelta(seconds=1)
        limits = dict(tool["limits"])
        if int(limits["statement_timeout_seconds"]) > 5:
            raise ValueError("冻结的追加采样超出单次查询预算")
        limits["max_result_rows"] = min(200, int(limits["max_result_rows"]))
        limits["max_result_bytes"] = min(131072, int(limits["max_result_bytes"]))
        tool = {**tool, "limits": limits}
        snapshot = {**database_snapshot, "tools": [tool]}
        outcomes, completed = [], 0
        stop = "COMPLETE"

        def gap(reason, detail):
            return ToolOutcome(
                step_id=f"{tool['step_id']}.sample.{len(outcomes) + 1}",
                tool_id=tool["tool_id"], tool_version=tool["tool_version"], status="GAP",
                gap=EvidenceGap(code=reason, tool_id=tool["tool_id"], detail=detail, retryable=False),
            )

        required = sampling.count * int(limits["statement_timeout_seconds"]) + (sampling.count - 1) * sampling.interval_seconds
        if (cutoff - started).total_seconds() < required:
            stop = "BUDGET_EXHAUSTED"
            outcomes.append(gap("SAMPLING_BUDGET_EXHAUSTED", "本轮剩余期限不足以完成所请求的采样；没有补造历史窗口"))
        else:
            previous_capture, request_ids = None, set()
            try:
                async with asyncio.timeout((cutoff - started).total_seconds()):
                    for index in range(sampling.count):
                        if index:
                            await asyncio.sleep(sampling.interval_seconds)
                        if datetime.now(timezone.utc) >= cutoff:
                            raise TimeoutError
                        result = await self._database_handler.execute(
                            replace(context, task_key=f"diagnostic:{tool['tool_id']}", max_attempts=1,
                                plan_snapshot={**context.plan_snapshot, "database_diagnostics": snapshot}),
                            execution_key_suffix=f":sample:{index + 1}",
                        )
                        if datetime.now(timezone.utc) >= cutoff:
                            raise TimeoutError
                        observation = result.observation
                        if observation is not None:
                            captured = observation.captured_at
                            if captured.tzinfo is None or captured < started - timedelta(seconds=1) or captured > datetime.now(timezone.utc) + timedelta(seconds=1) or (previous_capture is not None and (captured - previous_capture).total_seconds() < sampling.interval_seconds) or observation.executor_request_id in request_ids:
                                stop = "INVALID_SAMPLE_TIME"
                                outcomes.append(gap("SAMPLING_TIME_NOT_COMPARABLE", "采样返回旧记录或非递增采集时间，未采纳该记录"))
                                break
                            previous_capture = captured
                            request_ids.add(observation.executor_request_id)
                        outcomes.append(ToolOutcome(
                            step_id=f"{tool['step_id']}.sample.{index + 1}",
                            tool_id=tool["tool_id"], tool_version=tool["tool_version"], status=result.status,
                            observation=observation, gap=result.gap,
                        ))
                        if result.status != "SUCCEEDED" or observation is None:
                            stop = "QUERY_GAP"
                            break
                        completed += 1
                        if observation.truncated:
                            stop = "RESULT_LIMIT"
                            outcomes.append(gap("SAMPLING_RESULT_TRUNCATED", "单次采样触及结果上限，停止追加；保留已采集的截断证据"))
                            break
            except TimeoutError:
                stop = "BUDGET_EXHAUSTED"
                outcomes.append(gap("SAMPLING_BUDGET_EXHAUSTED", "追加采样触及本轮期限；保留已完成时点，不外推请求窗口"))
        return DbaToolResult(
            source_type="TOOL", source_id=tool["tool_id"], source_version=tool["tool_version"],
            definition_hash=invocation["catalog_hash"], output_schema="DBA_TOOL_RESULT.v1",
            measurement_semantics=invocation["measurement_semantics"], presentation_kind=invocation["presentation_kind"],
            status="SUCCEEDED" if stop == "COMPLETE" else "PARTIAL" if completed else "FAILED",
            tool_outcomes=tuple(outcomes),
            sampling=DiagnosticSamplingSummary(requested=sampling, completed_count=completed,
                started_at=started, finished_at=datetime.now(timezone.utc), stop_reason=stop),
        )
