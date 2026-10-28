"""固定工具和动态查询共用的有界追加采样，不补造过去窗口。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone

from aiops_agent.contracts.artifacts.database import EvidenceGap
from aiops_agent.contracts.tool_execution import DiagnosticSamplingSummary, ToolOutcome
from platform_core.contracts.aiops.investigation import DiagnosticSampling
from .handlers import TaskExecutionContext


async def run_diagnostic_samples(
    *,
    context: TaskExecutionContext,
    sampling: DiagnosticSampling | Mapping,
    tool_id: str,
    tool_version: str,
    step_id: str,
    limits: Mapping[str, int],
    execute_sample: Callable[[int], Awaitable[ToolOutcome]],
) -> tuple[tuple[ToolOutcome, ...], DiagnosticSamplingSummary]:
    """逐次授权取证；部分结果保留，任何缺口停止且不重跑整组。"""
    sampling = DiagnosticSampling.model_validate(sampling)
    started = datetime.now(timezone.utc)
    bounds = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in (
        context.lease_until, context.deadline_at,
    ) if value]
    if not context.lease_until or any(item.tzinfo is None for item in bounds):
        raise ValueError("追加采样必须具有带时区的任务租约期限")
    cutoff = min(*bounds, started + timedelta(seconds=60)) - timedelta(seconds=1)
    if not 1 <= int(limits["statement_timeout_seconds"]) <= 5:
        raise ValueError("冻结的追加采样超出单次查询预算")
    if not 1 <= int(limits["max_result_rows"]) <= 200 or not 1 <= int(limits["max_result_bytes"]) <= 131072:
        raise ValueError("冻结的追加采样超出单次结果预算")
    outcomes, completed = [], 0
    stop = "COMPLETE"

    def gap(code, detail):
        return ToolOutcome(
            step_id=f"{step_id}.sample.{len(outcomes) + 1}",
            tool_id=tool_id, tool_version=tool_version, status="GAP",
            gap=EvidenceGap(code=code, tool_id=tool_id, detail=detail, retryable=False),
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
                    outcome = await execute_sample(index + 1)
                    if datetime.now(timezone.utc) >= cutoff:
                        raise TimeoutError
                    observation = outcome.observation
                    if observation is not None:
                        if (str(observation.target_id) != str(context.target_id)
                            or observation.tool_id != tool_id or observation.tool_version != tool_version):
                            stop = "QUERY_GAP"
                            outcomes.append(gap("SAMPLING_IDENTITY_MISMATCH", "采样结果与冻结的目标或工具身份不一致，未采纳该记录"))
                            break
                        captured = observation.captured_at
                        if (captured.tzinfo is None or captured < started - timedelta(seconds=1)
                            or captured > datetime.now(timezone.utc) + timedelta(seconds=1)
                            or (previous_capture is not None and (captured - previous_capture).total_seconds() < sampling.interval_seconds)
                            or observation.executor_request_id in request_ids):
                            stop = "INVALID_SAMPLE_TIME"
                            outcomes.append(gap("SAMPLING_TIME_NOT_COMPARABLE", "采样返回旧记录或非递增采集时间，未采纳该记录"))
                            break
                        previous_capture = captured
                        request_ids.add(observation.executor_request_id)
                    outcomes.append(outcome.model_copy(update={"step_id": f"{step_id}.sample.{index + 1}"}))
                    if outcome.status != "SUCCEEDED" or observation is None:
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
    return tuple(outcomes), DiagnosticSamplingSummary(
        requested=sampling, completed_count=completed, started_at=started,
        finished_at=datetime.now(timezone.utc), stop_reason=stop,
    )
