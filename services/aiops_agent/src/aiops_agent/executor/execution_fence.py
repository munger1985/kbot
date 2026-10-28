"""Executor 持续读取持久化授权，取消本进程正在执行的精确 Driver 协程。"""

import asyncio
from asyncio import sleep as pause_between_checks
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClientError
from aiops_agent.executor.drivers import DiagnosticDriverError
from platform_core.contracts.aiops.executor import ReadonlyExecutionFenceReceipt


FENCE_CHECK_INTERVAL_SECONDS = 1.0
FENCE_CHECK_TIMEOUT_SECONDS = 2.0


@asynccontextmanager
async def readonly_execution_fence(control_plane, *, token, grant, receipt):
    """执行前、执行中及结果接纳前复核；控制面失联有界失败关闭。"""
    deadline = receipt.execution_deadline_at

    async def check():
        nonlocal deadline
        try:
            async with asyncio.timeout(FENCE_CHECK_TIMEOUT_SECONDS):
                result = await control_plane.check_readonly_execution(token, receipt, trace_id=grant.trace_id)
        except (AIOpsExecutionClientError, TimeoutError) as exc:
            raise DiagnosticDriverError("EXECUTION_CONTROL_PLANE_UNAVAILABLE") from exc
        except Exception as exc:
            raise DiagnosticDriverError("EXECUTION_CONTROL_PLANE_UNAVAILABLE") from exc
        if not isinstance(result, ReadonlyExecutionFenceReceipt):
            raise DiagnosticDriverError("EXECUTION_FENCE_INVALID")
        if not result.allowed:
            raise DiagnosticDriverError(result.reason)
        if result.execution_deadline_at > receipt.execution_deadline_at:
            raise DiagnosticDriverError("EXECUTION_FENCE_INVALID")
        deadline = min(deadline, result.execution_deadline_at)
        if datetime.now(timezone.utc) >= deadline:
            raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED")

    await check()
    parent = asyncio.current_task()
    cancellation_marker = object()
    failure = None
    injected = False

    async def watch():
        nonlocal failure, injected
        try:
            while True:
                remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
                if remaining <= 0:
                    raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED")
                await pause_between_checks(min(FENCE_CHECK_INTERVAL_SECONDS, remaining))
                await check()
        except DiagnosticDriverError as exc:
            failure = exc
            # 外部取消/绝对期限优先；不把它们替换为监测器自己的错误。
            if not parent.done() and not parent.cancelling():
                injected = parent.cancel(cancellation_marker)

    watcher = asyncio.create_task(watch(), name="aiops-readonly-execution-fence")
    try:
        yield
        if parent.cancelling() and not injected:
            # 即使 Driver 吞掉调用方取消，也不能把它转换为正常成功。
            raise asyncio.CancelledError
        if failure is not None:
            raise failure
        await check()
    except asyncio.CancelledError as exc:
        if injected:
            count = parent.uncancel()
            injected = False
            if count == 0 and exc.args and exc.args[0] is cancellation_marker:
                raise failure from exc
        raise
    finally:
        watcher.cancel()
        # gather 只收拢子监测器；此处新的外部取消仍须向上抛出。
        await asyncio.gather(watcher, return_exceptions=True)
        # 第三方吞掉自有取消时也必须还原取消计数，且不准返回成功。
        if injected:
            parent.uncancel()
