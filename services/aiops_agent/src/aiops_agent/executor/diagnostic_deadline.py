"""诊断的排队、凭据和执行共享签名 Grant 的绝对期限。"""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from aiops_agent.executor.drivers import DiagnosticDriverError


_active_deadline = ContextVar("aiops_diagnostic_deadline", default=None)


def identity_probe_budget():
    """可选身份探测预留连接清理时间，不挤掉已经完成的主查询。"""
    deadline = _active_deadline.get()
    if deadline is None:
        return 1.0
    return max(0.0, min(1.0, (deadline - datetime.now(timezone.utc)).total_seconds() - 2.0))


@asynccontextmanager
async def diagnostic_execution_window(semaphore, *, expires_at):
    remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED")
    try:
        async with asyncio.timeout(remaining):
            async with semaphore:
                # 排队期间不提前签发/持有目标库凭据；到期的等待者不进入 Driver。
                if datetime.now(timezone.utc) >= expires_at:
                    raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED")
                token = _active_deadline.set(expires_at)
                try:
                    yield
                    # 不信任第三方协程一定会传播取消；过期后返回的值也必须丢弃。
                    if datetime.now(timezone.utc) >= expires_at:
                        raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED")
                finally:
                    _active_deadline.reset(token)
    except TimeoutError as exc:
        raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED") from exc
