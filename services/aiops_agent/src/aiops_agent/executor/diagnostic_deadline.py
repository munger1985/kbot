"""诊断的排队、凭据和执行共享签名 Grant 的绝对期限。"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from aiops_agent.executor.drivers import DiagnosticDriverError


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
                yield
                # 不信任第三方协程一定会传播取消；过期后返回的值也必须丢弃。
                if datetime.now(timezone.utc) >= expires_at:
                    raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED")
    except TimeoutError as exc:
        raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED") from exc
