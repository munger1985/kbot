"""执行器只持有控制面租约；超时或清理不确定时不提前归还配额。"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from loguru import logger

from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClientError
from aiops_agent.executor.drivers import DiagnosticDriverError
from aiops_agent.executor.execution_fence import readonly_execution_fence
from aiops_agent.executor.readonly_lifecycle import collect_termination, observe_termination, report_termination
from aiops_agent.resource_admission import RETRYABLE_RESOURCE_CODES, wait_for_resource
from platform_core.contracts.aiops.executor import ResourceLeaseReceipt


@asynccontextmanager
async def readonly_credentials(control_plane, *, token, grant):
    try:
        issued = await wait_for_resource(
            lambda: control_plane.issue_credential(token, trace_id=grant.trace_id),
            retry_errors=(AIOpsExecutionClientError,),
        )
    except AIOpsExecutionClientError as exc:
        raise DiagnosticDriverError(exc.code, retryable=exc.code in RETRYABLE_RESOURCE_CODES) from exc
    receipt = issued.resource_lease
    if not isinstance(receipt, ResourceLeaseReceipt) or receipt.lease_id != grant.grant_id or receipt.execution_deadline_at > grant.expires_at:
        raise DiagnosticDriverError("RESOURCE_LEASE_INVALID")
    remaining = (receipt.execution_deadline_at - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED")
    with collect_termination(grant.db_type) as collector:
        try:
            async with asyncio.timeout(remaining), readonly_execution_fence(
                control_plane, token=token, grant=grant, receipt=receipt,
            ):
                yield issued
                if datetime.now(timezone.utc) >= receipt.execution_deadline_at:
                    raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED")
        except BaseException as exc:
            if collector["observation"] is None:
                observe_termination(grant.db_type, "TERMINATION_UNKNOWN", "NONE")
            if not isinstance(exc, TimeoutError):
                raise
            raise DiagnosticDriverError("EXECUTION_DEADLINE_EXCEEDED") from exc
        else:
            # 正常退出意味着 Driver 已返回并完成其清理；异常/取消保留租约到清理宽限结束。
            try:
                async with asyncio.timeout(3):
                    await control_plane.release_resource(receipt, trace_id=grant.trace_id)
            except Exception:
                logger.warning("只读取证租约未确认释放；保留控制面清理宽限")
        finally:
            if collector["observation"] is None and asyncio.current_task().cancelling():
                observe_termination(grant.db_type, "TERMINATION_UNKNOWN", "NONE")
            await report_termination(control_plane, receipt=receipt, collector=collector, trace_id=grant.trace_id)
