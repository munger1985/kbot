"""估算计划专用执行器；Worker不接触连接材料、原SQL、绑定值或原始计划。"""

import asyncio
from datetime import UTC, datetime
import re

from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClientError
from aiops_agent.domain.evidence.sql_plan_replay import SqlPlanReplayRejected
from aiops_agent.executor.diagnostic_deadline import diagnostic_execution_window
from aiops_agent.executor.drivers.base import DiagnosticDriverError
from aiops_agent.executor.resource_admission import readonly_resource
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureReport, SqlPlanCaptureResult


class SqlPlanCaptureExecutor:
    def __init__(self, *, grant_codec, control_plane, driver, readonly_semaphore):
        self._codec = grant_codec
        self._control = control_plane
        self._driver = driver
        self._semaphore = readonly_semaphore

    @staticmethod
    def _result(grant, *, nodes=(), identity=None, verification=None, error=None):
        return SqlPlanCaptureResult(capture_run_id=grant.run_id, request_id=grant.request_id,
            database_type=grant.db_type, captured_at=datetime.now(UTC), status="FAILED" if error else "SUCCEEDED",
            error_code=error, nodes=nodes, physical_identity=identity or {},
            native_verification=verification,
            oracle_parameter_peeking="NOT_PROVEN" if grant.db_type == "ORACLE" else "NOT_APPLICABLE")

    @staticmethod
    def _error(exc):
        if isinstance(exc, (DiagnosticDriverError, SqlPlanReplayRejected, AIOpsExecutionClientError)):
            code = exc.code
            if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", code):
                return code
        return "PLAN_CAPTURE_EXECUTION_FAILED"

    async def execute(self, command):
        # 签名/类型错误由专用HTTP入口处理，不能降级成可信的数据库结果。
        grant = self._codec.verify_plan_capture(command.grant)
        try:
            async with diagnostic_execution_window(self._semaphore, expires_at=grant.expires_at):
                exchanged = None
                try:
                    async with readonly_resource(self._control, token=command.grant, grant=grant,
                        acquire=lambda: self._control.exchange_plan_capture(command.grant, trace_id=grant.trace_id)) as exchanged:
                        if exchanged.database_type != grant.db_type:
                            raise DiagnosticDriverError("PLAN_CAPTURE_ENGINE_MISMATCH")
                        nodes, identity, verification = await self._driver.capture(exchanged)
                        result = self._result(grant, nodes=nodes, identity=identity, verification=verification)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    result = self._result(grant, error=self._error(exc))
                if exchanged is not None:
                    # 仅在Driver清理和最后围栏复核收敛后回传；报告不再授予执行权。
                    reported = await self._control.report_plan_capture(
                        SqlPlanCaptureReport(result=result, resource_lease=exchanged.resource_lease), trace_id=grant.trace_id)
                    if reported != result:
                        raise DiagnosticDriverError("PLAN_CAPTURE_REPORT_UNCONFIRMED")
                return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 回传丢失/超期不能当作成功；不重试兑换，也不重复采集原查询计划。
            return self._result(grant, error=self._error(exc))
