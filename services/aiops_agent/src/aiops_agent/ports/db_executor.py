"""Worker 调用隔离 DB Executor 的 Port。"""

from typing import Protocol

from platform_core.contracts.aiops.executor import (
    DynamicReadDiagnosticRequest,
    ReadDiagnosticRequest,
    ReadDiagnosticResult,
)
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureExecute, SqlPlanCaptureResult


class DatabaseExecutorClientError(RuntimeError):
    """隔离执行器调用失败，保留可安全展示的 HTTP 错误语义。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class DatabaseExecutorClientPort(Protocol):
    async def execute_plan_capture(
        self, request: SqlPlanCaptureExecute, *, trace_id: str
    ) -> SqlPlanCaptureResult: ...

    async def execute_diagnostic(
        self, request: ReadDiagnosticRequest, *, trace_id: str
    ) -> ReadDiagnosticResult: ...

    async def execute_dynamic_diagnostic(
        self, request: DynamicReadDiagnosticRequest, *, trace_id: str
    ) -> ReadDiagnosticResult: ...
