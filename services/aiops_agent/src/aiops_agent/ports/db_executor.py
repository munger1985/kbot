"""Worker 调用隔离 DB Executor 的 Port。"""

from typing import Protocol

from platform_core.contracts.aiops.executor import (
    DynamicReadDiagnosticRequest,
    ReadDiagnosticRequest,
    ReadDiagnosticResult,
)


class DatabaseExecutorClientPort(Protocol):
    async def execute_diagnostic(
        self, request: ReadDiagnosticRequest, *, trace_id: str
    ) -> ReadDiagnosticResult: ...

    async def execute_dynamic_diagnostic(
        self, request: DynamicReadDiagnosticRequest, *, trace_id: str
    ) -> ReadDiagnosticResult: ...
