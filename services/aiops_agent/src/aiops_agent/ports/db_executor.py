"""Worker 调用隔离 DB Executor 的 Port。"""

from typing import Protocol

from platform_core.contracts.aiops.executor import (
    ReadDiagnosticRequest,
    ReadDiagnosticResult,
)


class DatabaseExecutorClientPort(Protocol):
    async def execute_diagnostic(
        self, request: ReadDiagnosticRequest, *, trace_id: str
    ) -> ReadDiagnosticResult: ...
