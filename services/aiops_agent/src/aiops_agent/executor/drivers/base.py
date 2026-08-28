"""只读诊断 Driver Port。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from aiops_agent.diagnostics.registry import ResolvedDiagnosticTool
from aiops_agent.actions import RenderedAction
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticLimits,
)


class DiagnosticDriverError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class DriverQueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool
    db_version: str


class ReadonlyDatabaseDriver(Protocol):
    db_type: str

    async def execute(
        self,
        *,
        profile: DiagnosticConnectionProfile,
        secret: ResolvedSecret,
        tool: ResolvedDiagnosticTool,
        parameters: dict[str, Any],
        limits: DiagnosticLimits,
        trace_id: str,
    ) -> DriverQueryResult: ...

    async def execute_dynamic(
        self,
        *,
        profile: DiagnosticConnectionProfile,
        secret: ResolvedSecret,
        sql: str,
        parameters: dict[str, Any],
        limits: DiagnosticLimits,
        trace_id: str,
    ) -> DriverQueryResult: ...


class MutationDriverError(RuntimeError):
    def __init__(self, code: str, *, outcome_unknown: bool = False):
        super().__init__(code)
        self.code = code
        self.outcome_unknown = outcome_unknown


@dataclass(frozen=True)
class MutationDriverResult:
    bounded_result: dict[str, Any]


class MutationDatabaseDriver(Protocol):
    db_type: str

    async def execute_action(
        self,
        *,
        profile: DiagnosticConnectionProfile,
        secret: ResolvedSecret,
        action: RenderedAction,
        trace_id: str,
    ) -> MutationDriverResult: ...
