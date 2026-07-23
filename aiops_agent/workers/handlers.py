"""Worker 可解析的版本化 Handler Registry。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from aiops_agent.contracts.artifacts import (
    KernelReport,
    ObservationSet,
    ScopeResult,
)


@dataclass(frozen=True)
class TaskExecutionContext:
    run_id: str
    task_id: str
    task_key: str
    target_id: str
    agent_id: str
    trigger_type: str
    trace_id: str
    attempt: int
    deadline_at: str | None
    plan_snapshot: dict[str, Any]
    policy_snapshot: dict[str, Any]
    input_artifacts: tuple[dict[str, Any], ...]


class TaskHandler(Protocol):
    async def execute(self, context: TaskExecutionContext) -> BaseModel: ...


@dataclass(frozen=True)
class HandlerManifest:
    handler_id: str
    version: str
    output_schema_version: str
    idempotent: bool
    implementation: TaskHandler


class HandlerRegistry:
    def __init__(self, manifests: tuple[HandlerManifest, ...]):
        self._items = {
            (item.handler_id, item.version): item for item in manifests
        }
        if len(self._items) != len(manifests):
            raise ValueError("Handler ID 与版本不能重复")

    def resolve(self, handler_id: str, version: str) -> HandlerManifest:
        try:
            return self._items[(handler_id, version)]
        except KeyError as exc:
            raise LookupError(
                f"Handler 不存在：{handler_id}@{version}"
            ) from exc

    @property
    def manifests(self) -> tuple[HandlerManifest, ...]:
        return tuple(self._items.values())


class ScopeHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> ScopeResult:
        snapshots = context.plan_snapshot
        return ScopeResult(
            target_id=context.target_id,
            agent_id=context.agent_id,
            trigger_type=context.trigger_type,
            target_snapshot=dict(snapshots.get("target", {})),
            binding_snapshot=dict(snapshots.get("binding", {})),
            policy_snapshot=dict(context.policy_snapshot),
        )


class ObserveHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> ObservationSet:
        return ObservationSet(target_id=context.target_id)


class ReportHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> KernelReport:
        observation = next(
            (
                item["payload"]
                for item in context.input_artifacts
                if item["schema_version"] == "OBSERVATION_SET.v1"
            ),
            {},
        )
        return KernelReport(
            target_id=context.target_id,
            summary="确定性运行内核已完成 Scope、Observe 与 Report 闭环",
            observation_count=len(observation.get("observations", [])),
            gaps=tuple(observation.get("gaps", ())),
        )


def create_kernel_handler_registry() -> HandlerRegistry:
    return HandlerRegistry(
        (
            HandlerManifest(
                handler_id="kernel.scope",
                version="1",
                output_schema_version="SCOPE_RESULT.v1",
                idempotent=True,
                implementation=ScopeHandler(),
            ),
            HandlerManifest(
                handler_id="kernel.observe",
                version="1",
                output_schema_version="OBSERVATION_SET.v1",
                idempotent=True,
                implementation=ObserveHandler(),
            ),
            HandlerManifest(
                handler_id="kernel.report",
                version="1",
                output_schema_version="KERNEL_TEST_REPORT.v1",
                idempotent=True,
                implementation=ReportHandler(),
            ),
        )
    )


def create_runtime_handler_registry(
    *,
    monitor_provider_registry=None,
    secret_store=None,
) -> HandlerRegistry:
    """组合运行内核与步骤 5 Handler，版本必须精确匹配。"""
    kernel = create_kernel_handler_registry()
    manifests = list(kernel.manifests)
    if monitor_provider_registry is not None and secret_store is not None:
        from .monitoring_handlers import (
            MonitorObserveHandler,
            MonitorReportHandler,
            MonitorScopeHandler,
        )

        manifests.extend(
            (
                HandlerManifest(
                    handler_id="monitor.scope",
                    version="1",
                    output_schema_version="MONITOR_SCOPE_RESULT.v1",
                    idempotent=True,
                    implementation=MonitorScopeHandler(),
                ),
                HandlerManifest(
                    handler_id="monitor.observe",
                    version="1",
                    output_schema_version="OBSERVATION_SET.v1",
                    idempotent=True,
                    implementation=MonitorObserveHandler(
                        provider_registry=monitor_provider_registry,
                        secret_store=secret_store,
                    ),
                ),
                HandlerManifest(
                    handler_id="monitor.report",
                    version="1",
                    output_schema_version="OBSERVE_REPORT.v1",
                    idempotent=True,
                    implementation=MonitorReportHandler(),
                ),
            )
        )
    return HandlerRegistry(tuple(manifests))
