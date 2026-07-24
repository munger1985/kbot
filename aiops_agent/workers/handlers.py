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
    actor_id: str = ""
    original_request: str = ""
    lease_token: str = ""
    lease_until: str = ""


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
    db_executor_client=None,
    diagnostic_grant_codec=None,
    diagnostic_grant_issuer: str | None = None,
    diagnostic_grant_audience: str | None = None,
    diagnostic_grant_ttl_seconds: int = 45,
    diagnosis_model_client=None,
    diagnosis_prompt_registry=None,
    diagnostic_registry=None,
    knowledge_core_client=None,
    diagnosis_caller_service: str | None = None,
    action_registry=None,
    action_execution_enabled: bool = False,
) -> HandlerRegistry:
    """组合运行内核及各阶段 Handler，版本必须精确匹配。"""
    kernel = create_kernel_handler_registry()
    manifests = list(kernel.manifests)
    database_diagnostic_handler = None
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
    if db_executor_client is not None and diagnostic_grant_codec is not None:
        from .database_handlers import (
            DatabaseAggregateHandler,
            DatabaseDiagnosticHandler,
            DatabaseReportHandler,
            DatabaseScopeHandler,
        )

        database_diagnostic_handler = DatabaseDiagnosticHandler(
            executor_client=db_executor_client,
            grant_codec=diagnostic_grant_codec,
            grant_issuer=diagnostic_grant_issuer or "",
            grant_audience=diagnostic_grant_audience or "",
            grant_ttl_seconds=diagnostic_grant_ttl_seconds,
        )
        manifests.extend(
            (
                HandlerManifest(
                    handler_id="database.scope",
                    version="1",
                    output_schema_version="DATABASE_SCOPE_RESULT.v1",
                    idempotent=True,
                    implementation=DatabaseScopeHandler(),
                ),
                HandlerManifest(
                    handler_id="database.diagnostic",
                    version="1",
                    output_schema_version="DATABASE_DIAGNOSTIC_RESULT.v1",
                    idempotent=True,
                    implementation=database_diagnostic_handler,
                ),
                HandlerManifest(
                    handler_id="database.aggregate",
                    version="1",
                    output_schema_version=(
                        "DATABASE_OBSERVATION_AGGREGATE.v1"
                    ),
                    idempotent=True,
                    implementation=DatabaseAggregateHandler(),
                ),
                HandlerManifest(
                    handler_id="database.report",
                    version="1",
                    output_schema_version="DB_DIAGNOSTIC_REPORT.v1",
                    idempotent=True,
                    implementation=DatabaseReportHandler(),
                ),
            )
        )
    if (
        diagnosis_model_client is not None
        and diagnosis_prompt_registry is not None
        and diagnostic_registry is not None
        and database_diagnostic_handler is not None
        and knowledge_core_client is not None
    ):
        from .diagnosis_handlers import (
            BuildEvidenceIndexHandler,
            DiagnosisEvidenceCollectHandler,
            DiagnosisReportHandler,
            DiagnosisRoundAssessmentHandler,
            DiagnosisRoundDraftHandler,
            DiagnosisScopeHandler,
            EvidenceRequestValidatorHandler,
            GroundingVerificationHandler,
            InteractiveDiagnosisHandler,
            KnowledgeCitationHandler,
            RootCauseAssessmentHandler,
            SolutionDraftHandler,
        )

        manifests.extend(
            (
                HandlerManifest(
                    handler_id="diagnosis.scope",
                    version="1",
                    output_schema_version="DIAGNOSIS_SCOPE.v1",
                    idempotent=True,
                    implementation=DiagnosisScopeHandler(),
                ),
                HandlerManifest(
                    handler_id="diagnosis.evidence-index",
                    version="1",
                    output_schema_version="EVIDENCE_INDEX.v1",
                    idempotent=True,
                    implementation=BuildEvidenceIndexHandler(),
                ),
                HandlerManifest(
                    handler_id="diagnosis.knowledge-citation",
                    version="1",
                    output_schema_version="KNOWLEDGE_CITATION_PACK.v1",
                    idempotent=True,
                    implementation=KnowledgeCitationHandler(
                        knowledge_client=knowledge_core_client,
                        caller_service=diagnosis_caller_service or "",
                    ),
                ),
                HandlerManifest(
                    handler_id="diagnosis.round-draft",
                    version="1",
                    output_schema_version="DIAGNOSIS_ROUND_DRAFT.v1",
                    idempotent=True,
                    implementation=DiagnosisRoundDraftHandler(
                        model_client=diagnosis_model_client,
                        prompts=diagnosis_prompt_registry,
                    ),
                ),
                HandlerManifest(
                    handler_id="diagnosis.request-validator",
                    version="1",
                    output_schema_version="VALIDATED_EVIDENCE_PLAN.v1",
                    idempotent=True,
                    implementation=EvidenceRequestValidatorHandler(
                        registry=diagnostic_registry
                    ),
                ),
                HandlerManifest(
                    handler_id="diagnosis.evidence-collect",
                    version="1",
                    output_schema_version=(
                        "DIAGNOSIS_EVIDENCE_COLLECTION.v1"
                    ),
                    idempotent=True,
                    implementation=DiagnosisEvidenceCollectHandler(
                        database_handler=database_diagnostic_handler
                    ),
                ),
                HandlerManifest(
                    handler_id="diagnosis.round-assess",
                    version="1",
                    output_schema_version=(
                        "DIAGNOSIS_ROUND_ASSESSMENT.v1"
                    ),
                    idempotent=True,
                    implementation=DiagnosisRoundAssessmentHandler(
                        model_client=diagnosis_model_client,
                        prompts=diagnosis_prompt_registry,
                    ),
                ),
                HandlerManifest(
                    handler_id="diagnosis.interactive",
                    version="1",
                    output_schema_version="HITL_OUTCOME.v1",
                    idempotent=True,
                    implementation=InteractiveDiagnosisHandler(
                        registry=diagnostic_registry
                    ),
                ),
                HandlerManifest(
                    handler_id="diagnosis.root-cause",
                    version="1",
                    output_schema_version="ROOT_CAUSE_ASSESSMENT.v1",
                    idempotent=True,
                    implementation=RootCauseAssessmentHandler(),
                ),
                HandlerManifest(
                    handler_id="diagnosis.grounding",
                    version="1",
                    output_schema_version="GROUNDING_VERIFICATION.v1",
                    idempotent=True,
                    implementation=GroundingVerificationHandler(
                        model_client=diagnosis_model_client,
                        prompts=diagnosis_prompt_registry,
                    ),
                ),
                HandlerManifest(
                    handler_id="diagnosis.solution",
                    version="1",
                    output_schema_version="SOLUTION_DRAFT.v1",
                    idempotent=True,
                    implementation=SolutionDraftHandler(),
                ),
                HandlerManifest(
                    handler_id="diagnosis.report",
                    version="1",
                    output_schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                    idempotent=True,
                    implementation=DiagnosisReportHandler(),
                ),
            )
        )
        if action_registry is not None:
            from .change_handlers import (
                ActionPlanHandler,
                ProposalSnapshotHandler,
            )

            manifests.extend(
                (
                    HandlerManifest(
                        handler_id="change.action-plan",
                        version="1",
                        output_schema_version="ACTION_PLAN.v1",
                        idempotent=True,
                        implementation=ActionPlanHandler(
                            registry=action_registry,
                            execution_enabled=action_execution_enabled,
                        ),
                    ),
                    HandlerManifest(
                        handler_id="change.proposal",
                        version="1",
                        output_schema_version="PROPOSAL_OUTCOME.v1",
                        idempotent=True,
                        implementation=ProposalSnapshotHandler(),
                    ),
                )
            )
    return HandlerRegistry(tuple(manifests))
