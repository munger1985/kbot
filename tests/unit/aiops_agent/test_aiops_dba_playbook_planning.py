"""专业 DBA 调查计划、Playbook 与 Tool 执行框架测试。"""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.contracts.artifacts.database import (
    DatabaseDiagnosticResult,
    EvidenceGap,
)
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.application.investigation import (
    TurnPlanningService,
    TurnPlanningStageError,
)
from aiops_agent.application.investigation.projection import (
    safe_plan_projection,
)
from aiops_agent.application.investigation.discovery import build_playbook_plan
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.playbooks import PlaybookCatalogError, PlaybookRegistry
from aiops_agent.tools import (
    InvestigationCatalogChangedError,
    InvestigationTaskCompiler,
    ToolExecutionSnapshotBuilder,
    build_capability_snapshot,
)
from aiops_agent.workers.database_handlers import DatabaseDiagnosticHandler
from aiops_agent.workers.errors import RetryableTaskError
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.tool_handlers import DbaPlaybookInvocationHandler
from platform_core.contracts.aiops.conversation import (
    DbaIntent,
    MeasurementSemantics,
)
from platform_core.contracts.aiops.investigation import (
    InputMaterial,
    InvestigationAction,
    InvestigationPlan,
    InvestigationPlanningOutput,
    MaterialKind,
    TaskFrame,
    TaskObjective,
    TurnInputEnvelope,
)
from platform_core.contracts.aiops.playbooks import (
    DbaCapabilitySnapshot,
    DbaDomain,
    DbaPlaybookManifest,
    DbaPlaybookPlan,
    PlaybookLimits,
    PlaybookPlanItem,
    PlaybookToolStep,
    PresentationPreference,
    SourceCapabilitySnapshot,
)
from platform_core.contracts.aiops.types import DatabaseType
from platform_core.contracts.aiops.executor import ReadDiagnosticResult
from platform_core.identity import uuid7


def _manifest(
    *,
    playbook_id: str,
    intent: DbaIntent,
    domain: DbaDomain,
    tool_id: str = "db.instance.identity",
    required_target: tuple[str, ...] = ("DB_READONLY",),
    required_source: tuple[str, ...] = (),
) -> DbaPlaybookManifest:
    return DbaPlaybookManifest(
        playbook_id=playbook_id,
        version="1.0.0",
        database_types=(DatabaseType.ORACLE,),
        supported_intents=(intent,),
        domains=(domain,),
        required_source_capabilities=required_source,
        required_target_capabilities=required_target,
        required_privileges=("V_$INSTANCE", "V_$DATABASE"),
        input_schema=f"{playbook_id}.input.v1",
        limits=PlaybookLimits(max_rows=50, timeout_seconds=20),
        tool_dag=(PlaybookToolStep(step_id="collect", tool_id=tool_id),),
        output_schema=f"{playbook_id}.output.v1",
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
        presentation_kind=PresentationPreference.TABLE,
    )


def _playbook_plan(
    registry: PlaybookRegistry,
    manifests: tuple[DbaPlaybookManifest, ...],
    *,
    input_by_id: dict[str, dict] | None = None,
) -> DbaPlaybookPlan:
    inputs = input_by_id or {}
    return DbaPlaybookPlan(
        catalog_hash=registry.catalog_hash,
        items=tuple(
            PlaybookPlanItem(
                ordinal=ordinal,
                playbook_id=manifest.playbook_id,
                playbook_version=manifest.version,
                manifest_hash=registry.manifest_hash(
                    manifest.playbook_id, manifest.version
                ),
                reason="测试冻结Playbook",
                evidence_question="收集Playbook声明的诊断证据",
                measurement_semantics=manifest.measurement_semantics,
                input={
                    **manifest.defaults,
                    **inputs.get(manifest.playbook_id, {}),
                },
            )
            for ordinal, manifest in enumerate(manifests, start=1)
        ),
    )


def _capabilities(*, reachable: bool = True) -> DbaCapabilitySnapshot:
    return DbaCapabilitySnapshot(
        agent_id="agent-1",
        agent_version_id="agent-version-1",
        target_id="target-1",
        database_type=DatabaseType.ORACLE,
        database_version="19c",
        target_enabled=True,
        target_reachable=reachable,
        target_capabilities=("DB_READONLY",),
        privileges=("V_$INSTANCE", "V_$DATABASE"),
        source_snapshots=(
            SourceCapabilitySnapshot(
                source_id="prometheus-1",
                source_type="PROMETHEUS",
                enabled=True,
                reachable=True,
                capabilities=("PROMETHEUS_QUERY",),
            ),
        ),
    )


class _SessionBoundArtifact:
    """模拟 Session 关闭后访问 ORM 属性时抛出的脱离错误。"""

    def __init__(
        self,
        *,
        owner,
        artifact_id,
        ops_run_id,
        artifact_key,
        schema_version,
        payload_json,
    ) -> None:
        self._owner = owner
        self._values = {
            "artifact_id": artifact_id,
            "ops_run_id": ops_run_id,
            "artifact_key": artifact_key,
            "schema_version": schema_version,
            "payload_json": payload_json,
        }

    def __getattr__(self, name):
        if name not in self._values:
            raise AttributeError(name)
        if not self._owner._uow_active:
            raise RuntimeError("DetachedInstanceError")
        return self._values[name]


class _PlanningUow:
    def __init__(self) -> None:
        self._uow_active = False
        self.turn = SimpleNamespace(
            turn_id=uuid7(),
            conversation_id=uuid7(),
            domain_id=7,
            status="PLANNING",
            event_cursor=2,
            input_analysis_artifact_id=None,
            task_frame_artifact_id=None,
            current_plan_artifact_id=None,
            assessment_artifact_id=None,
            current_plan_revision=0,
            investigation_round=0,
            tool_call_count=0,
            no_progress_count=0,
        )
        self.version = SimpleNamespace(
            agent_version_id=uuid7(),
            policy_id=uuid7(),
        )
        self.policy = SimpleNamespace(
            policy_id=self.version.policy_id,
            rules_json={"readonly_database_enabled": True},
        )
        self.target = SimpleNamespace(
            target_id=uuid7(),
            display_name="订单生产库",
            db_type="ORACLE",
            version_code="19c",
            environment="PROD",
            db_role="PRIMARY",
            status="ENABLED",
            connectivity_status="CONNECTED",
            readonly_connection_enabled=True,
            controlled_change_enabled=False,
            diagnostic_credential_id=uuid7(),
            execution_credential_id=None,
            endpoint_json={
                "host": "db.internal", "port": 1521, "service": "PDB1"
            },
            domain_id=7,
            row_version=1,
            capabilities_json={
                "capabilities": ["DB_READONLY"],
                "privileges": ["V_$INSTANCE", "V_$DATABASE"],
            },
        )
        self.run = SimpleNamespace(
            ops_run_id=uuid7(),
            domain_id=7,
            agent_id=uuid7(),
            agent_version_id=self.version.agent_version_id,
            target_id=self.target.target_id,
            status="RUNNING",
            trace_id="trace-planning",
            deadline_at=None,
            plan_snapshot_json={},
        )
        self.source_artifact = SimpleNamespace(
            artifact_id=uuid7(),
            schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
            payload_json={"summary": "数据库不可用，Alert Log出现ORA-27157"},
            created_at=datetime.now(UTC),
            trust_level="MODEL_INFERENCE",
        )
        self.source_run = SimpleNamespace(
            ops_run_id=uuid7(),
            domain_id=7,
            final_artifact_id=self.source_artifact.artifact_id,
        )
        self.message = SimpleNamespace(
            sequence_no=1,
            message_type="USER_MESSAGE",
            payload_json={
                "text": "分析这段 ORA-27157 日志",
                "content": [
                    {
                        "content_type": "TEXT",
                        "text": "ORA-27157: OS post/wait facility removed",
                    }
                ],
            },
        )
        self.input_items = [
            SimpleNamespace(
                item_no=1,
                detected_kind=None,
                detection_confidence=None,
            )
        ]
        self.artifacts = []
        self.tasks = []
        self.invocations = []
        self.tool_invocations = []
        self.revisions = []
        self.evidence = []
        self.events = []
        self.commit_count = 0
        self.turns = SimpleNamespace(
            get_turn=self._get_turn,
            get_run_link=self._get_run_link,
            get_run_link_by_ops_run_id=self._get_run_link_by_ops_run_id,
            list_messages=self._list_messages,
            list_recent_conversation_messages=self._recent,
            add_playbook_invocation=self._add_invocation,
            add_tool_invocation=self._add_tool_invocation,
            add_investigation_revision=self._add_revision,
            list_input_items=self._list_input_items,
            get_evidence_by_artifact=self._get_evidence_by_artifact,
            add_evidence=self._add_evidence,
            add_event=self._add_event,
        )
        self.runs = SimpleNamespace(
            get_run=self._get_run,
            get_artifact=self._get_artifact,
            get_artifact_by_key=self._get_artifact_by_key,
            list_artifacts=self._list_artifacts,
            list_tasks=self._list_tasks,
            add_artifact=self._add_artifact,
            add_tasks=self._add_tasks,
            database_now=self._database_now,
        )
        self.agents = SimpleNamespace(
            version=self._get_version,
            version_source_ids=self._source_ids,
        )
        self.targets = SimpleNamespace(get_scoped=self._get_target)
        self.policies = SimpleNamespace(get_scoped=self._get_policy)
        self.diagnostic_sources = SimpleNamespace(get_scoped=self._get_source)

    async def __aenter__(self):
        self._uow_active = True
        return self

    async def __aexit__(self, *_):
        self._uow_active = False
        return False

    async def commit(self):
        self.commit_count += 1

    async def _get_turn(self, *, domain_id, turn_id, lock=False):
        del lock
        return self.turn if domain_id == 7 and turn_id == self.turn.turn_id else None

    async def _get_run_link(self, *, turn_id, purpose):
        if turn_id == self.turn.turn_id and purpose == "PRIMARY":
            return SimpleNamespace(ops_run_id=self.run.ops_run_id)
        return None

    async def _get_run_link_by_ops_run_id(self, *, ops_run_id):
        if ops_run_id == self.run.ops_run_id:
            return SimpleNamespace(turn_id=self.turn.turn_id)
        return None

    async def _list_messages(self, *, turn_id):
        return [self.message] if turn_id == self.turn.turn_id else []

    async def _recent(self, **_):
        return []

    async def _get_run(self, *, ops_run_id, lock=False):
        del lock
        if ops_run_id == self.run.ops_run_id:
            return self.run
        if ops_run_id == self.source_run.ops_run_id:
            return self.source_run
        return None

    async def _get_artifact(self, *, artifact_id):
        if artifact_id == self.source_artifact.artifact_id:
            return self.source_artifact
        return next(
            (
                artifact
                for artifact in self.artifacts
                if artifact.artifact_id == artifact_id
            ),
            None,
        )

    async def _get_artifact_by_key(self, *, ops_run_id, artifact_key):
        return next(
            (
                artifact
                for artifact in self.artifacts
                if artifact.ops_run_id == ops_run_id
                and artifact.artifact_key == artifact_key
            ),
            None,
        )

    async def _list_artifacts(self, *, ops_run_id):
        return [
            artifact
            for artifact in self.artifacts
            if artifact.ops_run_id == ops_run_id
        ]

    async def _list_tasks(self, *, ops_run_id, lock=False):
        del lock
        return [
            task
            for task in self.tasks
            if task.ops_run_id == ops_run_id
        ]

    async def _database_now(self):
        return datetime.now(UTC)

    async def _get_version(self, *, agent_id, agent_version_id):
        if (
            agent_id == self.run.agent_id
            and agent_version_id == self.version.agent_version_id
        ):
            return self.version
        return None

    async def _source_ids(self, **_):
        return []

    async def _get_target(self, *, target_id, domain_id):
        if target_id == self.target.target_id and domain_id == 7:
            return self.target
        return None

    async def _get_policy(self, *, policy_id, domain_id):
        if policy_id == self.policy.policy_id and domain_id == 7:
            return self.policy
        return None

    async def _get_source(self, **_):
        return None

    async def _add_artifact(self, row):
        self.artifacts.append(row)
        return row

    async def _add_tasks(self, rows):
        self.tasks.extend(rows)
        return rows

    async def _add_invocation(self, row):
        self.invocations.append(row)
        return row

    async def _add_tool_invocation(self, row):
        self.tool_invocations.append(row)
        return row

    async def _add_revision(self, row):
        self.revisions.append(row)
        return row

    async def _list_input_items(self, **_):
        return self.input_items

    async def _add_evidence(self, row):
        self.evidence.append(row)
        return row

    async def _get_evidence_by_artifact(self, *, turn_id, artifact_id):
        return next(
            (
                evidence
                for evidence in self.evidence
                if evidence.turn_id == turn_id
                and evidence.artifact_id == artifact_id
            ),
            None,
        )

    async def _add_event(self, row):
        row.created_at = datetime.now(UTC)
        self.events.append(row)
        return row


class _AgentCatalog:
    async def resolve_diagnosis_model(self, **_):
        return {"technical_name": "test-model", "revision": "1"}


class _PastedLogReasoner:
    def __init__(self, uow=None) -> None:
        self._uow = uow

    async def plan(self, **kwargs):
        if self._uow is not None:
            self.raw_input_persisted_before_model = any(
                artifact.artifact_key == "turn-user-input:1"
                for artifact in self._uow.artifacts
            )
        self.available_tools = kwargs["available_tools"]
        self.target_context = kwargs.get("target_context", {})
        output = InvestigationPlanningOutput(
            input_envelope=TurnInputEnvelope(
                materials=(
                    InputMaterial(
                        item_no=1,
                        material_kind=MaterialKind.ORACLE_ALERT_LOG,
                        summary="用户提供了 ORA-27157 Alert Log 片段",
                        key_facts=("ORA-27157",),
                        confidence=0.99,
                        contains_user_evidence=True,
                    ),
                ),
                explicit_question="分析这段 ORA-27157 日志",
                supplied_evidence_summary=("Oracle Alert Log 包含 ORA-27157",),
            ),
            task_frame=TaskFrame(
                objectives=(TaskObjective.EXPLAIN,),
                problem_statement="解释 ORA-27157 的含义和可能原因",
                known_facts=("Oracle 后台进程报告 OS post/wait facility removed",),
                unknowns=("操作系统 IPC 资源是否被人为移除",),
                success_criteria=("基于现有日志给出可审计解释",),
            ),
            plan=InvestigationPlan(
                revision_no=1,
                actions=(),
                answer_if_no_more_evidence=True,
                stop_reason="用户证据足以先解释错误机制",
            ),
        )
        digest = "a" * 64
        return StructuredModelResult(
            output=output,
            receipt=ModelInvocationReceipt(
                purpose="aiops.investigation-plan",
                schema_id="InvestigationPlanningOutput",
                model_technical_name="test-model",
                model_revision="1",
                prompt_id="aiops.investigation-planner",
                prompt_version="1",
                prompt_sha256=digest,
                input_sha256=digest,
                output_sha256=digest,
                duration_ms=1,
            ),
        )


class _ReplanReasoner(_PastedLogReasoner):
    async def replan(self, **kwargs):
        self.replan_inputs = dict(kwargs)
        planned = await self.plan(
            available_tools=(),
            target_context=kwargs["target_context"],
        )
        output = planned.output.model_copy(
            update={
                "plan": planned.output.plan.model_copy(
                    update={"revision_no": 2}
                )
            }
        )
        return StructuredModelResult(output=output, receipt=planned.receipt)


class _FrozenToolExecutor:
    def __init__(self) -> None:
        self.task_keys = []

    async def execute(self, context):
        self.task_keys.append(context.task_key)
        tool_id = context.task_key.removeprefix("diagnostic:")
        if tool_id == "db.instance.identity":
            return DatabaseDiagnosticResult(
                target_id=context.target_id,
                tool_id=tool_id,
                status="SUCCEEDED",
            )
        return DatabaseDiagnosticResult(
            target_id=context.target_id,
            tool_id=tool_id,
            status="GAP",
            gap=EvidenceGap(
                code="PRIVILEGE_MISSING",
                tool_id=tool_id,
                detail="缺少最小只读权限",
            ),
        )


class _IdentityGapToolExecutor(_FrozenToolExecutor):
    async def execute(self, context):
        self.task_keys.append(context.task_key)
        tool_id = context.task_key.removeprefix("diagnostic:")
        if tool_id == "db.instance.identity":
            return DatabaseDiagnosticResult(
                target_id=context.target_id,
                tool_id=tool_id,
                status="GAP",
                gap=EvidenceGap(
                    code="PRIVILEGE_MISSING",
                    tool_id=tool_id,
                    detail="实例身份视图不可读",
                ),
            )
        return DatabaseDiagnosticResult(
            target_id=context.target_id,
            tool_id=tool_id,
            status="SUCCEEDED",
        )


class _CapturingGrantCodec:
    def __init__(self) -> None:
        self.grant = None

    def issue(self, grant):
        self.grant = grant
        return "g" * 64


class _GapExecutorClient:
    def __init__(self) -> None:
        self.calls = []

    async def execute_diagnostic(self, request, *, trace_id):
        del trace_id
        self.calls.append(request)
        return ReadDiagnosticResult(
            executor_request_id=request.executor_request_id,
            status="GAP",
            error_code="PRIVILEGE_MISSING",
        )


class _RetryableGapExecutorClient(_GapExecutorClient):
    async def execute_diagnostic(self, request, *, trace_id):
        del trace_id
        self.calls.append(request)
        return ReadDiagnosticResult(
            executor_request_id=request.executor_request_id,
            status="GAP",
            error_code="TARGET_CONNECTION_TIMEOUT",
            retryable=True,
        )


class InvestigationFailureProjectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_replan_uses_snapshots_and_persists_supported_revision(
        self,
    ) -> None:
        """重规划不能在 UoW 关闭后继续依赖 ORM 实体。"""
        uow = _PlanningUow()
        initial = (
            await _PastedLogReasoner().plan(available_tools=())
        ).output
        plan_artifact_id = uuid7()
        task_frame_artifact_id = uuid7()
        assessment_artifact_id = uuid7()
        plan_artifact = _SessionBoundArtifact(
            owner=uow,
            artifact_id=plan_artifact_id,
            ops_run_id=uow.run.ops_run_id,
            artifact_key="turn-investigation-plan:1",
            schema_version=initial.plan.schema_version,
            payload_json=initial.plan.model_dump(mode="json"),
        )
        task_frame_artifact = _SessionBoundArtifact(
            owner=uow,
            artifact_id=task_frame_artifact_id,
            ops_run_id=uow.run.ops_run_id,
            artifact_key="turn-task-frame:1",
            schema_version=initial.task_frame.schema_version,
            payload_json=initial.task_frame.model_dump(mode="json"),
        )
        assessment_artifact = _SessionBoundArtifact(
            owner=uow,
            artifact_id=assessment_artifact_id,
            ops_run_id=uow.run.ops_run_id,
            artifact_key="evidence:assess",
            schema_version="DBA_SUFFICIENCY.v1",
            payload_json={
                "schema_version": "DBA_SUFFICIENCY.v1",
                "status": "NEEDS_EVIDENCE",
                "evidence": [],
                "gaps": [],
                "reasons": ["仍需补充证据"],
            },
        )
        prior_evidence = _SessionBoundArtifact(
            owner=uow,
            artifact_id=uuid7(),
            ops_run_id=uow.run.ops_run_id,
            artifact_key="diagnostic:a1",
            schema_version="DBA_TOOL_RESULT.v1",
            payload_json={},
        )
        uow.artifacts.extend(
            (
                plan_artifact,
                task_frame_artifact,
                assessment_artifact,
                prior_evidence,
            )
        )
        uow.turn.status = "REPLANNING"
        uow.turn.current_plan_revision = 1
        uow.turn.investigation_round = 1
        uow.turn.current_plan_artifact_id = plan_artifact_id
        uow.turn.task_frame_artifact_id = task_frame_artifact_id
        uow.turn.assessment_artifact_id = assessment_artifact_id
        uow.tasks.append(
            SimpleNamespace(
                ops_run_id=uow.run.ops_run_id,
                task_key="answer:compose",
                status="PENDING",
                depends_on_json=[],
                input_artifacts_json=[],
            )
        )
        reasoner = _ReplanReasoner()
        registry = PlaybookRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in DiagnosticRegistry.load().tools
            )
        )
        service = TurnPlanningService(
            uow_factory=lambda: uow,
            investigation_reasoner=reasoner,
            playbook_registry=registry,
            task_compiler=InvestigationTaskCompiler(registry),
            tool_snapshot_builder=ToolExecutionSnapshotBuilder(
                playbook_registry=registry,
                diagnostic_registry=DiagnosticRegistry.load(),
            ),
            agent_catalog=_AgentCatalog(),
        )

        result = await service.execute_replan(
            {
                "domain_id": 7,
                "turn_id": str(uow.turn.turn_id),
                "ops_run_id": str(uow.run.ops_run_id),
                "assessment_artifact_id": str(assessment_artifact_id),
                "revision_no": 2,
            }
        )

        self.assertEqual("COLLECTING", result["status"])
        self.assertEqual("COLLECTING", uow.turn.status)
        self.assertEqual(2, uow.turn.current_plan_revision)
        self.assertEqual("EVIDENCE_DRIVEN", uow.revisions[-1].revision_type)
        self.assertIn(
            "diagnostic:a1",
            next(
                task
                for task in uow.tasks
                if task.task_key == "evidence:assess:r2"
            ).input_artifacts_json,
        )
        self.assertEqual(
            initial.plan.model_dump(mode="json"),
            reasoner.replan_inputs["prior_plan"],
        )

    async def test_enabled_unreachable_target_is_attemptable_in_turn_budget(
        self,
    ) -> None:
        """上一轮连接健康失败不能提前禁用本Turn的只读尝试。"""
        uow = _PlanningUow()
        uow.target.connectivity_status = "UNREACHABLE"
        registry = PlaybookRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in DiagnosticRegistry.load().tools
            )
        )
        service = TurnPlanningService(
            uow_factory=lambda: uow,
            investigation_reasoner=_PastedLogReasoner(),
            playbook_registry=registry,
            task_compiler=InvestigationTaskCompiler(registry),
            tool_snapshot_builder=ToolExecutionSnapshotBuilder(
                playbook_registry=registry,
                diagnostic_registry=DiagnosticRegistry.load(),
            ),
            agent_catalog=_AgentCatalog(),
        )

        await service.execute(
            {"domain_id": 7, "turn_id": str(uow.turn.turn_id)}
        )

        database = uow.run.plan_snapshot_json["investigation_execution"]["database"]
        self.assertTrue(database["automatic_access_enabled"])
        self.assertIn(
            "TARGET_CONNECTIVITY_UNAVAILABLE",
            {item["code"] for item in database["initial_gaps"]},
        )

    async def test_pasted_oracle_log_reaches_evidence_assessment_without_tool(
        self,
    ) -> None:
        """用户已提供日志时，不需要Playbook也必须进入证据评估和回答。"""
        uow = _PlanningUow()
        reasoner = _PastedLogReasoner(uow)
        registry = PlaybookRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in DiagnosticRegistry.load().tools
            )
        )
        service = TurnPlanningService(
            uow_factory=lambda: uow,
            investigation_reasoner=reasoner,
            playbook_registry=registry,
            task_compiler=InvestigationTaskCompiler(registry),
            tool_snapshot_builder=ToolExecutionSnapshotBuilder(
                playbook_registry=registry,
                diagnostic_registry=DiagnosticRegistry.load(),
            ),
            agent_catalog=_AgentCatalog(),
        )

        result = await service.execute(
            {"domain_id": 7, "turn_id": str(uow.turn.turn_id)}
        )

        self.assertEqual("COLLECTING", result["status"])
        self.assertTrue(reasoner.raw_input_persisted_before_model)
        self.assertEqual("订单生产库", reasoner.target_context["display_name"])
        self.assertEqual("BOUND", reasoner.target_context["selection_status"])
        self.assertNotIn("connection_profile", reasoner.target_context)
        self.assertNotIn("diagnostic_credential_id", reasoner.target_context)
        self.assertEqual("ORACLE_ALERT_LOG", uow.input_items[0].detected_kind)
        self.assertEqual(1, len(uow.evidence))
        self.assertEqual("USER_PROVIDED", uow.evidence[0].evidence_role)
        self.assertEqual([], uow.invocations)
        self.assertEqual([], uow.tool_invocations)
        self.assertEqual(
            ["evidence:assess", "answer:compose"],
            [task.task_key for task in uow.tasks],
        )
        self.assertEqual("READY", uow.tasks[0].status)
        self.assertEqual(
            ("turn-user-input:1", "turn-input-analysis:1"),
            tuple(uow.tasks[0].input_artifacts_json),
        )
        self.assertEqual(1, len(uow.revisions))
        self.assertEqual(2, uow.commit_count)
        planned_event = next(
            event
            for event in uow.events
            if event.event_type == "investigation.planned"
        )
        self.assertEqual(
            {"revision_no": 1, "actions": []},
            planned_event.payload_json["plan"],
        )

    async def test_source_run_final_artifact_is_inherited_as_current_evidence(
        self,
    ) -> None:
        """告警或巡检继续对话必须继承来源Run结果，不只保存关联ID。"""
        uow = _PlanningUow()
        uow.run.plan_snapshot_json["source_run_id"] = str(
            uow.source_run.ops_run_id
        )
        registry = PlaybookRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in DiagnosticRegistry.load().tools
            )
        )
        service = TurnPlanningService(
            uow_factory=lambda: uow,
            investigation_reasoner=_PastedLogReasoner(),
            playbook_registry=registry,
            task_compiler=InvestigationTaskCompiler(registry),
            tool_snapshot_builder=ToolExecutionSnapshotBuilder(
                playbook_registry=registry,
                diagnostic_registry=DiagnosticRegistry.load(),
            ),
            agent_catalog=_AgentCatalog(),
        )

        await service.execute(
            {"domain_id": 7, "turn_id": str(uow.turn.turn_id)}
        )

        inherited = next(
            item
            for item in uow.artifacts
            if item.artifact_key == "turn-source-run-evidence:1"
        )
        self.assertEqual("SOURCE_RUN_EVIDENCE.v1", inherited.schema_version)
        self.assertEqual(
            {"USER_PROVIDED", "CONTEXT"},
            {item.evidence_role for item in uow.evidence},
        )
        self.assertIn(
            "turn-source-run-evidence:1",
            uow.tasks[0].input_artifacts_json,
        )

    async def test_terminal_planning_failure_updates_turn_and_run(self) -> None:
        uow = _PlanningUow()
        service = object.__new__(TurnPlanningService)
        service._uow_factory = lambda: uow

        result = await service.fail_terminal(
            {"domain_id": 7, "turn_id": str(uow.turn.turn_id)},
            error_code="AIOPS_INVESTIGATION_PLAN_INVALID",
            error_message="调查计划未通过安全校验",
        )

        self.assertEqual("FAILED", result["status"])
        self.assertEqual("FAILED", uow.turn.status)
        self.assertEqual("PLANNING", uow.turn.error_domain)
        self.assertEqual(
            "AIOPS_INVESTIGATION_PLAN_INVALID", uow.turn.error_code
        )
        self.assertEqual("FAILED", uow.run.status)
        self.assertIsNotNone(uow.turn.completed_at)
        self.assertIsNotNone(uow.run.completed_at)
        self.assertEqual("turn.status", uow.events[-1].event_type)
        self.assertEqual("FAILED", uow.events[-1].payload_json["status"])
        self.assertEqual(1, uow.commit_count)

    async def test_unexpected_planning_key_error_keeps_safe_stage_detail(self) -> None:
        """未预期的缺键错误必须保留安全定位信息，不能伪装成策略拒绝。"""
        service = object.__new__(TurnPlanningService)

        async def fail_after_model(_payload):
            raise KeyError("diagnostic-task:missing")

        service._execute_once = fail_after_model

        with self.assertRaises(TurnPlanningStageError) as raised:
            await service.execute({"domain_id": 7, "turn_id": str(uuid7())})

        self.assertEqual(
            "AIOPS_INVESTIGATION_PLAN_INTERNAL_ERROR",
            raised.exception.code,
        )
        self.assertEqual("KeyError", raised.exception.cause_type)
        self.assertEqual(
            "missing-key:diagnostic-task:missing",
            raised.exception.safe_detail,
        )
        self.assertNotIn("安全校验", str(raised.exception))

    def test_terminal_failure_summary_distinguishes_internal_error(self) -> None:
        """内部错误与模型计划安全校验失败使用不同的用户文案。"""
        internal = TurnPlanningService._terminal_failure_summary(
            "AIOPS_INVESTIGATION_PLAN_INTERNAL_ERROR"
        )
        invalid = TurnPlanningService._terminal_failure_summary(
            "AIOPS_INVESTIGATION_PLAN_INVALID"
        )

        self.assertIn("内部错误", internal)
        self.assertNotIn("安全校验", internal)
        self.assertIn("安全校验", invalid)

    def test_schema_failure_summary_does_not_claim_policy_rejection(
        self,
    ) -> None:
        """Schema合同漂移必须明确指向数据库结构，不能误报策略拒绝。"""
        summary = TurnPlanningService._terminal_failure_summary(
            "AIOPS_SCHEMA_INTEGRITY_ERROR"
        )

        self.assertIn("数据库结构", summary)
        self.assertNotIn("安全校验", summary)

    def test_safe_plan_projection_exposes_frozen_query_for_approval(
        self,
    ) -> None:
        """用户计划必须展示冻结SQL和参数，但不得泄露内部策略快照。"""
        projection = safe_plan_projection(
            {
                "revision_no": 2,
                "actions": [
                    {
                        "action_id": "a1",
                        "question": "查看当前会话明细",
                        "tool_id": "db.oracle.readonly_query",
                        "input": {
                            "sql": "SELECT sql_text AS sample FROM v$sqlstats",
                            "parameters": {},
                        },
                        "measurement_semantics": "CURRENT_ACTIVITY",
                        "depends_on": [],
                    }
                ],
            },
            execution_snapshot={
                "dynamic_invocations": {
                    "dynamic:a1": {
                        "action_id": "a1",
                        "policy_snapshot": {"star_projection_allowed": False},
                        "validated_query": {
                            "normalized_sql": (
                                "SELECT sql_text AS sample FROM v$sqlstats "
                                "FETCH FIRST 200 ROWS ONLY"
                            ),
                            "parameters": {},
                            "execution_decision": "APPROVAL_REQUIRED",
                            "approval_reason_codes": [
                                "DYNAMIC_SQL_SENSITIVE_COLUMN_APPROVAL_REQUIRED"
                            ],
                        },
                    }
                }
            },
        )

        self.assertEqual(2, projection["revision_no"])
        self.assertEqual(
            "ORACLE_SQL_DYNAMIC", projection["actions"][0]["tool_class"]
        )
        self.assertEqual(
            "APPROVAL_REQUIRED",
            projection["actions"][0]["execution_mode"],
        )
        action = projection["actions"][0]
        self.assertIn("v$sqlstats", action["sql_text"])
        self.assertEqual({}, action["parameters"])
        self.assertEqual(
            ["DYNAMIC_SQL_SENSITIVE_COLUMN_APPROVAL_REQUIRED"],
            action["approval_reason_codes"],
        )
        self.assertNotIn("policy_snapshot", str(projection))

    async def test_terminal_task_failure_updates_chat_turn(self) -> None:
        uow = _PlanningUow()
        runtime = object.__new__(AIOpsRuntimeService)
        now = datetime.now(UTC)

        await runtime._project_chat_turn_failure(
            uow=uow,
            run=uow.run,
            error_code="HANDLER_TERMINAL_FAILURE",
            public_summary="诊断任务执行失败",
            now=now,
        )

        self.assertEqual("FAILED", uow.turn.status)
        self.assertEqual("EXECUTION", uow.turn.error_domain)
        self.assertEqual("HANDLER_TERMINAL_FAILURE", uow.turn.error_code)
        self.assertEqual(now, uow.turn.completed_at)
        self.assertEqual("turn.status", uow.events[-1].event_type)
        self.assertEqual("FAILED", uow.events[-1].payload_json["status"])


class DbaPlaybookFrameworkTest(unittest.TestCase):
    def test_bound_target_is_frozen_and_identity_precedes_top_sql(self) -> None:
        investigation = InvestigationPlanningOutput(
            input_envelope=TurnInputEnvelope(
                materials=(
                    InputMaterial(
                        item_no=1,
                        material_kind=MaterialKind.QUESTION,
                        summary="用户要求分析 Top SQL",
                        key_facts=("分析当前数据库中的 Top SQL",),
                        confidence=1,
                    ),
                ),
                explicit_question="分析下数据库中的 Top SQL",
            ),
            task_frame=TaskFrame(
                objectives=(TaskObjective.DIAGNOSE,),
                problem_statement="分析当前数据库中的高负载 SQL",
                database_context={"selection_status": "UNKNOWN"},
                unknowns=("当前连接的是哪个数据库",),
                success_criteria=("定位累计资源消耗排名靠前的 SQL",),
            ),
            plan=InvestigationPlan(
                revision_no=1,
                actions=(
                    InvestigationAction(
                        action_id="a1",
                        question="当前数据库中排名靠前的 SQL 是哪些？",
                        tool_id="db.sql.top_current",
                        expected_evidence_kind="TOP_SQL",
                        measurement_semantics=(
                            MeasurementSemantics.CUMULATIVE_SINCE_LOAD
                        ),
                    ),
                ),
            ),
        )
        target_context = {
            "target_id": "target-1",
            "display_name": "订单生产库",
            "db_type": "ORACLE",
            "configured_version": "19c",
            "environment": "PROD",
            "db_role": "PRIMARY",
            "status": "ENABLED",
            "connectivity_status": "CONNECTED",
            "selection_status": "BOUND",
        }

        bound = TurnPlanningService._bind_target_to_plan(
            investigation=investigation,
            target_context=target_context,
            available_tools=(
                {"tool_id": "db.instance.identity", "version": "1.0.0"},
                {"tool_id": "db.sql.top_current", "version": "1.0.0"},
            ),
        )

        self.assertEqual(target_context, bound.task_frame.database_context)
        self.assertEqual(
            ["db.instance.identity", "db.sql.top_current"],
            [action.tool_id for action in bound.plan.actions],
        )
        self.assertEqual(("a1",), bound.plan.actions[1].depends_on)
        self.assertIn("订单生产库", bound.plan.actions[0].question)
        self.assertNotIn("哪个", bound.plan.actions[0].question)

    def test_existing_identity_action_is_normalized_without_duplication(
        self,
    ) -> None:
        investigation = InvestigationPlanningOutput(
            input_envelope=TurnInputEnvelope(
                materials=(
                    InputMaterial(
                        item_no=1,
                        material_kind=MaterialKind.QUESTION,
                        summary="用户要求分析 Top SQL",
                        key_facts=("分析当前数据库中的 Top SQL",),
                        confidence=1,
                    ),
                ),
                explicit_question="分析下数据库中的 Top SQL",
            ),
            task_frame=TaskFrame(
                objectives=(TaskObjective.DIAGNOSE,),
                problem_statement="分析当前数据库中的高负载 SQL",
                success_criteria=("定位累计资源消耗排名靠前的 SQL",),
            ),
            plan=InvestigationPlan(
                revision_no=1,
                actions=(
                    InvestigationAction(
                        action_id="a1",
                        question="正在分析的是哪个 Oracle 数据库实例？",
                        tool_id="db.instance.identity",
                        expected_evidence_kind="DATABASE_IDENTITY",
                        measurement_semantics=(
                            MeasurementSemantics.CURRENT_ACTIVITY
                        ),
                    ),
                    InvestigationAction(
                        action_id="a2",
                        question="当前数据库中排名靠前的 SQL 是哪些？",
                        tool_id="db.sql.top_current",
                        expected_evidence_kind="TOP_SQL",
                        measurement_semantics=(
                            MeasurementSemantics.CUMULATIVE_SINCE_LOAD
                        ),
                    ),
                ),
            ),
        )

        bound = TurnPlanningService._bind_target_to_plan(
            investigation=investigation,
            target_context={
                "target_id": "target-1",
                "display_name": "订单生产库",
                "db_type": "ORACLE",
                "selection_status": "BOUND",
            },
            available_tools=(
                {"tool_id": "db.instance.identity", "version": "1.0.0"},
            ),
        )

        self.assertEqual(2, len(bound.plan.actions))
        self.assertEqual("db.instance.identity", bound.plan.actions[0].tool_id)
        self.assertIn("已绑定 Target", bound.plan.actions[0].question)
        self.assertIn("订单生产库", bound.plan.actions[0].question)
        self.assertEqual(("a1",), bound.plan.actions[1].depends_on)

    def test_compiler_enforces_identity_and_resolves_later_actions(self) -> None:
        registry = PlaybookRegistry.load()
        actions = (
            SimpleNamespace(
                action_id="a1",
                tool_id="db.sql.top_current",
                depends_on=(),
            ),
            SimpleNamespace(
                action_id="a2",
                tool_id="db.instance.identity",
                depends_on=(),
            ),
            SimpleNamespace(
                action_id="a3",
                tool_id="db.oracle.readonly_query",
                depends_on=(),
            ),
            SimpleNamespace(
                action_id="a4",
                tool_id="custom.inspect",
                depends_on=("a5",),
            ),
            SimpleNamespace(
                action_id="a5",
                tool_id="custom.collect",
                depends_on=(),
            ),
        )

        compiled = InvestigationTaskCompiler(registry).compile(
            DbaPlaybookPlan(catalog_hash=registry.catalog_hash, items=()),
            investigation_actions=actions,
        )
        tasks = {item.task_key: item for item in compiled.tasks}

        self.assertEqual(
            ("diagnostic:a2",), tasks["diagnostic:a1"].depends_on
        )
        self.assertEqual(("diagnostic:a2",), tasks["dynamic:a3"].depends_on)
        self.assertEqual(
            ("diagnostic:a5",), tasks["diagnostic:a4"].depends_on
        )

    def test_investigation_action_freezes_only_its_selected_atomic_tool(self) -> None:
        """Playbook只能提供默认值，不能扩大模型Action的实际执行范围。"""
        diagnostic_registry = DiagnosticRegistry.load()
        registry = PlaybookRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in diagnostic_registry.tools
            )
        )
        service = object.__new__(TurnPlanningService)
        service._playbook_registry = registry
        capabilities = DbaCapabilitySnapshot(
            agent_id=str(uuid7()),
            agent_version_id=str(uuid7()),
            target_id=str(uuid7()),
            database_type="ORACLE",
            database_version="19c",
            target_enabled=True,
            target_reachable=True,
            target_capabilities=(
                "DB_READONLY",
                "dynamic_performance_views",
                "dba_catalog_views",
            ),
        )
        investigation = SimpleNamespace(
            suggested_playbook_ids=("oracle.sql.top_current",),
            plan=SimpleNamespace(
                actions=(
                    SimpleNamespace(
                        action_id="a1",
                        question="哪些SQL累计消耗最高？",
                        tool_id="db.sql.top_current",
                        input={"limit": 5},
                        depends_on=(),
                        measurement_semantics="CUMULATIVE_SINCE_LOAD",
                    ),
                )
            ),
        )

        plan = build_playbook_plan(service._playbook_registry)
        compiled = InvestigationTaskCompiler(registry).compile(
            plan,
            investigation_actions=investigation.plan.actions,
        )
        snapshot = ToolExecutionSnapshotBuilder(
            playbook_registry=registry,
            diagnostic_registry=diagnostic_registry,
        ).build(
            plan=plan,
            compiled=compiled,
            capabilities=capabilities,
            database_execution={"automatic_access_enabled": True},
            direct_actions=investigation.plan.actions,
        )

        invocation = snapshot["direct_invocations"][
            compiled.diagnostic_task_keys[0]
        ]
        self.assertEqual("a1", invocation["action_id"])
        self.assertEqual(
            "db.sql.top_current",
            invocation["tool"]["tool_id"],
        )

    def test_disabled_target_keeps_diagnostic_plan_with_access_gap(self) -> None:
        """Target 停用时保留诊断计划，但禁止自动数据库直连。"""
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._diagnostic_registry = DiagnosticRegistry.load()
        target = SimpleNamespace(
            domain_id=7,
            status="DISABLED",
            db_type="ORACLE",
            version_code="19c",
            row_version=3,
            endpoint_json={
                "host": "db.internal",
                "port": 1521,
                "service": "PDB1",
            },
            diagnostic_credential_id=uuid7(),
            readonly_connection_enabled=True,
            connectivity_status="UNREACHABLE",
            capabilities_json={},
        )

        _, snapshot = runtime._database_diagnostic_blueprint_snapshot(
            command=SimpleNamespace(blueprint_version="1"),
            target=target,
            binding=SimpleNamespace(),
            policy=None,
        )

        self.assertFalse(snapshot["automatic_access_enabled"])
        self.assertTrue(snapshot["tools"])
        self.assertEqual(
            {"TARGET_INACTIVE", "TARGET_CONNECTIVITY_UNAVAILABLE"},
            {
                item["code"]
                for item in snapshot["initial_gaps"]
                if item["code"].startswith("TARGET_")
            },
        )

    def test_playbook_handler_executes_only_frozen_tool_dag(self) -> None:
        executor = _FrozenToolExecutor()
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="playbook:1:oracle.sql.top_current",
            target_id=str(uuid7()),
            agent_id=str(uuid7()),
            trigger_type="CHAT",
            trace_id="trace-playbook",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "investigation_execution": {
                    "diagnostic_catalog_hash": "a" * 64,
                    "capability_snapshot_hash": "b" * 64,
                    "database": {
                        "domain_id": 7,
                        "db_type": "ORACLE",
                        "configured_version": "19c",
                        "target_row_version": 1,
                        "connection_profile": {},
                        "diagnostic_credential_id": str(uuid7()),
                    },
                    "invocations": {
                        "playbook:1:oracle.sql.top_current": {
                            "playbook_id": "oracle.sql.top_current",
                            "playbook_version": "1.0.0",
                            "manifest_hash": "c" * 64,
                            "measurement_semantics": "CUMULATIVE_SINCE_LOAD",
                            "presentation_kind": "TABLE_AND_CHART",
                            "output_schema": "oracle.sql.top_current.output.v1",
                            "tools": [
                                {
                                    "step_id": "identity",
                                    "depends_on": [],
                                    "tool_id": "db.instance.identity",
                                    "tool_version": "1.0.0",
                                },
                                {
                                    "step_id": "top_sql",
                                    "depends_on": ["identity"],
                                    "tool_id": "db.sql.top_current",
                                    "tool_version": "1.0.0",
                                },
                            ],
                        }
                    },
                }
            },
            policy_snapshot={},
            input_artifacts=(),
        )

        result = self.run_async(
            DbaPlaybookInvocationHandler(
                database_handler=executor
            ).execute(context)
        )

        self.assertEqual("PARTIAL", result.status)
        self.assertEqual(
            [
                "diagnostic:db.instance.identity",
                "diagnostic:db.sql.top_current",
            ],
            executor.task_keys,
        )

    def test_playbook_handler_preserves_frozen_database_access_denial(self) -> None:
        """执行阶段不得覆盖规划时冻结的数据库访问拒绝结论。"""
        class _AccessSnapshotExecutor:
            def __init__(self) -> None:
                self.access_values = []

            async def execute(self, context):
                snapshot = context.plan_snapshot["database_diagnostics"]
                self.access_values.append(snapshot["automatic_access_enabled"])
                return DatabaseDiagnosticResult(
                    target_id=context.target_id,
                    tool_id=context.task_key.removeprefix("diagnostic:"),
                    status="GAP",
                    gap=EvidenceGap(
                        code="DIAGNOSTIC_POLICY_DENIED",
                        detail="当前策略禁止数据库直连诊断",
                        retryable=False,
                    ),
                )

        executor = _AccessSnapshotExecutor()
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="playbook:1:oracle.sql.top_current",
            target_id=str(uuid7()),
            agent_id=str(uuid7()),
            trigger_type="CHAT",
            trace_id="trace-playbook-denied",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "investigation_execution": {
                    "diagnostic_catalog_hash": "a" * 64,
                    "capability_snapshot_hash": "b" * 64,
                    "database": {
                        "automatic_access_enabled": False,
                        "initial_gaps": [
                            {
                                "code": "DIAGNOSTIC_POLICY_DENIED",
                                "detail": "当前策略禁止数据库直连诊断",
                                "retryable": False,
                            }
                        ],
                    },
                    "invocations": {
                        "playbook:1:oracle.sql.top_current": {
                            "playbook_id": "oracle.sql.top_current",
                            "playbook_version": "1.0.0",
                            "manifest_hash": "c" * 64,
                            "measurement_semantics": "CUMULATIVE_SINCE_LOAD",
                            "presentation_kind": "TABLE",
                            "output_schema": "oracle.sql.top_current.output.v1",
                            "tools": [
                                {
                                    "step_id": "top_sql",
                                    "depends_on": [],
                                    "tool_id": "db.sql.top_current",
                                    "tool_version": "1.0.0",
                                }
                            ],
                        }
                    },
                }
            },
            policy_snapshot={},
            input_artifacts=(),
        )

        result = self.run_async(
            DbaPlaybookInvocationHandler(
                database_handler=executor
            ).execute(context)
        )

        self.assertEqual("FAILED", result.status)
        self.assertEqual([False], executor.access_values)

    def test_identity_gap_does_not_skip_independent_readonly_tool(self) -> None:
        executor = _IdentityGapToolExecutor()
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="playbook:1:oracle.storage.tablespace",
            target_id=str(uuid7()),
            agent_id=str(uuid7()),
            trigger_type="CHAT",
            trace_id="trace-playbook-identity-gap",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "investigation_execution": {
                    "diagnostic_catalog_hash": "a" * 64,
                    "capability_snapshot_hash": "b" * 64,
                    "database": {},
                    "invocations": {
                        "playbook:1:oracle.storage.tablespace": {
                            "playbook_id": "oracle.storage.tablespace",
                            "playbook_version": "1.0.0",
                            "manifest_hash": "c" * 64,
                            "measurement_semantics": "CURRENT_ACTIVITY",
                            "presentation_kind": "TABLE",
                            "output_schema": "oracle.storage.tablespace.output.v1",
                            "tools": [
                                {
                                    "step_id": "identity",
                                    "depends_on": [],
                                    "tool_id": "db.instance.identity",
                                    "tool_version": "1.0.0",
                                },
                                {
                                    "step_id": "tablespace",
                                    "depends_on": ["identity"],
                                    "tool_id": "db.storage.capacity",
                                    "tool_version": "1.0.0",
                                },
                            ],
                        }
                    },
                }
            },
            policy_snapshot={},
            input_artifacts=(),
        )

        result = self.run_async(
            DbaPlaybookInvocationHandler(
                database_handler=executor
            ).execute(context)
        )

        self.assertEqual("PARTIAL", result.status)
        self.assertEqual(
            [
                "diagnostic:db.instance.identity",
                "diagnostic:db.storage.capacity",
            ],
            executor.task_keys,
        )

    @staticmethod
    def run_async(awaitable):
        import asyncio

        return asyncio.run(awaitable)

    def test_repository_catalog_only_references_allowlisted_tools(self) -> None:
        tools = DiagnosticRegistry.load().tools
        registry = PlaybookRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in tools
            )
        )
        manifest = registry.latest("oracle.sql.top_current")
        validator = ToolExecutionSnapshotBuilder(
            playbook_registry=registry,
            diagnostic_registry=DiagnosticRegistry.load(),
        )
        validator.validate_catalog()

        self.assertEqual(
            MeasurementSemantics.CUMULATIVE_SINCE_LOAD,
            manifest.measurement_semantics,
        )
        self.assertEqual(
            {
                "oracle.sql.top_current",
                "oracle.instance.performance",
                "oracle.instance.wait_summary",
                "oracle.instance.archive",
                "oracle.session.active",
                "oracle.session.blocking_chain",
                "oracle.storage.tablespace",
                "oracle.transaction.long_running",
                "oracle.replication.status",
                "oracle.configuration.parameters",
                "oracle.storage.temp_undo",
                "oracle.instance.redo_alert",
                "oracle.maintenance.health",
            },
            {item.playbook_id for item in registry.manifests()},
        )
        self.assertEqual(64, len(registry.catalog_hash))

    def test_top_sql_playbook_freezes_exact_tools_hashes_and_user_limit(self) -> None:
        diagnostics = DiagnosticRegistry.load()
        registry = PlaybookRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in diagnostics.tools
            )
        )
        manifest = registry.latest("oracle.sql.top_current")
        capabilities = _capabilities().model_copy(
            update={
                "target_capabilities": (
                    "DB_READONLY",
                    "dynamic_performance_views",
                ),
                "privileges": (
                    "V_$INSTANCE",
                    "V_$DATABASE",
                    "V_$SQLSTATS",
                ),
            }
        )
        plan = _playbook_plan(
            registry,
            (manifest,),
            input_by_id={manifest.playbook_id: {"limit": 20}},
        )
        compiled = InvestigationTaskCompiler(registry).compile(plan)
        snapshot = ToolExecutionSnapshotBuilder(
            playbook_registry=registry,
            diagnostic_registry=diagnostics,
        ).build(
            plan=plan,
            compiled=compiled,
            capabilities=capabilities,
            database_execution={
                "domain_id": 7,
                "target_row_version": 1,
                "db_type": "ORACLE",
                "configured_version": "19c",
                "connection_profile": {},
                "diagnostic_credential_id": "credential-1",
            },
        )

        invocation = snapshot["invocations"][
            "playbook:1:oracle.sql.top_current"
        ]
        self.assertEqual(
            ["db.instance.identity", "db.sql.top_current"],
            [item["tool_id"] for item in invocation["tools"]],
        )
        self.assertEqual(20, invocation["tools"][1]["parameters"]["limit"])
        self.assertEqual(64, len(invocation["tools"][1]["template_sha256"]))
        self.assertIn("SELECT", invocation["tools"][1]["manual_sql"])
        self.assertEqual(
            ["V_$SQLSTATS"],
            invocation["tools"][1]["required_privileges"],
        )

    def test_compiler_adds_monitoring_tasks_to_evidence_barrier(self) -> None:
        registry = PlaybookRegistry(
            (
                _manifest(
                    playbook_id="oracle.sql.current",
                    intent=DbaIntent.OBSERVE,
                    domain=DbaDomain.SQL_PERFORMANCE,
                ),
            )
        )
        manifest = registry.latest("oracle.sql.current")
        plan = _playbook_plan(registry, (manifest,))
        compiled = InvestigationTaskCompiler(registry).compile(
            plan,
            monitoring_binding_ids=("binding-1", "binding-2"),
        )

        self.assertEqual(
            ("observe:binding-1", "observe:binding-2"),
            compiled.monitoring_task_keys,
        )
        monitor_task = next(
            item
            for item in compiled.tasks
            if item.task_key == "observe:binding-1"
        )
        self.assertEqual("TOOL_INVOKE", monitor_task.task_type)
        evidence_task = next(
            item
            for item in compiled.tasks
            if item.task_key == "evidence:assess"
        )
        self.assertEqual(
            (
                "playbook:1:oracle.sql.current",
                "observe:binding-1",
                "observe:binding-2",
            ),
            evidence_task.depends_on,
        )

    def test_model_tasks_outlive_configured_model_request_timeout(self) -> None:
        registry = PlaybookRegistry(
            (
                _manifest(
                    playbook_id="oracle.sql.current",
                    intent=DbaIntent.OBSERVE,
                    domain=DbaDomain.SQL_PERFORMANCE,
                ),
            )
        )
        manifest = registry.latest("oracle.sql.current")
        compiled = InvestigationTaskCompiler(
            registry,
            model_timeout_seconds=90,
        ).compile(_playbook_plan(registry, (manifest,)))
        tasks = {item.task_key: item for item in compiled.tasks}

        self.assertEqual(105, tasks["evidence:assess"].timeout_seconds)
        self.assertEqual(210, tasks["answer:compose"].timeout_seconds)

    def test_database_handler_consumes_frozen_tool_version(self) -> None:
        codec = _CapturingGrantCodec()
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="diagnostic:db.instance.identity",
            target_id=str(uuid7()),
            agent_id=str(uuid7()),
            trigger_type="CHAT",
            trace_id="trace-tool-version",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "database_diagnostics": {
                    "domain_id": 7,
                    "target_row_version": 1,
                    "db_type": "ORACLE",
                    "connection_profile": {
                        "host": "db.internal",
                        "port": 1521,
                        "service": "PDB1",
                        "tls_enabled": False,
                    },
                    "diagnostic_credential_id": str(uuid7()),
                    "capability_snapshot_hash": "a" * 64,
                    "tools": [
                        {
                            "tool_id": "db.instance.identity",
                            "tool_version": "1.0.0",
                            "variant": "oracle.default",
                            "template_sha256": "b" * 64,
                            "parameters": {},
                            "limits": {
                                "statement_timeout_seconds": 10,
                                "max_result_rows": 10,
                                "max_result_bytes": 1024,
                                "max_columns": 16,
                                "max_cell_chars": 1024,
                            },
                        }
                    ],
                }
            },
            policy_snapshot={},
            input_artifacts=(),
            lease_token="lease-token",
            lease_until=(datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
        )

        result = self.run_async(
            DatabaseDiagnosticHandler(
                executor_client=_GapExecutorClient(),
                grant_codec=codec,
                grant_issuer="aiops-worker",
                grant_audience="aiops-db-executor",
                grant_ttl_seconds=30,
            ).execute(context)
        )

        self.assertEqual("GAP", result.status)
        self.assertEqual("1.0.0", codec.grant.tool_version)

    def test_database_handler_retries_then_returns_final_gap(self) -> None:
        codec = _CapturingGrantCodec()
        client = _RetryableGapExecutorClient()
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="diagnostic:db.instance.identity",
            target_id=str(uuid7()),
            agent_id=str(uuid7()),
            trigger_type="CHAT",
            trace_id="trace-retryable-gap",
            attempt=1,
            max_attempts=2,
            deadline_at=None,
            plan_snapshot={
                "database_diagnostics": {
                    "domain_id": 7,
                    "target_row_version": 1,
                    "db_type": "ORACLE",
                    "configured_version": "19c",
                    "connection_profile": {
                        "host": "db.internal",
                        "port": 1521,
                        "service": "PDB1",
                        "tls_enabled": False,
                    },
                    "diagnostic_credential_id": str(uuid7()),
                    "capability_snapshot_hash": "a" * 64,
                    "tools": [
                        {
                            "tool_id": "db.instance.identity",
                            "tool_version": "1.0.0",
                            "variant": "oracle.default",
                            "template_sha256": "b" * 64,
                            "parameters": {},
                            "limits": {
                                "statement_timeout_seconds": 10,
                                "max_result_rows": 10,
                                "max_result_bytes": 1024,
                                "max_columns": 16,
                                "max_cell_chars": 1024,
                            },
                        }
                    ],
                }
            },
            policy_snapshot={},
            input_artifacts=(),
            lease_token="lease-token",
            lease_until=(datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
        )
        handler = DatabaseDiagnosticHandler(
            executor_client=client,
            grant_codec=codec,
            grant_issuer="aiops-worker",
            grant_audience="aiops-db-executor",
            grant_ttl_seconds=30,
        )

        with self.assertRaises(RetryableTaskError):
            self.run_async(handler.execute(context))
        final = self.run_async(
            handler.execute(replace(context, attempt=2))
        )

        self.assertEqual("GAP", final.status)
        self.assertEqual("TARGET_CONNECTION_TIMEOUT", final.gap.code)
        self.assertEqual(2, len(client.calls))

    def test_database_handler_uses_configured_version_when_identity_is_gap(
        self,
    ) -> None:
        codec = _CapturingGrantCodec()
        client = _GapExecutorClient()
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="diagnostic:db.storage.capacity",
            target_id=str(uuid7()),
            agent_id=str(uuid7()),
            trigger_type="CHAT",
            trace_id="trace-configured-version",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "database_diagnostics": {
                    "domain_id": 7,
                    "target_row_version": 1,
                    "db_type": "ORACLE",
                    "configured_version": "19c",
                    "connection_profile": {
                        "host": "db.internal",
                        "port": 1521,
                        "service": "PDB1",
                        "tls_enabled": False,
                    },
                    "diagnostic_credential_id": str(uuid7()),
                    "capability_snapshot_hash": "a" * 64,
                    "tools": [
                        {
                            "tool_id": "db.storage.capacity",
                            "tool_version": "1.0.0",
                            "variant": "oracle.default",
                            "template_sha256": "b" * 64,
                            "supported_version_min": 19,
                            "supported_version_max_exclusive": 27,
                            "parameters": {},
                            "limits": {
                                "statement_timeout_seconds": 10,
                                "max_result_rows": 10,
                                "max_result_bytes": 1024,
                                "max_columns": 16,
                                "max_cell_chars": 1024,
                            },
                        }
                    ],
                }
            },
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1",
                    "payload": {
                        "status": "GAP",
                        "tool_id": "db.instance.identity",
                    },
                },
            ),
            lease_token="lease-token",
            lease_until=(
                datetime.now(UTC) + timedelta(minutes=1)
            ).isoformat(),
        )

        result = self.run_async(
            DatabaseDiagnosticHandler(
                executor_client=client,
                grant_codec=codec,
                grant_issuer="aiops-worker",
                grant_audience="aiops-db-executor",
                grant_ttl_seconds=30,
            ).execute(context)
        )

        self.assertEqual("GAP", result.status)
        self.assertEqual(1, len(client.calls))
        self.assertEqual("db.storage.capacity", codec.grant.tool_id)

    def test_capability_snapshot_uses_only_enabled_reachable_resources(self) -> None:
        snapshot = build_capability_snapshot(
            agent_id="agent-1",
            agent_version=SimpleNamespace(agent_version_id="version-1"),
            target=SimpleNamespace(
                target_id="target-1",
                db_type="ORACLE",
                version_code="19c",
                status="ENABLED",
                connectivity_status="CONNECTED",
                readonly_connection_enabled=True,
                controlled_change_enabled=False,
                diagnostic_credential_id="credential-1",
                execution_credential_id=None,
                endpoint_json={
                    "host": "db.internal",
                    "port": 1521,
                    "service": "PDB1",
                },
                capabilities_json={
                    "capabilities": ["DB_SQL_STATS"],
                    "privileges": ["ORACLE_SELECT_V_SQLSTATS"],
                },
            ),
            sources=(
                SimpleNamespace(
                    diagnostic_source_id="source-1",
                    source_type="PROMETHEUS",
                    status="ENABLED",
                    connectivity_status="UNREACHABLE",
                    declared_capabilities_json={
                        "capabilities": ["PROMETHEUS_QUERY"]
                    },
                    discovered_capabilities_json={
                        "metric.query_range": {"supported": True}
                    },
                ),
                SimpleNamespace(
                    diagnostic_source_id="source-2",
                    source_type="LOKI",
                    status="DISABLED",
                    connectivity_status="CONNECTED",
                    declared_capabilities_json={
                        "capabilities": ["LOKI_QUERY"]
                    },
                    discovered_capabilities_json=None,
                ),
            ),
        )

        self.assertIn("DB_READONLY", snapshot.target_capabilities)
        self.assertIn(
            "dynamic_performance_views", snapshot.target_capabilities
        )
        self.assertIn("dba_catalog_views", snapshot.target_capabilities)
        self.assertIn("replication_views", snapshot.target_capabilities)
        self.assertIn("DB_SQL_STATS", snapshot.target_capabilities)
        self.assertEqual(
            frozenset({"PROMETHEUS_QUERY", "metric.query_range"}),
            snapshot.available_source_capabilities,
        )
        self.assertFalse(snapshot.source_snapshots[0].reachable)

    def test_registry_hash_is_independent_of_registration_order(self) -> None:
        first = _manifest(
            playbook_id="oracle.sql.current",
            intent=DbaIntent.OBSERVE,
            domain=DbaDomain.SQL_PERFORMANCE,
        )
        second = _manifest(
            playbook_id="oracle.instance.diagnose",
            intent=DbaIntent.DIAGNOSE,
            domain=DbaDomain.INSTANCE_PERFORMANCE,
        )
        left = PlaybookRegistry((first, second))
        right = PlaybookRegistry((second, first))

        self.assertEqual(left.catalog_hash, right.catalog_hash)
        self.assertEqual(
            left.manifest_hash(first.playbook_id, first.version),
            right.manifest_hash(first.playbook_id, first.version),
        )

    def test_registry_rejects_tool_outside_executor_catalog(self) -> None:
        with self.assertRaisesRegex(PlaybookCatalogError, "目录外 Tool"):
            PlaybookRegistry(
                (
                    _manifest(
                        playbook_id="oracle.sql.current",
                        intent=DbaIntent.OBSERVE,
                        domain=DbaDomain.SQL_PERFORMANCE,
                        tool_id="db.dynamic.sql",
                    ),
                ),
                allowed_tools=frozenset(
                    {("db.instance.identity", "1.0.0")}
                ),
            )

    def test_execution_snapshot_rejects_undeclared_tool_privileges(self) -> None:
        diagnostics = DiagnosticRegistry.load()
        manifest = _manifest(
            playbook_id="oracle.identity.invalid",
            intent=DbaIntent.OBSERVE,
            domain=DbaDomain.SQL_PERFORMANCE,
        ).model_copy(update={"required_privileges": ()})
        registry = PlaybookRegistry((manifest,))
        plan = _playbook_plan(registry, (manifest,))
        compiled = InvestigationTaskCompiler(registry).compile(plan)

        with self.assertRaisesRegex(PlaybookCatalogError, "未声明 Tool"):
            ToolExecutionSnapshotBuilder(
                playbook_registry=registry,
                diagnostic_registry=diagnostics,
            ).build(
                plan=plan,
                compiled=compiled,
                capabilities=_capabilities(),
                database_execution={},
            )

    def test_compiler_is_replayable_and_compiles_traceable_tasks(self) -> None:
        manifest = _manifest(
            playbook_id="oracle.sql.current",
            intent=DbaIntent.OBSERVE,
            domain=DbaDomain.SQL_PERFORMANCE,
        )
        registry = PlaybookRegistry((manifest,))
        first = _playbook_plan(registry, (manifest,))
        second = _playbook_plan(registry, (manifest,))
        compiled = InvestigationTaskCompiler(registry).compile(first)

        self.assertEqual(first, second)
        self.assertEqual(
            registry.manifest_hash(manifest.playbook_id, manifest.version),
            first.items[0].manifest_hash,
        )
        self.assertEqual("PLAYBOOK_INVOKE", compiled.tasks[0].task_type)
        self.assertEqual("EVIDENCE_ASSESS", compiled.tasks[-2].task_type)
        self.assertEqual("ANSWER", compiled.tasks[-1].task_type)
        self.assertEqual(
            compiled.invocation_task_keys,
            compiled.tasks[-2].input_artifact_keys,
        )
        self.assertEqual(
            ("evidence:assess",),
            compiled.tasks[-1].input_artifact_keys,
        )

    def test_compiler_rejects_changed_playbook_catalog(self) -> None:
        manifest = _manifest(
            playbook_id="oracle.sql.window",
            intent=DbaIntent.OBSERVE,
            domain=DbaDomain.SQL_PERFORMANCE,
        )
        registry = PlaybookRegistry((manifest,))
        stale_plan = _playbook_plan(registry, (manifest,)).model_copy(
            update={"catalog_hash": "0" * 64}
        )

        with self.assertRaises(InvestigationCatalogChangedError):
            InvestigationTaskCompiler(registry).compile(stale_plan)


if __name__ == "__main__":
    unittest.main()
