"""Agent Runtime 事务命令内核测试。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from agent_runtime.application import (
    AppendTaskProgressCommand,
    AgentRuntimeConflict,
    AgentRuntimeService,
    ArtifactInput,
    ClaimTaskCommand,
    CompleteTaskCommand,
    CreateRunCommand,
    FailTaskCommand,
    InstallPlanCommand,
    StaleTaskLease,
)
from agent_runtime.domain.planning import (
    ExecutionKind,
    ExecutionMode,
    PlanDraft,
    PlanValidator,
    TaskSpec,
)
from agent_runtime.domain.skills import SkillRegistry
from agent_runtime.specialists import register_builtin_manifests
from agent_runtime.specialists.root import RootAgentPlanner
from platform_core.identity import uuid7
from platform_core.contracts import AgentExecutionSpec


class _ModelResolver:
    _NAMES = {
        "router_llm": "router-model",
        "context_llm": "context-model",
        "composer_llm": "composer-model",
        "memory_llm": "memory-model",
        "query_vlm": "query-vlm-model",
        "memory_embedding": "memory-embedding-model",
        "future_reasoner": "reasoner-model",
    }

    async def resolve(self, models, *, roles=None):
        return {
            role: {
                "model_id": str(model_id),
                "served_model_name": self._NAMES[role],
                "category": 2 if role == "memory_embedding" else 1,
                "config_fingerprint": "a" * 64,
            }
            for role, model_id in models.items()
            if roles is None or role in roles
        }


class _NoopNotificationPublisher:
    async def publish(self, **kwargs):
        del kwargs


class _Store:
    def __init__(self):
        self.agents = {}
        self.runs = {}
        self.tasks = {}
        self.artifacts = {}
        self.events = []
        self.delegations = {}


class _Agents:
    def __init__(self, store):
        self.store = store

    async def get_active(self, *, agent_id, domain_id):
        agent = self.store.agents.get(agent_id)
        if (
            agent is None
            or int(agent.domain_id) != domain_id
            or agent.status != "ACTIVE"
        ):
            return None
        return agent

    async def add(self, entity):
        entity.agent_id = entity.agent_id or uuid7()
        entity.row_version = entity.row_version or 1
        self.store.agents[entity.agent_id] = entity
        return entity

    async def get_scoped(
        self, *, agent_id, domain_id, lock=False
    ):
        agent = self.store.agents.get(agent_id)
        if (
            agent is None
            or int(agent.domain_id) != domain_id
        ):
            return None
        return agent

    async def list_scoped(self, *, domain_id):
        return [
            agent
            for agent in self.store.agents.values()
            if int(agent.domain_id) == domain_id
        ]


class _Runs:
    def __init__(self, store):
        self.store = store

    async def add(self, entity):
        entity.run_id = entity.run_id or uuid7()
        entity.row_version = entity.row_version or 1
        entity.created_at = entity.created_at or datetime.now(timezone.utc)
        entity.updated_at = entity.updated_at or entity.created_at
        self.store.runs[entity.run_id] = entity
        return entity

    async def get_by_idempotency(
        self, *, domain_id, actor_id, idempotency_key, lock=False
    ):
        return next(
            (
                run
                for run in self.store.runs.values()
                if int(run.domain_id) == domain_id
                and run.actor_id == actor_id
                and run.idempotency_key == idempotency_key
            ),
            None,
        )

    async def get_scoped(
        self, *, run_id, domain_id, lock=False
    ):
        run = self.store.runs.get(run_id)
        if (
            run is None
            or int(run.domain_id) != domain_id
        ):
            return None
        return run

    async def get(self, *, run_id, lock=False):
        return self.store.runs.get(run_id)


class _Tasks:
    def __init__(self, store):
        self.store = store

    async def add_all(self, entities):
        now = datetime.now(timezone.utc)
        for entity in entities:
            entity.task_id = entity.task_id or uuid7()
            entity.row_version = entity.row_version or 1
            entity.attempt = entity.attempt or 0
            entity.created_at = entity.created_at or now
            entity.updated_at = entity.updated_at or now
            self.store.tasks[entity.task_id] = entity
        return entities

    async def claim_candidate(self, *, now, max_parallel_tasks):
        for task in self.store.tasks.values():
            if task.status != "READY":
                continue
            running = sum(
                1
                for other in self.store.tasks.values()
                if other.run_id == task.run_id
                and other.status == "RUNNING"
            )
            if running < max_parallel_tasks:
                return task
        return None

    async def claim_due_retry(self, *, now):
        return next(
            (
                task
                for task in self.store.tasks.values()
                if task.status == "RETRY_WAIT"
                and task.next_retry_at is not None
                and task.next_retry_at <= now
            ),
            None,
        )

    async def claim_expired_lease(self, *, now):
        return next(
            (
                task
                for task in self.store.tasks.values()
                if task.status == "RUNNING"
                and task.lease_until is not None
                and task.lease_until <= now
            ),
            None,
        )

    async def get(self, *, task_id, lock=False):
        return self.store.tasks.get(task_id)

    async def list_by_run(self, *, run_id, lock=False):
        return [
            task
            for task in self.store.tasks.values()
            if task.run_id == run_id
        ]


class _Artifacts:
    def __init__(self, store):
        self.store = store

    async def add(self, entity):
        entity.artifact_id = entity.artifact_id or uuid7()
        entity.created_at = entity.created_at or datetime.now(timezone.utc)
        self.store.artifacts[entity.artifact_id] = entity
        return entity

    async def get(self, *, artifact_id):
        return self.store.artifacts.get(artifact_id)

    async def list_by_task_ids(self, *, task_ids):
        selected = set(task_ids)
        return [
            artifact
            for artifact in self.store.artifacts.values()
            if artifact.task_id in selected
        ]


class _Events:
    def __init__(self, store):
        self.store = store

    async def add(self, entity):
        entity.created_at = entity.created_at or datetime.now(timezone.utc)
        self.store.events.append(entity)
        return entity

    async def get_by_key(self, *, run_id, event_key):
        return next(
            (
                event
                for event in self.store.events
                if event.run_id == run_id and event.event_key == event_key
            ),
            None,
        )

    async def next_sequence(self, *, run_id):
        return (
            max(
                (
                    int(event.sequence_no)
                    for event in self.store.events
                    if event.run_id == run_id
                ),
                default=0,
            )
            + 1
        )

    async def latest_sequence(self, *, run_id):
        return max(
            (
                int(event.sequence_no)
                for event in self.store.events
                if event.run_id == run_id
            ),
            default=0,
        )

    async def list_after(self, *, run_id, after_sequence, limit=200):
        return [
            event
            for event in self.store.events
            if event.run_id == run_id
            and int(event.sequence_no) > after_sequence
        ][:limit]


class _Delegations:
    def __init__(self, store):
        self.store = store

    async def add(self, entity):
        entity.row_version = entity.row_version or 1
        self.store.delegations[entity.delegation_id] = entity
        return entity

    async def get_by_task(self, *, parent_task_id, lock=False):
        return next(
            (
                delegation
                for delegation in self.store.delegations.values()
                if delegation.parent_task_id == parent_task_id
            ),
            None,
        )


class _Uow:
    def __init__(self, store):
        self.agents = _Agents(store)
        self.runs = _Runs(store)
        self.tasks = _Tasks(store)
        self.artifacts = _Artifacts(store)
        self.events = _Events(store)
        self.delegations = _Delegations(store)
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self):
        self.committed = True


class AgentRuntimeServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = _Store()
        self.model_resolver = _ModelResolver()
        self.agent_id = uuid7()
        self.store.agents[self.agent_id] = SimpleNamespace(
            agent_id=self.agent_id,
            domain_id=20,
            display_name="文档助手",
            description=None,
            status="ACTIVE",
            enabled_capabilities_json=["document"],
            models_json={
                "router_llm": str(uuid7()),
                "context_llm": str(uuid7()),
                "composer_llm": str(uuid7()),
                "memory_llm": str(uuid7()),
                "query_vlm": str(uuid7()),
                "memory_embedding": str(uuid7()),
            },
            data_profile_name=None,
            instruction="仅基于可验证证据回答。",
            config_json={"answer_language": "zh-CN"},
            row_version=1,
        )
        registry = register_builtin_manifests(SkillRegistry())
        self.registry = registry
        validator = PlanValidator(
            skill_exists=registry.contains,
            capability_exists=lambda service, capability: False,
            public_artifact_types={"GROUNDED_ANSWER"},
        )
        self.service = AgentRuntimeService(
            uow_factory=lambda: _Uow(self.store),
            plan_validator=validator,
            skill_registry=registry,
            model_resolver=self.model_resolver,
            notification_publisher=_NoopNotificationPublisher(),
        )
        self.execution_spec = AgentExecutionSpec(
            schema_version="1.0",
            owner_app_id="knowledge_retrieval",
            domain_id=20,
            consumer_agent_id=self.agent_id,
            consumer_agent_version_id=uuid7(),
            agent_kind="KNOWLEDGE_RETRIEVAL",
            display_name="文档助手",
            enabled_capabilities=("document",),
            models={
                "router_llm": uuid7(),
                "context_llm": uuid7(),
                "composer_llm": uuid7(),
                "memory_llm": uuid7(),
                "query_vlm": uuid7(),
                "memory_embedding": uuid7(),
            },
            instruction="仅基于可验证证据回答。",
            resource_context={"answer_language": "zh-CN"},
        )
        self.create_command = CreateRunCommand(
            domain_id=20,
            agent_id=self.agent_id,
            execution_spec=self.execution_spec,
            actor_id="user-1",
            request_id="request-1",
            trace_id="trace-1",
            idempotency_key="create-1",
            original_input="总结上传文档",
        )

    async def test_create_run_atomically_installs_document_plan(self):
        service = AgentRuntimeService(
            uow_factory=lambda: _Uow(self.store),
            plan_validator=PlanValidator(
                skill_exists=self.registry.contains,
                capability_exists=lambda service, capability: False,
                public_artifact_types={"GROUNDED_ANSWER"},
            ),
            skill_registry=self.registry,
            root_planner=RootAgentPlanner(),
            model_resolver=self.model_resolver,
            notification_publisher=_NoopNotificationPublisher(),
        )
        created = await service.create_run(self.create_command)

        self.assertEqual(created.status, "RUNNING")
        self.assertEqual(len(self.store.tasks), 3)
        self.assertEqual(
            [event.event_type for event in self.store.events],
            ["RUN_CREATED", "RUN_STARTED"],
        )
        run = self.store.runs[created.run_id]
        self.assertIsNotNone(run.final_task_id)

    async def test_create_run_rejects_aiops_agent(self):
        aiops_spec = self.execution_spec.model_copy(
            update={
                "owner_app_id": "aiops",
                "agent_kind": "AIOPS",
                "enabled_capabilities": ("aiops",),
            }
        )
        service = AgentRuntimeService(
            uow_factory=lambda: _Uow(self.store),
            plan_validator=PlanValidator(
                skill_exists=self.registry.contains,
                capability_exists=lambda service, capability: (
                    service == "aiops_agent"
                    and capability == "diagnosis"
                ),
                public_artifact_types={"GROUNDED_ANSWER"},
            ),
            skill_registry=self.registry,
            root_planner=RootAgentPlanner(),
            model_resolver=self.model_resolver,
            notification_publisher=_NoopNotificationPublisher(),
        )
        with self.assertRaises(AgentRuntimeConflict) as raised:
            await service.create_run(
                self.create_command.model_copy(
                    update={
                        "idempotency_key": "create-aiops",
                        "original_input": "分析数据库性能问题",
                        "execution_spec": aiops_spec,
                    }
                )
            )

        self.assertEqual(raised.exception.code, "AGENT_KIND_UNSUPPORTED")

    async def test_create_run_freezes_agent_configuration(self):
        receipt = await self.service.create_run(self.create_command)
        snapshot = self.store.runs[receipt.run_id].config_snapshot_json

        self.assertEqual(snapshot["language"], "zh-CN")
        self.assertEqual(
            snapshot["agent"]["display_name"], "文档助手"
        )
        self.assertEqual(
            snapshot["agent"]["models"]["composer_llm"][
                "served_model_name"
            ],
            "composer-model",
        )
        self.assertEqual(
            snapshot["agent"]["models"]["memory_llm"][
                "served_model_name"
            ],
            "memory-model",
        )
        self.assertEqual(snapshot["retrieval"]["collection_ids"], [])

    async def test_runtime_preserves_app_owned_capabilities(self):
        app_spec = self.execution_spec.model_copy(
            update={
                "owner_app_id": "km_asset",
                "enabled_capabilities": (
                    "conversation", "document", "data_query"
                ),
            }
        )
        receipt = await self.service.create_run(
            self.create_command.model_copy(
                update={
                    "idempotency_key": "create-app-capabilities",
                    "execution_spec": app_spec,
                }
            )
        )

        snapshot = self.store.runs[receipt.run_id].config_snapshot_json
        self.assertEqual(
            ["conversation", "document", "data_query"],
            snapshot["agent"]["enabled_capabilities"],
        )

    async def test_create_run_is_idempotent_and_checks_fingerprint(self):
        first = await self.service.create_run(self.create_command)
        second = await self.service.create_run(self.create_command)

        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(len(self.store.runs), 1)
        self.assertEqual(len(self.store.events), 1)

        changed = self.create_command.model_copy(
            update={"original_input": "不同问题"}
        )
        with self.assertRaises(AgentRuntimeConflict) as caught:
            await self.service.create_run(changed)
        self.assertEqual(caught.exception.code, "IDEMPOTENCY_CONFLICT")

    async def test_plan_claim_complete_is_one_persisted_state_flow(self):
        created = await self.service.create_run(self.create_command)
        plan = PlanDraft(
            plan_version="1.0",
            objective="生成最终回答",
            tasks=(
                TaskSpec(
                    task_key="compose",
                    task_type="COMPOSE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="response_composer",
                    skill_id="response-composer",
                    skill_version="1.0.0",
                    expected_outputs=("GROUNDED_ANSWER",),
                    timeout_seconds=60,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
            ),
            final_task_key="compose",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        started = await self.service.install_plan(
            InstallPlanCommand(
                domain_id=20,
                run_id=created.run_id,
                expected_row_version=1,
                plan=plan,
                actor_id="root-agent",
                trace_id="trace-1",
                idempotency_key="plan-1",
            )
        )
        self.assertEqual(started.status, "RUNNING")

        lease = await self.service.claim_task(
            ClaimTaskCommand(
                worker_id="worker-1",
                lease_seconds=120,
                trace_id="trace-1",
            )
        )
        self.assertIsNotNone(lease)
        leased_task = self.store.tasks[lease.task_id]
        leased_task.error_code = "WORKER_LEASE_EXPIRED"
        leased_task.error_message = "历史租约已过期"
        await self.service.append_task_progress(
            AppendTaskProgressCommand(
                task_id=lease.task_id,
                worker_id="worker-1",
                lease_token=lease.lease_token,
                event_type="answer.delta",
                payload={"chunk_index": 1, "delta": "完成"},
                actor_id="worker-1",
                trace_id="trace-1",
                idempotency_key="delta-1",
            )
        )
        completed = await self.service.complete_task(
            CompleteTaskCommand(
                task_id=lease.task_id,
                expected_row_version=lease.row_version,
                worker_id="worker-1",
                lease_token=lease.lease_token,
                artifact=ArtifactInput(
                    artifact_type="GROUNDED_ANSWER",
                    schema_version="GroundedAnswer.v1",
                    producer="response-composer",
                    producer_version="1.0.0",
                    payload={"answer": "完成"},
                ),
                actor_id="worker-1",
                trace_id="trace-1",
                idempotency_key="complete-1",
            )
        )

        self.assertEqual(completed.task_status, "SUCCEEDED")
        self.assertEqual(completed.run_status, "COMPLETED")
        self.assertIsNotNone(completed.artifact_id)
        self.assertIsNone(leased_task.error_code)
        self.assertIsNone(leased_task.error_message)
        result = await self.service.get_result(
            run_id=created.run_id,
            domain_id=20,
        )
        self.assertEqual(result.payload, {"answer": "完成"})
        self.assertEqual(
            [event.event_type for event in self.store.events],
            [
                "RUN_CREATED",
                "RUN_STARTED",
                "TASK_STARTED",
                "skill.started",
                "answer.delta",
                "ARTIFACT_CREATED",
                "answer.completed",
                "TASK_COMPLETED",
                "RUN_COMPLETED",
            ],
        )

    async def test_old_worker_cannot_complete_with_stale_lease(self):
        await self.test_plan_claim_complete_is_one_persisted_state_flow()
        task = next(iter(self.store.tasks.values()))
        with self.assertRaises(StaleTaskLease):
            await self.service.complete_task(
                CompleteTaskCommand(
                    task_id=task.task_id,
                    expected_row_version=int(task.row_version),
                    worker_id="old-worker",
                    lease_token=uuid7(),
                    artifact=ArtifactInput(
                        artifact_type="GROUNDED_ANSWER",
                        schema_version="GroundedAnswer.v1",
                        producer="response-composer",
                        producer_version="1.0.0",
                        payload={"answer": "重复"},
                    ),
                    actor_id="old-worker",
                    trace_id="trace-2",
                    idempotency_key="complete-stale",
                )
            )

    async def test_due_retry_returns_to_ready_and_gets_new_lease(self):
        created = await self.service.create_run(self.create_command)
        plan = PlanDraft(
            plan_version="1.0",
            objective="生成最终回答",
            tasks=(
                TaskSpec(
                    task_key="compose",
                    task_type="COMPOSE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="response_composer",
                    skill_id="response-composer",
                    skill_version="1.0.0",
                    expected_outputs=("GROUNDED_ANSWER",),
                    timeout_seconds=60,
                    max_retries=1,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
            ),
            final_task_key="compose",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        await self.service.install_plan(
            InstallPlanCommand(
                domain_id=20,
                run_id=created.run_id,
                expected_row_version=1,
                plan=plan,
                actor_id="root-agent",
                trace_id="trace-1",
                idempotency_key="retry-plan",
            )
        )
        first = await self.service.claim_task(
            ClaimTaskCommand(
                worker_id="worker-1",
                lease_seconds=120,
                trace_id="trace-1",
            )
        )
        await self.service.fail_task(
            FailTaskCommand(
                task_id=first.task_id,
                expected_row_version=first.row_version,
                worker_id="worker-1",
                lease_token=first.lease_token,
                error_code="TEMPORARY_ERROR",
                error_message="临时失败",
                retryable=True,
                retry_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                actor_id="worker-1",
                trace_id="trace-1",
                idempotency_key="retry-fail",
            )
        )

        second = await self.service.claim_task(
            ClaimTaskCommand(
                worker_id="worker-2",
                lease_seconds=120,
                trace_id="trace-2",
            )
        )
        self.assertEqual(second.task_id, first.task_id)
        self.assertEqual(second.attempt, 2)
        self.assertNotEqual(second.lease_token, first.lease_token)

    async def test_expired_final_lease_fails_required_run(self):
        await self.test_due_retry_returns_to_ready_and_gets_new_lease()
        task = next(iter(self.store.tasks.values()))
        task.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)

        lease = await self.service.claim_task(
            ClaimTaskCommand(
                worker_id="worker-3",
                lease_seconds=120,
                trace_id="trace-3",
            )
        )

        self.assertIsNone(lease)
        self.assertEqual(task.status, "FAILED")
        run = next(iter(self.store.runs.values()))
        self.assertEqual(run.status, "FAILED")
        self.assertEqual(run.error_code, "WORKER_LEASE_EXPIRED")
        summary = await self.service.get_run(
            run_id=run.run_id, domain_id=run.domain_id
        )
        self.assertEqual(
            "Task 租约过期且已达到最大尝试次数",
            summary.error_message,
        )

    async def test_dependency_artifact_is_in_successor_lease(self):
        created = await self.service.create_run(self.create_command)
        planner = RootAgentPlanner()
        decision = planner.decide(
            agent_snapshot={
                "enabled_capabilities": ["document"],
                "config": {},
            }
        )
        await self.service.install_plan(
            InstallPlanCommand(
                domain_id=20,
                run_id=created.run_id,
                expected_row_version=1,
                plan=planner.build_plan(
                    objective="总结上传文档",
                    decision=decision,
                ),
                actor_id="root-agent",
                trace_id="trace-1",
                idempotency_key="document-plan-1",
            )
        )
        rewrite_lease = await self.service.claim_task(
            ClaimTaskCommand(
                worker_id="worker-1",
                lease_seconds=120,
                trace_id="trace-1",
            )
        )
        await self.service.complete_task(
            CompleteTaskCommand(
                task_id=rewrite_lease.task_id,
                expected_row_version=rewrite_lease.row_version,
                worker_id="worker-1",
                lease_token=rewrite_lease.lease_token,
                artifact=ArtifactInput(
                    artifact_type="CONTEXT_REWRITE",
                    schema_version="ContextRewriteOutput.v1",
                    producer="context-rewrite",
                    producer_version="1.0.0",
                    payload={
                        "raw_input": "总结上传文档",
                        "standalone_query": "总结上传文档",
                        "retrieval_queries": ["总结上传文档"],
                        "resolved_references": [],
                        "active_topic": None,
                        "ambiguity": False,
                        "clarification_question": None,
                        "memory_refs": [],
                    },
                ),
                actor_id="worker-1",
                trace_id="trace-1",
                idempotency_key="rewrite-complete",
            )
        )
        retrieval_lease = await self.service.claim_task(
            ClaimTaskCommand(
                worker_id="worker-1",
                lease_seconds=120,
                trace_id="trace-1",
            )
        )
        await self.service.complete_task(
            CompleteTaskCommand(
                task_id=retrieval_lease.task_id,
                expected_row_version=retrieval_lease.row_version,
                worker_id="worker-1",
                lease_token=retrieval_lease.lease_token,
                artifact=ArtifactInput(
                    artifact_type="CITATION_PACK",
                    schema_version="DocumentRetrievalResult.v1",
                    producer="knowledge-retrieval",
                    producer_version="1.0.0",
                    payload={"citation_pack": {"citations": []}},
                ),
                actor_id="worker-1",
                trace_id="trace-1",
                idempotency_key="retrieval-complete",
            )
        )

        compose_lease = await self.service.claim_task(
            ClaimTaskCommand(
                worker_id="worker-1",
                lease_seconds=120,
                trace_id="trace-1",
            )
        )
        self.assertEqual(compose_lease.task_key, "response_compose")
        self.assertEqual(len(compose_lease.input_artifacts), 2)
        self.assertEqual(
            {
                item.artifact_type
                for item in compose_lease.input_artifacts
            },
            {"CONTEXT_REWRITE", "CITATION_PACK"},
        )


if __name__ == "__main__":
    unittest.main()
