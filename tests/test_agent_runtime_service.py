"""Agent Runtime 事务命令内核测试。"""

from datetime import datetime, timedelta, timezone
import unittest

from agent_runtime.application import (
    AgentRuntimeConflict,
    AgentRuntimeService,
    ArtifactInput,
    ClaimTaskCommand,
    CompleteTaskCommand,
    CreateRunCommand,
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
from agent_runtime.domain.skills import (
    ArtifactDeclaration,
    DataClassification,
    SkillManifest,
    SkillRegistry,
)
from platform_core.identity import uuid7


class _Store:
    def __init__(self):
        self.runs = {}
        self.tasks = {}
        self.artifacts = {}
        self.events = []


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
        self, *, app_id, domain_id, actor_id, idempotency_key, lock=False
    ):
        return next(
            (
                run
                for run in self.store.runs.values()
                if int(run.app_id) == app_id
                and int(run.domain_id) == domain_id
                and run.actor_id == actor_id
                and run.idempotency_key == idempotency_key
            ),
            None,
        )

    async def get_scoped(
        self, *, run_id, app_id, domain_id, lock=False
    ):
        run = self.store.runs.get(run_id)
        if (
            run is None
            or int(run.app_id) != app_id
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


class _Uow:
    def __init__(self, store):
        self.runs = _Runs(store)
        self.tasks = _Tasks(store)
        self.artifacts = _Artifacts(store)
        self.events = _Events(store)
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
        registry = SkillRegistry()
        registry.register(
            SkillManifest(
                skill_id="response-composer",
                version="1.0.0",
                owner="agent-runtime",
                specialist="response_composer",
                description="组合最终回答",
                input_schema="ComposeInput.v1",
                output_artifacts=(
                    ArtifactDeclaration(
                        artifact_type="FINAL_RESPONSE",
                        schema_version="1.0",
                    ),
                ),
                execution_mode=ExecutionMode.READ_ONLY,
                idempotent=True,
                timeout_seconds=60,
                data_classification=DataClassification.INTERNAL,
            ),
            lambda: None,
        )
        validator = PlanValidator(
            skill_exists=registry.contains,
            capability_exists=lambda service, capability: False,
            public_artifact_types={"FINAL_RESPONSE"},
        )
        self.service = AgentRuntimeService(
            uow_factory=lambda: _Uow(self.store),
            plan_validator=validator,
            skill_registry=registry,
        )
        self.create_command = CreateRunCommand(
            app_id=1,
            domain_id=20,
            agent_id=uuid7(),
            actor_id="user-1",
            request_id="request-1",
            trace_id="trace-1",
            idempotency_key="create-1",
            original_input="总结上传文档",
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
                    expected_outputs=("FINAL_RESPONSE",),
                    timeout_seconds=60,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
            ),
            final_task_key="compose",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        started = await self.service.install_plan(
            InstallPlanCommand(
                app_id=1,
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
        completed = await self.service.complete_task(
            CompleteTaskCommand(
                task_id=lease.task_id,
                expected_row_version=lease.row_version,
                worker_id="worker-1",
                lease_token=lease.lease_token,
                artifact=ArtifactInput(
                    artifact_type="FINAL_RESPONSE",
                    schema_version="1.0",
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
        self.assertEqual(
            [event.event_type for event in self.store.events],
            [
                "RUN_CREATED",
                "RUN_STARTED",
                "TASK_STARTED",
                "ARTIFACT_CREATED",
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
                        artifact_type="FINAL_RESPONSE",
                        schema_version="1.0",
                        producer="response-composer",
                        producer_version="1.0.0",
                        payload={"answer": "重复"},
                    ),
                    actor_id="old-worker",
                    trace_id="trace-2",
                    idempotency_key="complete-stale",
                )
            )


if __name__ == "__main__":
    unittest.main()
