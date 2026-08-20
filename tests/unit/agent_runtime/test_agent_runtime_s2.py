"""Agent Runtime 第二阶段的记忆、Hybrid 与通知边界测试。"""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from agent_runtime.application.commands import LeasedArtifact
from agent_runtime.application.conversations import ConversationService
from agent_runtime.application.memory import (
    MemoryConsolidationWorker,
    MemoryJobLease,
)
from agent_runtime.application.runtime_service import (
    AgentRuntimeConflict,
    AgentRuntimeService,
)
from agent_runtime.domain.memory_policy import memory_scope
from agent_runtime.domain.planning import PlanLimits, PlanValidator
from agent_runtime.domain.skills import SkillRegistry
from agent_runtime.runtime import ExecutionContext
from agent_runtime.specialists.hybrid import (
    DataConstraintExtractSkill,
    DocumentScopeExtractSkill,
)
from agent_runtime.specialists.data_query import QueryResult
from agent_runtime.specialists.root import (
    RootAgentPlanner,
    RouteDecision,
    RouteType,
)
from agent_runtime.specialists.registry import register_builtin_manifests
from platform_core.identity import uuid7


class _PromptResolver:
    async def resolve(self, key):
        return SimpleNamespace(
            content=f"prompt:{key}",
            ref=lambda: {"prompt_key": key, "version": "1.0.0"},
        )


class _ModelClient:
    def __init__(self, response):
        self.response = response

    async def get_llm_json(self, **kwargs):
        del kwargs
        return self.response


class _NotificationPublisher:
    def __init__(self):
        self.calls = []

    async def publish(self, **kwargs):
        self.calls.append(kwargs)


def _context() -> ExecutionContext:
    return ExecutionContext(
        domain_id=20,
        agent_id=uuid7(),
        run_id=uuid7(),
        task_id=uuid7(),
        task_key="hybrid",
        actor_id="user-1",
        request_id="request-1",
        trace_id="trace-1",
        original_input="结合制度和销售数据判断是否达标",
        config_snapshot={
            "agent": {
                "models": {
                    "composer_llm": {"served_model_name": "composer"}
                }
            }
        },
        policy_snapshot={},
        input_artifacts=(),
    )


class AgentRuntimeS2Test(unittest.IsolatedAsyncioTestCase):
    def test_only_strict_low_sensitivity_preferences_are_shared(self):
        self.assertEqual(
            "USER_SHARED",
            memory_scope(
                "user.preference.response_language",
                {"language": "zh-CN"},
            ),
        )
        self.assertEqual(
            "USER_AGENT",
            memory_scope(
                "user.preference.response_language",
                {"language": "zh-CN", "email": "a@example.com"},
            ),
        )
        self.assertEqual(
            "USER_AGENT",
            memory_scope("business.customer", {"name": "张三"}),
        )

    def test_all_hybrid_routes_build_valid_dependency_graphs(self):
        planner = RootAgentPlanner()
        registry = register_builtin_manifests(SkillRegistry())
        validator = PlanValidator(
            skill_exists=registry.contains,
            capability_exists=lambda service, capability: False,
            public_artifact_types={"GROUNDED_ANSWER"},
        )
        expected = {
            RouteType.HYBRID_PARALLEL: "hybrid-parallel-plan-v1",
            RouteType.HYBRID_DOCUMENT_FIRST: "hybrid-document-first-plan-v1",
            RouteType.HYBRID_DATA_FIRST: "hybrid-data-first-plan-v1",
        }
        for route, version in expected.items():
            plan = planner.build_plan(
                objective="联合分析",
                decision=RouteDecision(
                    route_type=route,
                    confidence=1,
                    reason="test",
                ),
            )
            self.assertEqual(version, plan.plan_version)
            validator.validate(plan, PlanLimits())
            keys = {item.task_key for item in plan.tasks}
            self.assertEqual(len(keys), len(plan.tasks))
            self.assertEqual("response_compose", plan.final_task_key)
            for task in plan.tasks:
                self.assertTrue(set(task.depends_on).issubset(keys))
            if route == RouteType.HYBRID_DATA_FIRST:
                compose = next(
                    item for item in plan.tasks
                    if item.task_key == "response_compose"
                )
                self.assertIn("document_scope", compose.depends_on)
                self.assertIn(
                    "task_output:document_scope", compose.input_refs
                )

    async def test_hybrid_extractors_reject_uncontrolled_output(self):
        valid = await DataConstraintExtractSkill(
            model_client=_ModelClient({"metrics": ["销售额"]}),
            prompt_resolver=_PromptResolver(),
        ).execute(_context())
        self.assertEqual(valid.artifact.artifact_type, "DATA_QUERY_CONSTRAINTS")

        with self.assertRaisesRegex(ValueError, "UNEXPECTED_FIELDS"):
            await DataConstraintExtractSkill(
                model_client=_ModelClient({"sql": "SELECT * FROM secret"}),
                prompt_resolver=_PromptResolver(),
            ).execute(_context())

    async def test_km_enumeration_scope_uses_query_result_and_caps_at_ten(self):
        rows = tuple({
            "asset_id": f"asset-{index}",
            "title": f"Asset {index:02d}",
            "product": "OAC",
            "solution": "Analytics",
            "bundle_id": str(uuid7()),
            "bundle_revision_id": str(uuid7()),
            "asset_count": 1,
        } for index in range(12))
        query = QueryResult(
            query_result_id=uuid7(),
            provider="SEMANTIC",
            columns=(),
            rows=rows,
            row_count=12,
            truncated=False,
            provenance={},
        )
        artifact = LeasedArtifact(
            artifact_id=uuid7(),
            artifact_type="QUERY_RESULT",
            schema_version="QUERY_RESULT.v1",
            producer="data-query",
            producer_version="1.0.0",
            payload=query.model_dump(mode="json"),
            content_hash="hash",
            security_level=0,
        )
        context = _context().model_copy(update={
            "original_input": "列出关于 OAC 的 asset",
            "config_snapshot": {
                "route": {
                    "answer_basis": "SEMANTIC_RELEVANCE_ENUMERATION"
                },
                "agent": {
                    "models": {
                        "composer_llm": {"served_model_name": "composer"}
                    }
                },
            },
            "input_artifacts": (artifact,),
        })

        result = await DocumentScopeExtractSkill(
            model_client=_ModelClient({"sql": "不应调用模型"}),
            prompt_resolver=_PromptResolver(),
        ).execute(context)

        payload = result.artifact.payload
        self.assertEqual(12, payload["total_count"])
        self.assertEqual(10, len(payload["assets"]))
        self.assertEqual(10, len(payload["bundle_targets"]))
        self.assertTrue(payload["truncated"])
        with self.assertRaisesRegex(ValueError, "QUERY_INVALID"):
            await DocumentScopeExtractSkill(
                model_client=_ModelClient({"query": "x" * 513}),
                prompt_resolver=_PromptResolver(),
            ).execute(_context())

    async def test_memory_worker_failure_only_updates_async_job(self):
        job = SimpleNamespace(
            status="PROCESSING",
            lease_token=uuid7(),
            attempt_count=1,
            max_attempts=3,
            next_attempt_at=None,
            error_code=None,
            error_message=None,
            lease_owner="worker-1",
            lease_until=datetime.now(timezone.utc),
        )

        class _Uow:
            def __init__(self):
                self.memory_jobs = SimpleNamespace(
                    get=AsyncMock(return_value=job)
                )
                self.commit = AsyncMock()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback

        worker = MemoryConsolidationWorker(
            uow_factory=_Uow,
            model_client=None,
            prompt_resolver=None,
            worker_id="worker-1",
            poll_interval_seconds=1,
        )
        lease = MemoryJobLease(
            job_id=uuid7(),
            lease_token=job.lease_token,
            attempt_count=1,
            max_attempts=3,
            conversation_id=uuid7(),
            turn_id=uuid7(),
            turn_sequence=1,
            domain_id=20,
            actor_id="user-1",
            agent_id=uuid7(),
            user_item_id=uuid7(),
            user_message="问题",
            assistant_message={"text": "回答已完成"},
            previous_summary={},
            existing_memories=(),
        )
        await worker._fail(lease, RuntimeError("模型暂时不可用"))

        self.assertEqual("RETRY_WAIT", job.status)
        self.assertEqual("回答已完成", lease.assistant_message["text"])

    async def test_forgotten_memory_is_excluded_from_later_recall(self):
        memory_id = uuid7()
        row = SimpleNamespace(
            memory_id=memory_id,
            status="ACTIVE",
            valid_to=None,
            row_version=1,
        )

        class _MemoryItems:
            async def get_scoped(self, **kwargs):
                del kwargs
                return row

            async def list_active(self, **kwargs):
                del kwargs
                return [row] if row.status == "ACTIVE" else []

        class _Uow:
            def __init__(self):
                self.memory_items = _MemoryItems()
                self.memory_sources = SimpleNamespace(
                    delete_by_memory=AsyncMock()
                )
                self.commit = AsyncMock()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback

        service = ConversationService(
            uow_factory=_Uow,
            runtime_service=None,
        )
        await service.forget_memory(
            memory_id=memory_id,
            domain_id=20,
            actor_id="user-1",
        )
        recalled = await service.list_memories(
            domain_id=20,
            actor_id="user-1",
            agent_id=uuid7(),
            limit=20,
        )

        self.assertEqual("DELETED", row.status)
        self.assertEqual([], recalled)

    async def test_notification_publisher_is_required(self):
        with self.assertRaisesRegex(ValueError, "通知 Outbox Publisher"):
            AgentRuntimeService(uow_factory=None, notification_publisher=None)

    async def test_terminal_run_event_reaches_notification_boundary(self):
        publisher = _NotificationPublisher()

        class _Events:
            async def next_sequence(self, **kwargs):
                del kwargs
                return 1

            async def add(self, event):
                return event

        service = AgentRuntimeService(
            uow_factory=None,
            notification_publisher=publisher,
        )
        await service._append_event(
            SimpleNamespace(events=_Events()),
            run=SimpleNamespace(run_id=uuid7()),
            event_type="RUN_COMPLETED",
            event_key="run-completed",
            actor_type="SYSTEM",
            actor_id="runtime",
            trace_id="trace-1",
            payload={"status": "COMPLETED"},
        )

        self.assertEqual(1, len(publisher.calls))
        self.assertEqual(
            "agent.run.completed",
            publisher.calls[0]["event_type"],
        )


if __name__ == "__main__":
    unittest.main()
