"""正式知识 Runtime 的离线业务围栏和规划测试；没有网络、实库或真实模型。"""

from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import UUID

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [str(path) for path in ROOT.glob("services/*/src")]
from agent_runtime.application.knowledge_query import KnowledgeQueryPlan, plan_knowledge_query
from agent_runtime.application.runtime_service import AgentRuntimeService, AgentRuntimeNotFound
from knowledge_algorithms.contracts.errors import AlgorithmError
from platform_core.contracts.knowledge_app import KnowledgeLibraryView, KnowledgeSearchResult

ORACLE = (ROOT / "database/oracle").exists()
ID = UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a11")


class RuntimeBusinessTests(unittest.IsolatedAsyncioTestCase):
    def query_plan(self, **changes):
        return KnowledgeQueryPlan.model_validate({
            "mode": "RELEVANCE", "requested_count": 27, "output_types": ["TEXT", "FIGURE"],
            "text": "图1-1的引用文字", "combination": None, "predicates": [], "order_by": [], **changes})

    async def test_query_plan_preserves_explicit_count_and_modalities(self):
        plan = self.query_plan()
        query = plan.compile(())
        self.assertEqual(query.requested_count, 27)
        self.assertEqual(query.output_types, ("TEXT", "FIGURE"))
        self.assertEqual(query.text, "图1-1的引用文字")

    async def test_query_plan_preserves_order_and_appends_stable_tiebreak(self):
        query = self.query_plan(mode="FILTERED_ORDER", text=None, order_by=[
            {"field": "created_at", "descending": True}]).compile(())
        self.assertEqual([(row.field, row.descending) for row in query.order_by],
                         [("created_at", True), ("stable_key", False)])

    async def test_query_plan_rejects_fake_semantic_exhaustive_and_arbitrary_fields(self):
        with self.assertRaises(AlgorithmError):
            self.query_plan(mode="EXHAUSTIVE", order_by=[{"field": "title", "descending": False}]).compile(())
        with self.assertRaises(ValueError):
            self.query_plan(order_by=[{"field": "DROP_TABLE", "descending": False}])
        with self.assertRaises(AlgorithmError):
            self.query_plan().compile(("image-one", "image-two"))

    async def test_query_plan_uses_frozen_model_limit_without_receiving_candidates(self):
        plan = self.query_plan(combination="ALL")
        client = SimpleNamespace(json=AsyncMock(return_value={"choices": [{"message": {
            "content": plan.model_dump_json()}}]}))
        query = await plan_knowledge_query(text="两图都有关的27项", image_handles=["one", "two"],
            model={"model_id": str(ID), "max_tokens": 1234}, prompt="合成查询编译", client=client, auth_context=None)
        payload = client.json.call_args.kwargs["payload"]
        self.assertEqual(payload["max_tokens"], 1234)
        self.assertNotIn("candidates", json.dumps(payload))
        self.assertEqual(query.combination, "ALL")

    def runtime(self, run=None, conversation=None):
        uow = SimpleNamespace(
            runs=SimpleNamespace(get_scoped=AsyncMock(return_value=run), get_business_scoped=AsyncMock(return_value=run)),
            conversations=SimpleNamespace(get_scoped=AsyncMock(return_value=conversation)),
            artifacts=SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
                run_id=ID, payload_json={"knowledge_composition": {"retrieval_ref": str(ID),
                    "items": [{"key": "a" * 64}]}}))))
        @asynccontextmanager
        async def factory():
            yield uow
        integration = SimpleNamespace(open_history=AsyncMock(return_value="exact-stream"))
        runtime = AgentRuntimeService(uow_factory=factory, knowledge_integration=integration,
                                      notification_publisher=object())
        return runtime, uow, integration

    def history_scope(self):
        return {"run_id": ID, "key": "a" * 64, "actor_id": ID, "domain_id": 1 if ORACLE else ID,
                "auth_context": None, **({} if ORACLE else {"tenant_id": ID, "owner_app_id": "knowledge_retrieval"})}

    def run_record(self):
        return SimpleNamespace(run_id=ID, actor_id=ID, status="COMPLETED", result_artifact_id=ID,
            config_snapshot_json={"agent": {"owner_app_id": "knowledge_retrieval"},
                                  "conversation": {"conversation_id": str(ID)}})

    async def test_history_deleted_conversation_blocks_before_downstream(self):
        runtime, _, integration = self.runtime(run=self.run_record())
        with self.assertRaises(AgentRuntimeNotFound):
            await runtime.open_knowledge_content(**self.history_scope())
        integration.open_history.assert_not_awaited()

    async def test_history_cancelled_run_blocks_before_downstream(self):
        run = self.run_record()
        run.status = "CANCELLED"
        runtime, _, integration = self.runtime(run=run, conversation=SimpleNamespace(status="ACTIVE"))
        with self.assertRaises(AgentRuntimeNotFound):
            await runtime.open_knowledge_content(**self.history_scope())
        integration.open_history.assert_not_awaited()

    async def test_history_only_returns_exact_original_key(self):
        runtime, _, integration = self.runtime(run=self.run_record(), conversation=SimpleNamespace(status="ACTIVE"))
        self.assertEqual(await runtime.open_knowledge_content(**self.history_scope()), "exact-stream")
        self.assertEqual(integration.open_history.call_args.kwargs["key"], "a" * 64)
        with self.assertRaises(AgentRuntimeNotFound):
            await runtime.open_knowledge_content(**{**self.history_scope(), "key": "b" * 64})
        self.assertEqual(integration.open_history.await_count, 1)

    async def test_history_wrong_actor_denied(self):
        run = self.run_record()
        run.actor_id = "another-actor"
        runtime, _, integration = self.runtime(run=run)
        with self.assertRaises(AgentRuntimeNotFound):
            await runtime.open_knowledge_content(**self.history_scope())
        integration.open_history.assert_not_awaited()

    async def test_public_output_dto_rejects_internal_handle(self):
        value = {"retrieval_ref": str(ID), "status": "READY", "reason": None, "mode": "RELEVANCE",
            "continuation": None, "primary_hits": [], "items": [], "linked": [], "channels": [],
            "model_delivery": "NOT_APPLICABLE"}
        KnowledgeSearchResult.model_validate(value)
        with self.assertRaises(ValueError):
            KnowledgeSearchResult.model_validate({**value, "scope": {"content_handle": "internal"}})


if __name__ == "__main__":
    unittest.main()
