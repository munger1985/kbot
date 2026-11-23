"""正式知识 Runtime 的离线业务围栏和规划测试；没有网络、实库或真实模型。"""

from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [str(path) for path in ROOT.glob("services/*/src")]
from agent_runtime.application.knowledge_query import KnowledgeQueryPlan, PlannedImageQuery, plan_knowledge_query
from agent_runtime.adapters.knowledge_ocr import read_question_images
from agent_runtime.application.knowledge_delivery import QuestionImage
from platform_clients.knowledge_computation import KnowledgeComputationClientError
from agent_runtime.application.runtime_service import AgentRuntimeService, AgentRuntimeNotFound
from agent_runtime.composition.knowledge_runtime import RuntimeKnowledgeIntegration
from knowledge_algorithms.contracts.errors import AlgorithmError
from platform_core.contracts.knowledge_app import KnowledgeLibraryView, KnowledgeSearchResult
from platform_core.security import AUTH_CONTEXT_HEADER, INTERNAL_TOKEN_HEADER

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

    async def test_visual_question_uses_independent_recall_without_discarding_topic(self):
        query = self.query_plan(combination="COMPOSED").compile(("image-one", "image-two"),
                                                               independent_modalities=True)
        self.assertEqual(query.image_handles, ("image-one", "image-two"))
        self.assertEqual(query.text, "图1-1的引用文字")
        self.assertEqual(query.combination, "ANY")

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

    async def test_visual_query_planner_marks_explicit_transcription(self):
        plan = self.query_plan(text=None).model_dump()
        plan["transcription_requested"] = True
        client = SimpleNamespace(json=AsyncMock(return_value={"choices": [{"message": {
            "content": json.dumps(plan)}}]}))
        value = await plan_knowledge_query(text="请逐字转写图片", image_handles=["one"],
            model={"model_id": str(ID), "max_tokens": 1234}, prompt="合成查询编译",
            client=client, auth_context=None, independent_modalities=True, classify_transcription=True)
        self.assertIsInstance(value, PlannedImageQuery)
        self.assertTrue(value.transcription_requested)
        self.assertEqual(value.query.image_handles, ("one",))

    async def test_ocr_failure_does_not_drop_images_or_other_recognized_text(self):
        from model_serving.ocr.schema import OCRResponse

        raw = (QuestionImage("image/png", b"image-one"), QuestionImage("image/webp", b"image-two"))
        revision = "a" * 64
        response = OCRResponse(model_id=ID, provider="fixture", model_revision="ocr-engine/v2",
                               text="合成内容", blocks=[]).model_dump(mode="json")
        client = SimpleNamespace(binding=AsyncMock(return_value={"model_id": str(ID), "revision": revision}),
            infer=AsyncMock(side_effect=[KnowledgeComputationClientError(503, "OCR_PROVIDER_UNAVAILABLE"),
                response]))
        value = await read_question_images(raw, client=client, auth_context=None, model_id=str(ID), run_id=ID)
        self.assertEqual(value[0], raw[0])
        self.assertEqual(value[1].data, raw[1].data)
        self.assertEqual(value[1].ocr_text, "合成内容")
        payload = client.infer.await_args.kwargs["payload"]
        self.assertEqual(payload["expected_revision"], revision)
        self.assertEqual(payload["mime_type"], "image/webp")

    async def test_ocr_rejects_wrong_model_and_invalid_response(self):
        raw = (QuestionImage("image/png", b"image-one"),)
        revision = "a" * 64
        for invalid in ({"model_id": str(UUID(int=1))}, {"model_revision": " "}, {"text": None}):
            with self.subTest(invalid=invalid):
                reply = {"model_id": str(ID), "model_revision": "ocr-engine/v2", "text": "合成文字", **invalid}
                client = SimpleNamespace(binding=AsyncMock(return_value={"model_id": str(ID), "revision": revision}),
                                         infer=AsyncMock(return_value=reply))
                self.assertEqual(await read_question_images(raw, client=client, auth_context=None,
                                                            model_id=str(ID), run_id=ID), raw)

    async def test_ocr_revision_conflict_stops_batch_and_preserves_originals(self):
        raw = (QuestionImage("image/png", b"image-one"), QuestionImage("image/webp", b"image-two"))
        revision = "a" * 64
        client = SimpleNamespace(binding=AsyncMock(return_value={"model_id": str(ID), "revision": revision}),
                                 infer=AsyncMock(side_effect=KnowledgeComputationClientError(409, "MODEL_REVISION_CHANGED")))
        self.assertEqual(await read_question_images(raw, client=client, auth_context=None,
                                                    model_id=str(ID), run_id=ID), raw)
        self.assertEqual(client.infer.await_count, 1)
        self.assertEqual(client.infer.await_args.kwargs["payload"]["expected_revision"], revision)

    async def test_text_only_query_plan_never_requests_visual_outputs(self):
        question = "荧光光谱法是什么"
        plan = self.query_plan(output_types=["FIGURE", "MIXED"], text=None, combination="ALL")
        client = SimpleNamespace(json=AsyncMock(return_value={"choices": [{"message": {
            "content": plan.model_dump_json()}}]}))
        query = await plan_knowledge_query(text=question, image_handles=(),
            model={"model_id": str(ID), "max_tokens": 1234}, prompt="合成查询编译",
            client=client, auth_context=None, text_only=True)
        self.assertEqual(query.output_types, ("TEXT", "TABLE"))
        self.assertEqual(query.text, question)
        self.assertFalse(query.image_handles)
        self.assertIsNone(query.combination)

    async def test_runtime_planning_and_composition_use_model_service_credentials(self):
        question = "解释一下审计工作管理"
        plan = self.query_plan(text=question)
        def endpoint(role):
            return SimpleNamespace(
                base_url=f"http://{role}.test", audience=role, timeout_seconds=5)
        settings = SimpleNamespace(knowledge_retrieval=endpoint("knowledge-app"),
            aiops=endpoint("aiops"), llm=endpoint("llm"), vlm=endpoint("vlm"),
            knowledge=SimpleNamespace(payload_bytes=8192, image_token_cost=100))
        model = {"model_id": str(ID), "max_tokens": 1234, "config_fingerprint": "frozen"}
        context = SimpleNamespace(original_input=question, input_artifacts=(),
            config_snapshot={"app_knowledge": {"scope": {}, "delegation": {}},
                "agent": {"models": {"context_llm": model, "composer_llm": model}}})
        identity = SimpleNamespace(model_copy=Mock(return_value=object()))
        requests = []

        @asynccontextmanager
        async def respond(method, url, *, headers, **kwargs):
            payload = kwargs["json"]
            requests.append((method, url, headers, payload["max_tokens"]))
            content_type = list if url.endswith("/inference") else str
            self.assertTrue(all(isinstance(message["content"], content_type) for message in payload["messages"]))
            body = plan.model_dump_json() if len(requests) == 1 else "依据资料 [C1]。"
            async def chunks(size):
                yield json.dumps({"choices": [{"message": {"content": body}}]}, ensure_ascii=False).encode()
            yield SimpleNamespace(status=200, content=SimpleNamespace(iter_chunked=chunks))

        def sign(*, audience, caller_service, context):
            self.assertEqual(caller_service, "agent-runtime")
            return {INTERNAL_TOKEN_HEADER: "fixture-credential",
                    AUTH_CONTEXT_HEADER: f"{audience}-context-{len(requests)}"}

        session = SimpleNamespace(request=respond, close=AsyncMock())
        with patch("agent_runtime.composition.knowledge_runtime.create_knowledge_delegation_codec"), patch(
                "agent_runtime.composition.knowledge_runtime.decode",
                return_value=SimpleNamespace(owner_binding_ref="knowledge_retrieval")), patch(
                "platform_clients.knowledge_computation.aiohttp.ClientSession", return_value=session), patch(
                "platform_clients.knowledge_model_binding.build_internal_auth_headers", side_effect=sign), patch(
                "platform_clients.knowledge_computation.build_scoped_internal_auth_headers",
                side_effect=AssertionError("模型调用不能使用知识计算身份")):
            runtime = RuntimeKnowledgeIntegration(settings=settings, caller_service="agent-runtime",
                uow_factory=None, prompt_resolver=SimpleNamespace(resolve=AsyncMock(
                    return_value=SimpleNamespace(content="合成测试"))))
            runtime.auth_context = Mock(return_value=identity)
            runtime.clients["knowledge_retrieval"].for_delegation = Mock()
            self.assertEqual((await runtime.query(context)).text, question)
            for role in ("llm", "vlm"):
                if role == "vlm":
                    context.config_snapshot["agent"]["models"]["query_vlm"] = model
                composer = await runtime.composer(context)
                reply = await composer.gateway.generate(question=question, evidence_parts=[],
                    model_snapshot_ref="frozen")
                self.assertEqual(reply.used_labels, ("C1",))

        self.assertEqual([row[1] for row in requests], [
            "http://llm.test/internal/v1/chat/completions",
            "http://llm.test/internal/v1/chat/completions",
            "http://vlm.test/internal/v1/inference",
        ])
        self.assertEqual([row[2][AUTH_CONTEXT_HEADER] for row in requests],
                         ["llm-context-0", "llm-context-1", "vlm-context-2"])
        self.assertTrue(all(row[2][INTERNAL_TOKEN_HEADER] == "fixture-credential"
                            and row[3] == 1234 for row in requests))
        self.assertEqual(identity.model_copy.call_count, 3)
        identity.model_copy.assert_called_with(
            update={"scopes": ("model.inference",), "calling_service": "agent-runtime"})

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

    async def test_history_pdf_requires_an_answer_reference(self):
        runtime, uow, integration = self.runtime(run=self.run_record(), conversation=SimpleNamespace(status="ACTIVE"))
        with self.assertRaises(AgentRuntimeNotFound):
            await runtime.open_knowledge_content(**self.history_scope(), source_pdf=True)
        integration.open_history.assert_not_awaited()
        payload = uow.artifacts.get.return_value.payload_json
        payload["references"] = [{"reference_type": "APP_KNOWLEDGE", "owner_app_id": "knowledge_retrieval",
                                  "evidence_key": "a" * 64}]
        self.assertEqual(await runtime.open_knowledge_content(**self.history_scope(), source_pdf=True), "exact-stream")
        self.assertTrue(integration.open_history.call_args.kwargs["source_pdf"])

    async def test_public_output_dto_rejects_internal_handle(self):
        value = {"retrieval_ref": str(ID), "status": "READY", "reason": None, "mode": "RELEVANCE",
            "continuation": None, "primary_hits": [], "items": [], "linked": [], "channels": [],
            "model_delivery": "NOT_APPLICABLE", "scope_exhausted": False}
        KnowledgeSearchResult.model_validate(value)
        with self.assertRaises(ValueError):
            KnowledgeSearchResult.model_validate({**value, "scope": {"content_handle": "internal"}})


if __name__ == "__main__":
    unittest.main()
