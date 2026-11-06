"""知识子执行及正式 Composer 接线的离线回归，不访问真实数据库或模型。"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [str(path) for path in ROOT.glob("services/*/src")]
sys.path[:0] = [str(ROOT / "tests/unit/knowledge_algorithms"), str(Path(__file__).parent)]
import test_foundation as support
from knowledge_algorithm_fixtures import FakeBackend
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.runtime import KnowledgeCompositionArtifact
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.retrieval.selection import PayloadCost
from aiops_agent.application.knowledge_child import OperationsKnowledgeChild, child_id, project_answer
from aiops_agent.application.operations_knowledge_authorization import validate_parent
from aiops_agent.contracts.knowledge import KnowledgeSearchRequest
from aiops_agent.domain.knowledge_retrieval import RetrievalAccess
from agent_runtime.application.attachments import ConversationAttachmentStore
from agent_runtime.runtime import SkillArtifact, SkillResult
from agent_runtime.specialists.response_composer.skill import ResponseComposerSkill

ORACLE = (ROOT / "database/oracle").exists()


class ChildTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.actor, self.turn, self.agent, self.version = (uuid4() for _ in range(4))
        self.domain, self.tenant = (10, None) if ORACLE else (uuid4(), uuid4())
        self.access = RetrievalAccess(self.domain, self.tenant, str(self.actor))
        self.parent = SimpleNamespace(created_by=self.actor, status="RUNNING", conversation_id=uuid4())
        self.conversation = SimpleNamespace(status="ACTIVE", created_by=self.actor,
            agent_id=self.agent, agent_version_id=self.version)
        self.agent_record = SimpleNamespace(status="ACTIVE", current_version_id=self.version)
        version = SimpleNamespace(models_json={"composer_llm": str(uuid4())}, instruction="本轮诊断 Agent 指令")
        self.tickets = {}
        async def binding(access, identity, **kwargs):
            return self.tickets.get(identity)
        async def create_ticket(access, **kwargs):
            self.tickets[kwargs["ticket_id"]] = kwargs
        self.uow = SimpleNamespace(
            turns=SimpleNamespace(get_turn=AsyncMock(return_value=self.parent)),
            conversations=SimpleNamespace(get_conversation=AsyncMock(return_value=self.conversation)),
            agents=SimpleNamespace(get=AsyncMock(return_value=self.agent_record), version=AsyncMock(return_value=version)),
            knowledge_documents=SimpleNamespace(child_binding=AsyncMock(side_effect=binding),
                create_query_ticket=AsyncMock(side_effect=create_ticket)), commit=AsyncMock())
        @asynccontextmanager
        async def factory():
            yield self.uow
        self.binding = {name: str(uuid4()) for name in ("asset_id", "asset_version_id", "document_id",
            "document_version_id", "ingestion_job_id", "parse_run_id", "index_generation_id", "index_ref_id")}
        self.binding.update(title="合成手册", asset_kind="MANUAL")
        self.service = SimpleNamespace(_uow_factory=factory,
            _documents=SimpleNamespace(access=lambda context: self.access, retrieval=SimpleNamespace(
                clock=lambda: datetime.now(timezone.utc), limits=SimpleNamespace(max_versions=256, scope_ttl_seconds=900))),
            search=AsyncMock(return_value={"results": [{"document_bindings": [self.binding]}]}))
        self.client = SimpleNamespace(create_aiops_knowledge_run=AsyncMock(),
            get_run=AsyncMock(return_value={"status": "COMPLETED", "row_version": 1}),
            get_result=AsyncMock(), cancel_run=AsyncMock())
        self.client.open_knowledge_content = AsyncMock(side_effect=self.download)
        self.payload = {"answer": "合成结论 [D1]", "used_citation_labels": ["D1"], "references": [
            {"citation_label": "D1", "reference_type": "APP_KNOWLEDGE", "owner_app_id": "aiops",
             "run_id": str(child_id(self.turn, "run")), "document_version_id": self.binding["document_version_id"],
             "parse_run_id": self.binding["parse_run_id"], "evidence_key": "a"*64, "quote": ""}]}
        self.client.get_result.return_value = {"artifact_type": "GROUNDED_ANSWER", "payload": self.payload}
        self.child = OperationsKnowledgeChild(service=self.service, client=self.client)
        self.request = KnowledgeSearchRequest(query="如何处理合成问题？")
        self.context = SimpleNamespace()
        self.authorization = patch("aiops_agent.application.knowledge_search._current_knowledge_authorization",
            AsyncMock(return_value=(self.request, ())))
        self.authorization.start()
        self.addCleanup(self.authorization.stop)

    async def download(self, **kwargs):
        async def blocks():
            yield "合成证据正文".encode()
        return SimpleNamespace(blocks=blocks(), mime_type="text/plain", close=AsyncMock())

    async def answer(self):
        return await self.child.answer(context=self.context, turn_id=self.turn, agent_id=self.agent,
            agent_version_id=self.version, request=self.request)

    async def test_freezes_real_agent_and_exact_source_then_replays_without_reselection(self):
        answer, references = await self.answer()
        self.assertEqual(answer, "合成结论 [K1]")
        self.assertEqual(references[0]["quote"], "合成证据正文")
        spec = self.client.create_aiops_knowledge_run.call_args.kwargs["payload"]["execution_spec"]
        self.assertEqual(spec["instruction"], "本轮诊断 Agent 指令")
        self.assertEqual(spec["consumer_agent_id"], str(self.agent))
        self.assertEqual(spec["knowledge_binding"]["query_ticket"], str(child_id(self.turn, "ticket")))
        self.assertEqual(spec["models"]["context_llm"], spec["models"]["composer_llm"])
        await self.answer()
        self.assertEqual(self.service.search.await_count, 1)
        self.assertEqual(self.uow.knowledge_documents.create_query_ticket.await_count, 1)
        self.client.cancel_run.assert_not_awaited()

    async def test_no_business_candidates_does_not_create_runtime(self):
        self.service.search.return_value = {"results": []}
        answer, references = await self.answer()
        self.assertFalse(references)
        self.client.create_aiops_knowledge_run.assert_not_awaited()

    async def test_deleted_parent_rejects_before_search(self):
        self.conversation.status = "DELETED"
        with self.assertRaises(AlgorithmError):
            await self.answer()
        self.service.search.assert_not_awaited()

    async def test_agent_switch_rejects_before_search(self):
        self.agent_record.current_version_id = uuid4()
        with self.assertRaises(AlgorithmError):
            await self.answer()
        self.client.create_aiops_knowledge_run.assert_not_awaited()

    async def test_parent_withdrawal_after_start_cancels_child(self):
        async def create(**kwargs):
            self.parent.status = "CANCELLED"
        self.client.create_aiops_knowledge_run.side_effect = create
        self.client.get_run.return_value = {"status": "RUNNING", "row_version": 2}
        with self.assertRaises(AlgorithmError):
            await self.answer()
        self.client.cancel_run.assert_awaited_once()
        self.assertEqual(self.client.cancel_run.call_args.kwargs["run_id"], child_id(self.turn, "run"))

    async def test_timeout_cancels_known_child_id(self):
        self.child.timeout = 0
        self.client.get_run.return_value = {"status": "RUNNING", "row_version": 2}
        with self.assertRaises(TimeoutError):
            await self.answer()
        self.client.cancel_run.assert_awaited_once()

    async def test_failed_child_does_not_emit_answer(self):
        self.client.get_run.return_value = {"status": "FAILED", "row_version": 2}
        with self.assertRaises(ValueError):
            await self.answer()
        self.client.get_result.assert_not_awaited()

    async def test_cancellation_failure_does_not_override_original_error(self):
        self.child.timeout = 0
        self.client.get_run.side_effect = OSError("合成断连")
        with self.assertRaises(TimeoutError):
            await self.answer()

    async def test_foreign_app_run_parse_or_document_rejected(self):
        for field, value in (("owner_app_id", "knowledge_retrieval"), ("run_id", str(uuid4())),
                ("document_version_id", str(uuid4())), ("parse_run_id", str(uuid4()))):
            payload = {**self.payload, "references": [{**self.payload["references"][0], field: value}]}
            with self.subTest(field=field), self.assertRaises(AlgorithmError):
                project_answer(payload, run_id=child_id(self.turn, "run"), bindings=[self.binding], label_prefix="K")

    async def test_reference_closure_rejects_missing_or_invented_labels(self):
        for answer, used in (("无标签", ["D1"]), ("伪造 [D2]", ["D2"]), ("结论 [D1]", [])):
            with self.assertRaises(AlgorithmError):
                project_answer({**self.payload, "answer": answer, "used_citation_labels": used},
                    run_id=child_id(self.turn, "run"), bindings=[self.binding], label_prefix="K")

    async def test_duplicate_reference_labels_are_rejected(self):
        duplicate = {**self.payload, "references": self.payload["references"] * 2}
        with self.assertRaises(AlgorithmError):
            project_answer(duplicate, run_id=child_id(self.turn, "run"), bindings=[self.binding], label_prefix="K")

    async def test_reference_order_follows_used_label_order(self):
        first = self.payload["references"][0]
        second = {**first, "citation_label":"D2"}
        payload = {**self.payload, "answer":"第二项 [D2]，第一项 [D1]", "used_citation_labels":["D2","D1"], "references":[first,second]}
        answer, refs = project_answer(payload, run_id=child_id(self.turn,"run"), bindings=[self.binding], label_prefix="K")
        self.assertEqual(answer, "第二项 [K1]，第一项 [K2]")
        self.assertEqual([ref["label"] for ref in refs], ["K1","K2"])


class ComposerWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_formal_composition_projects_exact_identity_and_same_stream_body(self):
        with patch.object(support, "FakeBackend", lambda: FakeBackend("knowledge_retrieval")):
            await support.DeliveryTests.setup_evidence(self)
        run_id, retrieval_id = uuid4(), uuid4()
        result = await self.composer.compose(run_ref=str(run_id), turn_ref="turn",
            model_snapshot_ref="model", evidence_set=self.evidence, content_port=self.port,
            question="合成问题", budget=PayloadCost(10000,10,10000))
        composition = KnowledgeCompositionArtifact(str(retrieval_id), self.evidence, result.delivery,
            result.answer, result.labels, result.excluded, result.selection)
        raw = SkillResult(artifact=SkillArtifact(artifact_type="KNOWLEDGE_COMPOSITION",
            schema_version="knowledge.composition.v1", payload=encode(composition)))
        adapter = ResponseComposerSkill(model_client=None, prompt_resolver=None,
            app_knowledge=SimpleNamespace(execute=AsyncMock(return_value=raw)))
        context = SimpleNamespace(run_id=run_id, input_artifacts=(SimpleNamespace(
            artifact_id=uuid4(), security_level=0, artifact_type="RETRIEVAL_EVIDENCE", payload={"evidence": {"evidence_set": {"primary_hits": [1]}}}),),
            original_input="合成", config_snapshot={})
        context.task_id, context.actor_id = uuid4(), "synthetic"
        outputs = [value async for value in adapter.execute_stream(context)]
        payload = outputs[-1].artifact.payload
        self.assertEqual("".join(value.payload["delta"] for value in outputs[:-1]), payload["answer"])
        self.assertEqual(payload["references"][0]["document_version_id"], self.p.document_version_ref)
        self.assertEqual(payload["references"][0]["parse_run_id"], self.p.parse_run_ref)
        self.assertEqual(payload["references"][0]["evidence_key"], self.p.stable_key)
        self.assertEqual(payload["knowledge_composition"]["retrieval_ref"], str(retrieval_id))

    async def test_atomic_attachment_write_is_idempotent_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "a" / "image.png"
            ConversationAttachmentStore._write_once(target, b"synthetic")
            ConversationAttachmentStore._write_once(target, b"synthetic")
            with self.assertRaises(ValueError):
                ConversationAttachmentStore._write_once(target, b"changed")
            outside = root / "outside"
            outside.mkdir()
            alias = root / "alias"
            alias.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                ConversationAttachmentStore._write_once(alias / "bad.png", b"bad")
            self.assertFalse((outside / "bad.png").exists())


if __name__ == "__main__":
    unittest.main()
