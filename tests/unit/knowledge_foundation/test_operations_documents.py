"""AIOps 文档端口、历史授权和模型引用保护的离线合同，不访问业务环境。"""

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from types import SimpleNamespace
import unittest
import inspect
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from aiops_agent.application.operations_documents import OperationsDocuments
from aiops_agent.application.operations_knowledge_authorization import OperationsKnowledgeAuthorization
from aiops_agent.contracts.knowledge import KnowledgeSearchRequest
from aiops_agent.domain.knowledge_retrieval import RetrievalAccess
from aiops_agent.repositories.knowledge_documents import OperationsDocumentRepository
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.identity import fingerprint


class OperationsDocumentsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.ids = [uuid4() for _ in range(6)]
        self.version, self.document, self.job, self.parse, self.generation, self.ref = self.ids
        self.access = RetrievalAccess(42, None, "actor")
        self.record = dict(document_id=str(self.document), document_version_id=str(self.version),
            ingestion_job_id=str(self.job), parse_run_id=str(self.parse), generation_id=str(self.generation),
            processing_status="PARSED", status="PARSED", active_generation_id=None, subject_ref="actor")
        self.uow = SimpleNamespace(knowledge_documents=SimpleNamespace(inspect=AsyncMock(side_effect=lambda *a:dict(self.record))),
            knowledge_retrieval=SimpleNamespace(publish=AsyncMock()))
        @asynccontextmanager
        async def factory():
            yield self.uow
        self.ingestion=SimpleNamespace(clock=lambda:self.now, get_job=AsyncMock(),
            store=SimpleNamespace(put_bytes=AsyncMock(return_value=SimpleNamespace(handle_ref="source"))))
        self.pipeline=SimpleNamespace(process=AsyncMock(return_value={"generation_id":str(self.generation)}), worker_id="worker")
        self.docs=OperationsDocuments(uow_factory=factory, ingestion=self.ingestion, projection=None,
            retrieval=None, parse_profile=SimpleNamespace(max_input_bytes=100), pipeline=self.pipeline)

    async def test_ready_generation_is_not_rebuilt(self):
        result=await self.docs.process(self.access,self.version)
        self.assertEqual(result["generation_id"],str(self.generation))
        self.pipeline.process.assert_not_awaited()

    async def test_other_live_worker_is_not_stolen_or_failed(self):
        self.record.update(processing_status="PARSING",generation_id=None)
        self.ingestion.get_job.return_value=SimpleNamespace(status="PARSING",lease_until=self.now+timedelta(seconds=60),lease_owner="another")
        result=await self.docs.process(self.access,self.version)
        self.assertEqual(result["processing_status"],"PARSING")
        self.pipeline.process.assert_not_awaited()

    async def test_withdrawn_document_cannot_be_processed(self):
        self.record["status"]="WITHDRAWN"
        with self.assertRaises(AlgorithmError):
            await self.docs.process(self.access,self.version)
        self.pipeline.process.assert_not_awaited()

    async def test_stage_validates_hash_and_path_before_storing_bytes(self):
        raw=b"synthetic"
        for name,digest in (("../source.txt",sha256(raw).hexdigest()),("source.txt","a"*64)):
            with self.assertRaises(AlgorithmError):
                await self.docs.stage(file_name=name,media_type="text/plain",body=raw,expected_hash=digest)
        self.ingestion.store.put_bytes.assert_not_awaited()
        staged=await self.docs.stage(file_name="source.txt",media_type="text/plain",body=raw,expected_hash=sha256(raw).hexdigest())
        self.assertEqual(staged["source"].owner_binding_ref,"aiops")
        self.assertEqual(staged["source"].document_version_ref,str(staged["document_version_id"]))

    async def test_publish_requires_exact_document_job_parse_and_generation(self):
        index=SimpleNamespace(document_id=self.document,document_version_id=self.version,
            ingestion_job_id=self.job,parse_run_id=self.parse,index_generation_id=self.generation)
        for field in ("document_id","ingestion_job_id","parse_run_id","generation_id"):
            saved=self.record[field]
            self.record[field]=str(uuid4())
            with self.assertRaises(AlgorithmError):
                await self.docs.publish(self.uow,self.access,[index])
            self.record[field]=saved
        self.uow.knowledge_retrieval.publish.assert_not_awaited()
        await self.docs.publish(self.uow,self.access,[index])
        self.uow.knowledge_retrieval.publish.assert_awaited_once()


class HistoricalTicketTests(unittest.IsolatedAsyncioTestCase):
    async def test_expired_ticket_keeps_exact_run_subject_spec_and_digest_fence(self):
        now=datetime.now(timezone.utc)
        access=RetrievalAccess(42,None,"actor",run_ref="run",spec_hash="spec")
        value={"run_ref":"run","spec_hash":"spec","snapshots":[{"version":"exact"}]}
        row=SimpleNamespace(status="QUERY_TICKET",subject_ref="actor",expires_at=now-timedelta(seconds=1),
            policy_json=value,scope_hash=fingerprint(value),allowed_versions_hash=fingerprint(value["snapshots"]))
        repo=OperationsDocumentRepository(None,lambda:None)
        repo._repos=lambda _:{"knowledge_scope_snapshot":SimpleNamespace(get=AsyncMock(return_value=row))}
        with self.assertRaises(AlgorithmError):
            await repo.query_ticket(access,uuid4(),now=now)
        self.assertEqual(await repo.query_ticket(access,uuid4(),now=now,allow_expired=True),value)
        for wrong in (replace(access,subject_ref="other"),replace(access,run_ref="other"),replace(access,spec_hash="other")):
            with self.assertRaises(AlgorithmError):
                await repo.query_ticket(wrong,uuid4(),now=now,allow_expired=True)
        value["snapshots"].append({"version":"injected"})
        with self.assertRaises(AlgorithmError):
            await repo.query_ticket(access,uuid4(),now=now,allow_expired=True)

    async def test_history_expiry_exception_does_not_skip_current_business_authorization(self):
        access=RetrievalAccess(42,None,"actor",run_ref="run",spec_hash="spec",historical_read=True)
        request=KnowledgeSearchRequest(query="synthetic")
        ticket={"parent_turn_id":None,"request":request.model_dump(mode="json"),"agent_id":str(uuid4()),
            "target_id":None,"snapshots":[]}
        documents=SimpleNamespace(query_ticket=AsyncMock(return_value=ticket))
        repo=SimpleNamespace(list_published_candidates=AsyncMock(return_value=[]))
        @asynccontextmanager
        async def factory():
            yield SimpleNamespace(knowledge_documents=documents,operations_knowledge=repo)
        authorization=OperationsKnowledgeAuthorization(service=SimpleNamespace(_uow_factory=factory))
        with patch("aiops_agent.application.operations_knowledge_authorization._current_knowledge_authorization",
                   AsyncMock(return_value="KNOWLEDGE_DISABLED")):
            with self.assertRaises(AlgorithmError):
                await authorization.authorize(access,"ops-query:"+str(uuid4()))
        self.assertTrue(documents.query_ticket.await_args.kwargs["allow_expired"])
        repo.list_published_candidates.assert_not_awaited()


class ModelReferencesTests(unittest.IsolatedAsyncioTestCase):
    async def test_library_and_ops_configuration_models_are_real_references(self):
        from knowledge_retrieval_app.repositories.agent import KnowledgeRetrievalAgentRepository
        from aiops_agent.repositories.agent import AIOpsAgentRepository
        model=uuid4()
        library=SimpleNamespace(library_id=uuid4(),display_name="synthetic",status="DISABLED",domain_id=42,
            configuration_json={"models":{"embedding":str(model),"composer":str(uuid4())}})
        result=SimpleNamespace(all=lambda:[library])
        session=SimpleNamespace(scalars=AsyncMock(return_value=result))
        repo=KnowledgeRetrievalAgentRepository(session, **({"write_guard": lambda: None} if "write_guard" in inspect.signature(KnowledgeRetrievalAgentRepository).parameters else {}))
        refs=await repo.knowledge_model_references(model_id=model, **({"tenant_id": uuid4()} if "tenant_id" in inspect.signature(repo.knowledge_model_references).parameters else {}))
        self.assertEqual([(row["resource_type"],row["usage"]) for row in refs],[("knowledge_library","embedding")])
        base=SimpleNamespace(knowledge_base_id=uuid4(),display_name="synthetic",status="ACTIVE",domain_id=42,
            configuration_json={"models":{"embedding":str(model)}})
        config=SimpleNamespace(domain_id=42,models_json={"parser_llm":str(model)})
        session.scalars=AsyncMock(side_effect=[SimpleNamespace(all=lambda:[base]),SimpleNamespace(all=lambda:[config])])
        ops=AIOpsAgentRepository(session, **({"write_guard": lambda: None} if "write_guard" in inspect.signature(AIOpsAgentRepository).parameters else {}))
        refs=await ops.knowledge_model_references(model_id=model)
        self.assertEqual({row["resource_type"] for row in refs},{"operations_knowledge_base","operations_case_configuration"})


if __name__ == "__main__":
    unittest.main()
