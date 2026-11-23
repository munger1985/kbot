"""正式 App 业务闭环的合成回归；仅 SQLite 内存与受控解析替身。"""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [
    str(path) for path in ROOT.glob("services/*/src")]
sys.path[:0] = [str(ROOT / "tests/unit/knowledge_ingestion"), str(ROOT / "tests/unit/knowledge_algorithms")]
import test_app_knowledge_retrieval as support
from knowledge_algorithm_fixtures import CharacterTokenizer, parse_profile, projection_profile
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.projections.builder import ProjectionBuilder
from platform_core.contracts.knowledge_app import KnowledgeDocumentPage, KnowledgePassageContext, KnowledgeSearchRequest
from knowledge_retrieval_app.entities.knowledge_library import KnowledgeLibraryEntity
from knowledge_retrieval_app.entities.agent import KnowledgeRetrievalAgentEntity, KnowledgeRetrievalAgentVersionEntity
from knowledge_retrieval_app.domain.knowledge_library import LibraryConflict
from knowledge_retrieval_app.repositories.knowledge_library import KnowledgeLibraryRepository
from knowledge_retrieval_app.application.knowledge_library import KnowledgeLibraryService, LibraryAuthorization


class SyntheticSession(support.AsyncSyntheticSession):
    def get_bind(self):
        return self.session.get_bind()

    async def delete(self, row):
        self.session.delete(row)

    async def scalar(self, *args, **kwargs):
        return self.session.scalar(*args, **kwargs)


class LibraryBusinessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await support.AppRetrievalTests.asyncSetUp(self)
        KnowledgeLibraryEntity.__table__.create(self.engine)
        KnowledgeRetrievalAgentEntity.__table__.create(self.engine)
        KnowledgeRetrievalAgentVersionEntity.__table__.create(self.engine)
        self.fixture.sessions = lambda: SyntheticSession(self.engine)
        previous = self.uow_factory
        class BusinessUow:
            async def __aenter__(inner):
                inner.delegate = previous()
                value = await inner.delegate.__aenter__()
                value.library = lambda scope: KnowledgeLibraryRepository(value.session, scope)
                from knowledge_retrieval_app.repositories.knowledge_queue import KnowledgeJobQueueRepository
                value.knowledge_queue = KnowledgeJobQueueRepository(value.session, lambda: None)
                return value
            async def __aexit__(inner, *args):
                return await inner.delegate.__aexit__(*args)
        self.uow_factory = BusinessUow
        self.fixture.service.uow_factory = self.uow_factory
        self.service.uow_factory = self.uow_factory
        self.service.authorization = LibraryAuthorization(self.uow_factory)
        self.service.profiles = {"default": self.service.profiles["baseline"]}
        self.business = KnowledgeLibraryService(uow_factory=self.uow_factory,
            ingestion=self.fixture.service, retrieval=self.service,
            parse_profile=parse_profile(), configurations={"default": object()})
        created = await self.business.create(self.fixture.scope, display_name="合成资料库",
                                            description=None, profile_ref="default")
        self.library_id = UUID(created["library_id"])

    async def asyncTearDown(self):
        await support.AppRetrievalTests.asyncTearDown(self)

    async def upload(self, key="upload-key", raw="论文图文与知识检索。".encode()):
        return await self.business.upload(self.fixture.scope, self.library_id,
            name="synthetic.txt", mime_type="text/plain", raw=raw, idempotency_key=key)

    async def test_upload_idempotency_conflict_and_private_acl(self):
        first = await self.upload()
        self.assertEqual(first["version_id"], (await self.upload())["version_id"])
        with self.assertRaises(AlgorithmError):
            await self.upload(raw=b"different")
        with self.assertRaises(AlgorithmError):
            await self.business.files(replace(self.fixture.scope, subject_ref="other"), self.library_id,
                                      page=1, page_size=20)

    async def test_library_catalog_and_guarded_delete(self):
        manager = self.fixture.scope
        empty = UUID((await self.business.create(manager, display_name="  ＡＢＣ  ", description=None,
                                                 profile_ref="default"))["library_id"])
        with self.assertRaisesRegex(LibraryConflict, "NAME_CONFLICT"):
            await self.business.create(manager, display_name="abc", description=None, profile_ref="default")
        disabled = await self.business.update(manager, empty, expected_revision=1, changes={"status": "DISABLED"})
        self.assertNotIn(str(empty), [row["library_id"] for row in await self.business.list(manager)])
        self.assertEqual((await self.business.get(manager, empty, management=True))["status"], "DISABLED")
        with self.assertRaisesRegex(LibraryConflict, "REVISION_CONFLICT"):
            await self.business.delete(manager, empty, expected_revision=1)
        await self.business.delete(manager, empty, expected_revision=disabled["row_version"])
        with self.assertRaises(AlgorithmError):
            await self.business.get(manager, empty, management=True)

        actor, agent_id = self.fixture.scope.subject_ref, uuid4()
        async with self.uow_factory() as uow:
            uow.session.add(KnowledgeRetrievalAgentEntity(agent_id=agent_id, domain_id=manager.domain_id,
                display_name="合成 Agent", status="DISABLED", current_version_id=uuid4(), created_by=actor, updated_by=actor))
            uow.session.add(KnowledgeRetrievalAgentVersionEntity(agent_version_id=uuid4(), agent_id=agent_id,
                version_no=1, library_id=self.library_id, data_model_ids_json=[], models_json={}, config_json={}, created_by=actor))
            await uow.commit()
        self.assertEqual((await self.business.get(manager, self.library_id, management=True))["delete_block"], "AGENT_REFERENCE")
        with self.assertRaisesRegex(LibraryConflict, "AGENT_REFERENCE"):
            await self.business.delete(manager, self.library_id, expected_revision=1)
        await self.upload()
        with self.assertRaisesRegex(LibraryConflict, "HAS_DOCUMENTS"):
            await self.business.delete(manager, self.library_id, expected_revision=1)

    async def test_directory_filters_precede_paging_and_exact_link_rechecks_acl(self):
        manager = self.fixture.scope
        contributor = replace(manager, subject_ref=str(uuid4()))
        reader = replace(manager, subject_ref=str(uuid4()))
        await self.business.update(manager, self.library_id, expected_revision=1, changes={"grants": {
            manager.subject_ref: ["READ", "UPLOAD", "MANAGE", "REVIEW"],
            contributor.subject_ref: ["READ", "UPLOAD"],
            reader.subject_ref: ["READ"],
        }})
        private = await self.upload()
        own = await self.business.upload(contributor, self.library_id, name="mine.txt",
            mime_type="text/plain", raw=b"own synthetic", idempotency_key="upload-key")
        docs = await self.business.files(manager, self.library_id, title="SYNTHETIC", page_size=1)
        self.assertEqual(docs["items"][0]["title"], "synthetic.txt")
        self.assertFalse(docs["has_more"])
        tasks = await self.business.files(contributor, self.library_id, view="processing")
        self.assertEqual(list(tasks["items"][0]["versions"]), [own["version_id"]])
        self.assertIn("cancel", tasks["items"][0]["versions"][own["version_id"]]["allowed_actions"])
        self.assertFalse((await self.business.files(reader, self.library_id, version_id=own["version_id"]))["items"])
        self.assertFalse((await self.business.files(contributor, self.library_id, view="processing",
            version_id=private["version_id"]))["items"])

    async def test_review_parse_publish_search_withdraw(self):
        from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker
        from knowledge_retrieval_app.application.knowledge_worker import KnowledgeBusinessWorker
        paragraphs = ["更早的条款。", "上两段。", "上一条款。", "论文图文与知识检索。", "下一条款。", "授权视图不应显示。", "下两段。", "更后的条款。"]
        value = await self.upload(raw="\n\n".join(paragraphs).encode())
        version_id = UUID(value["version_id"])
        self.fixture.job = UUID(value["job_id"])
        self.fixture.version = version_id
        empty = await self.business.search(self.fixture.scope, self.library_id,
                                          KnowledgeSearchRequest(text="知识检索"))
        self.assertFalse(empty["primary_hits"])
        with self.assertRaises(AlgorithmError):
            await self.fixture.claim()
        await self.business.review(self.fixture.scope, self.library_id, version_id,
                                   decision="APPROVE", comment="合成审核", parse_strategy="TEXT")
        await self.fixture.parse()
        async def original(content):
            return None if content.text == paragraphs[5] else content
        pipeline = KnowledgePipelineWorker(ingestion=self.fixture.service, projection=self.fixture.projections,
            builder_factory=lambda *_: ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), None),
            profile=projection_profile(), allowed_policy=original, policy_revision="synthetic/v1", worker_id="auto-publish")
        worker = KnowledgeBusinessWorker(uow_factory=self.uow_factory, pipeline=pipeline,
                                         worker_id="auto-publish", poll_seconds=.1)
        self.assertTrue(await worker.process_once())
        self.assertTrue(await worker.process_once())
        result = await self.business.search(self.fixture.scope, self.library_id,
                                           KnowledgeSearchRequest(text="知识检索", requested_count=2))
        self.assertTrue(result["primary_hits"])
        self.assertIn("知识检索", result["items"][0]["text"])
        self.assertEqual(result["items"][0]["version_id"], str(version_id))
        self.assertEqual(result["items"][0]["source_title"], value["name"])
        self.assertNotIn("content_handle", repr(result))
        hit = next(item for item in result["items"] if item["key"] == result["primary_hits"][0]["key"])
        self.assertEqual(hit["document_id"], value["document_id"])
        self.assertTrue(hit["passage_key"])
        anchor = (result["retrieval_ref"], hit["key"])
        context = KnowledgePassageContext.model_validate(await self.business.passage_context(
            self.fixture.scope, *anchor, before=2, after=2))
        self.assertTrue(context.has_before and context.has_after)
        self.assertIsNone(context.pdf_page)
        self.assertEqual(["".join(segment.text for segment in block.segments) for block in context.blocks],
                         [paragraphs[i] for i in (1, 2, 3, 4, 6)])
        self.assertEqual([block.anchor for block in context.blocks], [False, False, True, False, False])
        self.assertEqual([segment.text for segment in context.blocks[2].segments if segment.matched], ["知识检索"])
        expanded = await self.business.passage_context(self.fixture.scope, *anchor, before=4, after=4)
        self.assertFalse(expanded["has_before"] or expanded["has_after"])
        self.assertEqual(len(expanded["blocks"]), 7)
        with self.assertRaises(AlgorithmError):
            await self.business.passage_context(self.fixture.scope, anchor[0], "0" * 64, before=2, after=2)
        with self.assertRaises(AlgorithmError):
            await self.business.passage_context(replace(self.fixture.scope, subject_ref="outsider"), *anchor, before=2, after=2)
        await self.business.withdraw(self.fixture.scope, self.library_id, version_id)
        with self.assertRaises(AlgorithmError):
            await self.service.receipt(self.fixture.scope, result["retrieval_ref"])
        with self.assertRaises(AlgorithmError):
            await self.business.passage_context(self.fixture.scope, *anchor, before=2, after=2)

    async def test_grants_revision_invalidates_scope_and_prevents_last_manager_removal(self):
        scope = await self.service.create_scope(self.fixture.scope, resource_ref="library:" + str(self.library_id))
        with self.assertRaises(AlgorithmError):
            await self.business.update(self.fixture.scope, self.library_id, expected_revision=1,
                                       changes={"grants": {"reader": ["READ"]}})
        await self.business.update(self.fixture.scope, self.library_id, expected_revision=1,
                                   changes={"display_name": "更新后的合成资料库"})
        with self.assertRaises(AlgorithmError):
            await self.service.validate(self.fixture.scope, scope)

    async def test_unpublished_visibility_and_stream_reauthorization(self):
        value = await self.upload()
        version_id = UUID(value["version_id"])
        await self.business.update(self.fixture.scope, self.library_id, expected_revision=1,
            changes={"grants": {self.fixture.scope.subject_ref: ["READ", "UPLOAD", "REVIEW", "MANAGE", "WITHDRAW"],
                               "reader": ["READ"]}})
        reader = replace(self.fixture.scope, subject_ref="reader")
        self.assertFalse((await self.business.files(reader, self.library_id, page=1, page_size=20))["items"])
        with self.assertRaises(AlgorithmError):
            await self.business.source(reader, self.library_id, version_id)
        files = await self.business.files(self.fixture.scope, self.library_id, page=1, page_size=20)
        version = files["items"][0]["versions"][value["version_id"]]
        self.assertEqual(version["processing_status"], "AWAITING_APPROVAL")
        self.assertFalse(version["generation_ready"])
        _, stream = await self.business.source(self.fixture.scope, self.library_id, version_id)
        await self.business.update(self.fixture.scope, self.library_id, expected_revision=2,
            changes={"status": "DISABLED"})
        with self.assertRaises(AlgorithmError):
            _ = b"".join([chunk async for chunk in stream])

    async def test_task_progress_follows_committed_parse_events_and_index_lease(self):
        manager = self.fixture.scope
        self.fixture.now = datetime.now(timezone.utc)
        receipt = await self.upload()
        version_id = receipt["version_id"]
        self.fixture.job, self.fixture.version = UUID(receipt["job_id"]), UUID(version_id)

        async def progress(expected):
            page = await self.business.files(manager, self.library_id, view="processing")
            row = KnowledgeDocumentPage.model_validate(page).items[0].versions[version_id]
            self.assertEqual(row.progress_stage, expected)
            self.assertNotEqual(row.status, "PUBLISHED")
            return row

        await progress("AWAITING_APPROVAL")
        await self.business.review(manager, self.library_id, self.fixture.version,
                                   decision="APPROVE", comment=None, parse_strategy="TEXT")
        await progress("PARSE_QUEUED")
        await self.fixture.claim()
        await progress("PREPARING")
        self.fixture.service.page_limit = 1
        await self.fixture.service.poll(manager, self.fixture.job)
        await progress("EXTRACTING")
        await self.fixture.service.poll(manager, self.fixture.job)
        await progress("ORGANIZING")
        for _ in range(10):
            if (await self.fixture.service.poll(manager, self.fixture.job))["status"] == "PARSED":
                break
        self.assertEqual((await progress("INDEX_QUEUED")).parse_status, "PARSED")
        async with self.uow_factory() as uow:
            repos = uow.knowledge_ingestion._repos(manager)
            outboxes = repos["knowledge_outbox"]
            outbox = (await uow.session.scalars(outboxes.statement().where(
                outboxes.entity.document_version_id == self.fixture.version))).one()
            outbox_id = outbox.outbox_id
            claim = await uow.knowledge_ingestion.claim_outbox(manager, outbox_id, now=self.fixture.now,
                                                                worker_id="index-worker", lease_seconds=60)
            await uow.commit()
        self.assertFalse((await progress("INDEXING")).generation_ready)
        async with self.uow_factory() as uow:
            await uow.knowledge_ingestion.finish_outbox(manager, outbox_id, now=self.fixture.now,
                worker_id="index-worker", attempt=claim["attempt"], error_code="PROJECTION_FAILED")
            await uow.commit()
        self.assertEqual((await progress("INDEX_RETRY")).parse_status, "PARSED")

    async def test_initial_projection_resumes_committed_batch_without_model_replay(self):
        from knowledge_algorithm_fixtures import FakeModel, spec

        receipt = await self.upload(raw="第一段文字。\n\n第二段文字。\n\n第三段文字。".encode())
        scope, version = self.fixture.scope, UUID(receipt["version_id"])
        self.fixture.job, self.fixture.version = UUID(receipt["job_id"]), version
        await self.business.review(scope, self.library_id, version,
                                   decision="APPROVE", comment=None, parse_strategy="TEXT")
        await self.fixture.parse()

        class FailSecond(FakeModel):
            async def represent(self, value, specification):
                if self.calls:
                    raise AlgorithmError(ErrorCode.TIMEOUT, "合成模型超时", stage="projection_model")
                return await super().represent(value, specification)

        async def original(content):
            return content

        access = self.fixture.projections
        document = await access.allowed_document(scope, self.fixture.job, policy=original, policy_revision="fixture")
        profile = projection_profile(batch_size=1, representations=(spec(),))
        interrupted = FailSecond()
        builder = ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), interrupted)
        with self.assertRaises(AlgorithmError) as failure:
            await access.prepare(scope, self.fixture.job, document=document, builder=builder, profile=profile)
        self.assertEqual(failure.exception.code, ErrorCode.TIMEOUT)
        lease = await self.fixture.service.get_job(scope, self.fixture.job)
        async with self.uow_factory() as uow:
            identity, status, manifest = await uow.knowledge_ingestion.initial_projection_checkpoint(
                lease, builder.projection_ref(document, profile))
        self.assertEqual(status, "PREPARING")
        self.assertEqual(list(manifest["batches"]), ["1"])
        self.assertEqual(manifest["planned_batch_count"], 3)
        active = (await self.business.files(scope, self.library_id, view="processing"))["items"][0]["versions"][str(version)]
        self.assertEqual(active["index_progress"], {"committed_batches": 1, "total_batches": 3, "validated": False})

        real_read = self.fixture.store.read
        checkpoint_ref = manifest["batches"]["1"]["ref"]

        async def corrupted_read(handle, *, max_bytes):
            raw = await real_read(handle, max_bytes=max_bytes)
            return raw.replace(b'\"payload_hash\":\"', b'\"payload_hash\":\"invalid-') if handle == checkpoint_ref else raw

        resumed = FakeModel()
        builder = ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), resumed)
        with patch.object(self.fixture.store, "read", side_effect=corrupted_read):
            with self.assertRaisesRegex(AlgorithmError, "已提交索引批次摘要"):
                await access.prepare(scope, self.fixture.job, document=document, builder=builder, profile=profile)
        self.assertEqual(resumed.calls, [])
        from knowledge_retrieval_app.repositories.knowledge_ingestion import KnowledgeIngestionRepository
        with patch.object(KnowledgeIngestionRepository, "complete_generation", side_effect=RuntimeError("验证前中断")):
            with self.assertRaisesRegex(RuntimeError, "验证前中断"):
                await access.prepare(scope, self.fixture.job, document=document, builder=builder, profile=profile)
        awaiting = (await self.business.files(scope, self.library_id, view="processing"))["items"][0]["versions"][str(version)]
        self.assertEqual(awaiting["index_progress"], {"committed_batches": 3, "total_batches": 3, "validated": False})
        generation = await access.prepare(scope, self.fixture.job, document=document, builder=builder, profile=profile)
        self.assertEqual(generation, identity)
        self.assertEqual([call.text for call in resumed.calls], ["第二段文字。", "第三段文字。"])
        async with self.uow_factory() as uow:
            status, completed = await uow.knowledge_ingestion.projection_manifest(lease, generation)
        self.assertEqual(status, "READY")
        self.assertEqual(completed["unit_count"], 3)
        self.assertEqual(completed["batches"]["1"], manifest["batches"]["1"])
        row = (await self.business.files(scope, self.library_id, view="processing"))["items"][0]["versions"][str(version)]
        self.assertEqual((row["progress_stage"], row["status"]), ("READY", "PARSED"))
        self.assertEqual(row["index_progress"], {"committed_batches": 3, "total_batches": 3, "validated": True})

    async def test_formal_queue_resumes_projection_on_another_worker(self):
        from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker
        from knowledge_retrieval_app.application.knowledge_worker import KnowledgeBusinessWorker
        value = await self.upload()
        self.fixture.job, self.fixture.version = UUID(value["job_id"]), UUID(value["version_id"])
        reviewer = replace(self.fixture.scope, subject_ref=str(uuid4()))
        await self.business.update(self.fixture.scope, self.library_id, expected_revision=1, changes={"grants": {
            self.fixture.scope.subject_ref: ["READ", "UPLOAD", "MANAGE"],
            reviewer.subject_ref: ["READ", "REVIEW"],
        }})
        await self.business.review(reviewer, self.library_id, self.fixture.version,
                                   decision="APPROVE", comment=None, parse_strategy="TEXT")
        # 解析由 fixture 原 Worker 完成，索引由另一独立 Worker 接管。
        await self.fixture.parse()
        async def original(content):
            return content
        pipeline = KnowledgePipelineWorker(ingestion=self.fixture.service,
            projection=self.fixture.projections,
            builder_factory=lambda scope, document: ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), None),
            profile=projection_profile(), allowed_policy=original, policy_revision="synthetic/v1",
            worker_id="another-worker")
        worker = KnowledgeBusinessWorker(uow_factory=self.uow_factory, pipeline=pipeline,
                                         worker_id="another-worker", poll_seconds=.1)
        # 共享 fixture 曾直接准备了另一代次；Worker 先恢复其未确认的 outbox。
        self.assertTrue(await worker.process_once())
        from datetime import timedelta
        from knowledge_retrieval_app.repositories.knowledge_ingestion import KnowledgeIngestionRepository
        finish_outbox = KnowledgeIngestionRepository.finish_outbox

        async def fail_delivery(repo, scope, outbox_id, **kwargs):
            await finish_outbox(repo, scope, outbox_id, **kwargs)
            if not kwargs.get("error_code"):
                raise RuntimeError("模拟发布事务在提交前中断")

        with patch.object(KnowledgeIngestionRepository, "finish_outbox", fail_delivery):
            self.assertTrue(await worker.process_once())
        tasks = await self.business.files(self.fixture.scope, self.library_id, view="processing")
        pending = tasks["items"][0]["versions"][value["version_id"]]
        self.assertEqual((pending["status"], pending["progress_stage"]), ("PARSED", "INDEX_RETRY"))
        self.assertTrue(pending["generation_ready"])
        not_visible = await self.business.search(reviewer, self.library_id, KnowledgeSearchRequest(text="知识检索"))
        self.assertFalse(not_visible["primary_hits"])
        self.fixture.now += timedelta(seconds=10)
        self.assertTrue(await worker.process_once())
        rows = await self.business.files(self.fixture.scope, self.library_id, page=1, page_size=20)
        version = rows["items"][0]["versions"][value["version_id"]]
        self.assertEqual(version["processing_status"], "READY")
        self.assertTrue(version["generation_ready"])
        tasks = await self.business.files(self.fixture.scope, self.library_id, view="processing")
        self.assertEqual(tasks["items"][0]["versions"][value["version_id"]]["progress_stage"], "READY")
        self.assertFalse(await worker.process_once())
        self.assertEqual(version["status"], "PUBLISHED")
        self.assertEqual(version["generation_id"], pending["generation_id"])
        self.assertNotIn("publish", version["allowed_actions"])
        found = await self.business.search(reviewer, self.library_id, KnowledgeSearchRequest(text="知识检索"))
        self.assertIn(value["version_id"], {item["version_id"] for item in found["items"]})

    async def test_invalid_partial_updates_and_grants(self):
        from pydantic import ValidationError
        from platform_core.contracts.knowledge_app import KnowledgeLibraryUpdate
        for field in ("display_name", "status", "grants"):
            with self.assertRaises(ValidationError):
                KnowledgeLibraryUpdate.model_validate({"expected_revision": 1, field: None})
        for grants in ({self.fixture.scope.subject_ref: ["MANAGE"]},
                       {self.fixture.scope.subject_ref: ["READ", "MANAGE"], "reader": ["READ", "READ"]}):
            with self.assertRaises(AlgorithmError):
                await self.business.update(self.fixture.scope, self.library_id, expected_revision=1,
                                           changes={"grants": grants})

    async def test_reprocess_keeps_original_version_and_auditable_failed_job(self):
        manager = self.fixture.scope
        receipt = await self.upload()
        version_id, old_job = UUID(receipt["version_id"]), UUID(receipt["job_id"])
        await self.business.review(manager, self.library_id, version_id,
            decision="APPROVE", comment="原审核", parse_strategy="TEXT")
        async with self.uow_factory() as uow:
            repos = uow.knowledge_ingestion._repos(manager)
            job = await repos["knowledge_ingestion_job"].get(old_job)
            run = await repos["knowledge_parse_run"].get(job.parse_run_id)
            original_profile = run.profile_hash
            version = await repos["knowledge_document_version"].get(version_id)
            job.status = run.status = version.status = "FAILED"
            job.error_code = "INVALID_CONTRACT"
            await uow.commit()
        with self.assertRaises(AlgorithmError):
            await self.business.reprocess(replace(manager, subject_ref="outsider"), self.library_id, version_id,
                mode="ORIGINAL", expected_job_id=old_job, parse_strategy=None)
        with self.assertRaises(AlgorithmError):
            await self.business.reprocess(manager, self.library_id, version_id, mode="CURRENT",
                expected_job_id=old_job, parse_strategy="VISUAL")
        retried = await self.business.reprocess(manager, self.library_id, version_id, mode="ORIGINAL",
                                               expected_job_id=old_job, parse_strategy=None)
        with self.assertRaisesRegex(LibraryConflict, "TASK_REVISION_CONFLICT"):
            await self.business.reprocess(manager, self.library_id, version_id, mode="ORIGINAL",
                                          expected_job_id=old_job, parse_strategy=None)
        self.assertEqual(await self.upload(), receipt)
        page = await self.business.files(manager, self.library_id, view="processing", version_id=str(version_id))
        record = page["items"][0]["versions"][str(version_id)]
        self.assertEqual((record["job_id"], record["comment"], record["review"]),
                         (retried["job_id"], "原审核", "APPROVED"))
        self.assertEqual(record["parse_task_history"][0]["error_code"], "INVALID_CONTRACT")
        self.assertEqual(record["parse_task_history"][0]["job_id"], str(old_job))
        self.fixture.job, self.fixture.version = UUID(retried["job_id"]), version_id
        async with self.uow_factory() as uow:
            lease = await uow.knowledge_ingestion.get_job(manager, self.fixture.job)
            self.assertEqual(lease.profile.profile_hash, original_profile)
        self.assertEqual((await self.fixture.parse())["status"], "PARSED")
        self.assertFalse((await self.business.search(manager, self.library_id,
            KnowledgeSearchRequest(text="知识检索")))["primary_hits"])

    async def test_published_reprocess_keeps_serving_old_index_until_automatic_switch(self):
        from datetime import timedelta
        from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker
        from knowledge_retrieval_app.application.knowledge_worker import KnowledgeBusinessWorker
        from knowledge_retrieval_app.repositories.knowledge_ingestion import KnowledgeIngestionRepository

        manager = self.fixture.scope
        profile = projection_profile(representations=())

        async def original(content):
            return content

        async def resolve_index_profile(_access, _models):
            return profile

        self.business.resolve_index_profile = resolve_index_profile
        pipeline = KnowledgePipelineWorker(ingestion=self.fixture.service, projection=self.fixture.projections,
            builder_factory=lambda *_: ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), None),
            profile=profile, allowed_policy=original, policy_revision="synthetic/v1", worker_id="reprocess-worker")
        worker = KnowledgeBusinessWorker(uow_factory=self.uow_factory, pipeline=pipeline,
            worker_id="reprocess-worker", poll_seconds=.1)

        async def run_once():
            with patch("knowledge_retrieval_app.application.knowledge_worker.utcnow", lambda: self.fixture.now):
                return await worker.process_once()

        await run_once()  # 恢复共享 fixture 的已有 outbox。
        receipt = await self.upload()
        ref = receipt["version_id"]
        version_id = self.fixture.version = UUID(ref)
        old_job = self.fixture.job = UUID(receipt["job_id"])
        await self.business.review(manager, self.library_id, version_id, decision="APPROVE",
                                   comment="保留原审批", parse_strategy="TEXT")
        await self.fixture.parse()
        self.assertTrue(await run_once())

        async def processing():
            page = await self.business.files(manager, self.library_id, view="processing", version_id=ref)
            return page["items"][0]["versions"][ref]

        async def pointers():
            async with self.uow_factory() as uow:
                versions = uow.knowledge_ingestion._repos(manager)["knowledge_document_version"]
                version = await versions.get(version_id)
                return (version.active_generation_id, version.published_parse_run_id,
                        version.allowed_view_hash, version.allowed_view_manifest_ref)

        async def assert_searchable():
            result = await self.business.search(manager, self.library_id, KnowledgeSearchRequest(text="知识检索"))
            self.assertIn(ref, {item["version_id"] for item in result["items"]})

        before = await processing()
        self.assertEqual((before["status"], before["progress_stage"]), ("PUBLISHED", "READY"))
        self.assertTrue({"reprocess", "rebuild-index"}.issubset(before["allowed_actions"]))
        old_pointers = await pointers()
        await assert_searchable()

        action = dict(mode="ORIGINAL", expected_job_id=old_job, parse_strategy=None)
        result = await self.business.reprocess(manager, self.library_id, version_id, **action)
        new_job = self.fixture.job = UUID(result["job_id"])
        self.assertNotEqual(new_job, old_job)
        with self.assertRaisesRegex(LibraryConflict, "TASK_REVISION_CONFLICT"):
            await self.business.reprocess(manager, self.library_id, version_id, **action)
        with self.assertRaisesRegex(LibraryConflict, "INDEX_BATCH_CONFLICT"):
            await self.business.index_rebuild(manager, self.library_id, action="START", version_id=version_id,
                                              expected_generation_id=old_pointers[0])
        during = await processing()
        self.assertEqual((during["status"], during["progress_stage"], during["comment"]),
                         ("PUBLISHED", "PARSE_QUEUED", "保留原审批"))
        self.assertFalse({"reprocess", "rebuild-index"} & set(during["allowed_actions"]))
        self.assertEqual(during["parse_task_history"][0]["job_id"], str(old_job))
        self.assertEqual(await pointers(), old_pointers)
        await assert_searchable()

        self.assertEqual((await self.fixture.parse())["status"], "PARSED")
        self.assertEqual((await processing())["progress_stage"], "INDEX_QUEUED")
        self.assertEqual(await pointers(), old_pointers)
        finish = KnowledgeIngestionRepository.finish_outbox

        async def fail_before_commit(repo, scope, outbox_id, **kwargs):
            await finish(repo, scope, outbox_id, **kwargs)
            if not kwargs.get("error_code"):
                raise RuntimeError("模拟切换发布指针后的事务中断")

        with patch.object(KnowledgeIngestionRepository, "finish_outbox", fail_before_commit):
            self.assertTrue(await run_once())
        self.assertEqual((await processing())["progress_stage"], "INDEX_RETRY")
        self.assertEqual(await pointers(), old_pointers)
        await assert_searchable()

        disabled = await self.business.update(manager, self.library_id, expected_revision=1, changes={"status": "DISABLED"})
        self.fixture.now += timedelta(seconds=10)
        self.assertTrue(await run_once())
        self.assertEqual(await pointers(), old_pointers)
        self.assertFalse(await run_once())
        await self.business.update(manager, self.library_id, expected_revision=disabled["row_version"], changes={"status": "ACTIVE"})
        await assert_searchable()
        self.fixture.now += timedelta(seconds=16)
        self.assertTrue(await run_once())
        after = await processing()
        new_pointers = await pointers()
        self.assertTrue(all(new != old for new, old in zip(new_pointers, old_pointers)))
        self.assertEqual((after["status"], after["progress_stage"], after["generation_id"], after["job_id"]),
                         ("PUBLISHED", "READY", str(new_pointers[0]), str(new_job)))
        self.assertTrue({"reprocess", "rebuild-index"}.issubset(after["allowed_actions"]))
        self.assertFalse(await run_once())
        await assert_searchable()

    async def test_retry_index_requeues_same_outbox_without_new_parse_job(self):
        from sqlalchemy import select
        manager = self.fixture.scope
        receipt = await self.upload()
        self.fixture.version, self.fixture.job = UUID(receipt["version_id"]), UUID(receipt["job_id"])
        await self.business.review(manager, self.library_id, self.fixture.version,
                                   decision="APPROVE", comment=None, parse_strategy="TEXT")
        await self.fixture.parse()
        async with self.uow_factory() as uow:
            repos = uow.knowledge_ingestion._repos(manager)
            job = await repos["knowledge_ingestion_job"].get(self.fixture.job)
            job.context_json = {**job.context_json, "pipeline_status": "INDEX_FAILED",
                                "pipeline_failures": 5, "pipeline_error": "PROJECTION_FAILED"}
            outboxes = repos["knowledge_outbox"]
            outbox = (await uow.session.scalars(outboxes.statement().where(
                outboxes.entity.document_version_id == self.fixture.version))).one()
            outbox_id = outbox.outbox_id
            outbox.status, outbox.attempt = "FAILED", 8
            await uow.commit()
        queue = await self.business.files(manager, self.library_id, view="processing", version_id=receipt["version_id"])
        entry = queue["items"][0]["versions"][receipt["version_id"]]
        self.assertEqual((entry["parse_status"], entry["processing_status"]), ("PARSED", "INDEX_FAILED"))
        self.assertEqual(entry["progress_stage"], "INDEX_FAILED")
        self.assertIn("retry-index", entry["allowed_actions"])
        self.assertNotIn("reprocess", entry["allowed_actions"])
        self.assertNotIn("rebuild-index", entry["allowed_actions"])
        result = await self.business.retry_index(manager, self.library_id, self.fixture.version,
                                                expected_job_id=self.fixture.job)
        self.assertEqual(result["job_id"], receipt["job_id"])
        queue = await self.business.files(manager, self.library_id, view="processing", version_id=receipt["version_id"])
        self.assertEqual(queue["items"][0]["versions"][receipt["version_id"]]["progress_stage"], "INDEX_QUEUED")
        async with self.uow_factory() as uow:
            repos = uow.knowledge_ingestion._repos(manager)
            outbox = await repos["knowledge_outbox"].get(outbox_id)
            self.assertEqual((outbox.status, int(outbox.attempt)), ("PENDING", 8))
            jobs = repos["knowledge_ingestion_job"]
            rows = (await uow.session.scalars(select(jobs.entity).where(
                jobs.entity.document_version_id == self.fixture.version))).all()
            self.assertEqual([(job.ingestion_job_id, job.status) for job in rows], [(self.fixture.job, "PARSED")])
        with self.assertRaises(AlgorithmError):
            await self.business.retry_index(manager, self.library_id, self.fixture.version,
                                            expected_job_id=self.fixture.job)

    async def test_single_file_rebuild_automatically_publishes_without_reparsing(self):
        from knowledge_algorithm_fixtures import FakeModel
        from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker
        from knowledge_retrieval_app.application.knowledge_worker import KnowledgeBusinessWorker

        manager = self.fixture.scope
        model = FakeModel()
        profile = projection_profile(representations=())

        async def original(content):
            return content

        async def resolve(_access, models):
            self.assertEqual(models, {})
            return profile

        self.business.resolve_index_profile = resolve
        pipeline = KnowledgePipelineWorker(ingestion=self.fixture.service, projection=self.fixture.projections,
            builder_factory=lambda *_: ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), model),
            profile=profile, allowed_policy=original, policy_revision="synthetic/v1", worker_id="file-index-test")
        worker = KnowledgeBusinessWorker(uow_factory=self.uow_factory, pipeline=pipeline,
            worker_id="file-index-test", poll_seconds=.1)
        await worker.process_once()
        receipt = await self.upload(key="index-one")
        version_id = self.fixture.version = UUID(receipt["version_id"])
        self.fixture.job = UUID(receipt["job_id"])
        await self.business.review(manager, self.library_id, version_id, decision="APPROVE",
                                   comment=None, parse_strategy="TEXT")
        await self.fixture.parse()
        # 停用时即使生成完整索引，也不自动发布；重新启用后保留已有治理入口。
        disabled = await self.business.update(manager, self.library_id, expected_revision=1, changes={"status": "DISABLED"})
        self.assertTrue(await worker.process_once())
        await self.business.update(manager, self.library_id, expected_revision=disabled["row_version"], changes={"status": "ACTIVE"})

        async def current():
            directory = await self.business.files(manager, self.library_id, version_id=str(version_id), view="processing")
            return directory["items"][0]["versions"][str(version_id)]

        before = await current()
        self.assertEqual(before["status"], "PARSED")
        self.assertIn("rebuild-index", before["allowed_actions"])
        start = dict(action="START", version_id=version_id)
        with self.assertRaises(AlgorithmError):
            await self.business.index_rebuild(replace(manager, subject_ref="outsider"), self.library_id, **start,
                                              expected_generation_id=UUID(before["generation_id"]))
        with self.assertRaisesRegex(LibraryConflict, "INDEX_BATCH_CONFLICT"):
            await self.business.index_rebuild(manager, self.library_id, **start, expected_generation_id=uuid4())
        batch = (await self.business.index_rebuild(manager, self.library_id, **start,
                    expected_generation_id=UUID(before["generation_id"])))["batch"]
        self.assertEqual((batch["scope"], batch["source_published"], batch["status"]), ("FILE", False, "RUNNING"))
        total = before["index_progress"]["total_batches"]
        self.assertEqual(batch["items"][0]["index_progress"],
                         {"committed_batches": 0, "total_batches": total, "validated": False})
        queued = await current()
        self.assertEqual((queued["progress_stage"], queued["generation_id"]), ("REBUILD_QUEUED", before["generation_id"]))
        self.assertEqual(queued["index_progress"], batch["items"][0]["index_progress"])
        self.assertNotIn("publish", queued["allowed_actions"])
        self.assertTrue(await worker.process_once())
        self.assertFalse(await worker.process_once())
        candidate_batch = (await self.business.index_rebuild(manager, self.library_id, version_id=version_id))["batch"]
        self.assertEqual(candidate_batch["status"], "PUBLISHED")
        self.assertEqual(candidate_batch["items"][0]["index_progress"],
                         {"committed_batches": total, "total_batches": total, "validated": True})
        async with self.uow_factory() as uow:
            row = await uow.library(manager).authorized(self.library_id, "MANAGE")
            candidate = UUID(row.configuration_json["file_index_rebuilds"][str(version_id)]["members"][str(version_id)]["candidate_id"])
        after = await current()
        self.assertEqual((after["status"], after["generation_id"], after["job_id"], after["progress_stage"]),
                         ("PUBLISHED", str(candidate), before["job_id"], "REBUILD_COMPLETE"))
        self.assertFalse(model.calls)
        result = await self.business.search(manager, self.library_id, KnowledgeSearchRequest(text="知识检索"))
        self.assertIn(str(version_id), {item["version_id"] for item in result["items"]})
        # 模拟旧版已交付但仍等待切换的批次；新 Worker 启动时复用候选，不重新解析或索引。
        from unittest.mock import AsyncMock
        await self.business.index_rebuild(manager, self.library_id, action="START",
            version_id=version_id, expected_generation_id=candidate)
        with patch.object(KnowledgeLibraryRepository, "complete_rebuild", AsyncMock()):
            self.assertTrue(await worker.process_once())
        self.assertEqual((await self.business.index_rebuild(manager, self.library_id, version_id=version_id))["batch"]["status"], "READY")
        self.assertEqual((await current())["generation_id"], str(candidate))
        await worker.recover_ready_rebuilds()
        self.assertEqual((await self.business.index_rebuild(manager, self.library_id, version_id=version_id))["batch"]["status"], "PUBLISHED")
        self.assertNotEqual((await current())["generation_id"], str(candidate))
        # 旧版也可能已采用候选，但还没有填写文件的发布指针。
        async with self.uow_factory() as uow:
            row = await uow.library(manager).authorized(self.library_id, "MANAGE")
            config = dict(row.configuration_json)
            batch = {**config.pop("file_index_rebuilds")[str(version_id)], "source_published": False}
            row.configuration_json = {**config, "index_rebuild": batch}
            repos = uow.knowledge_ingestion._repos(manager)
            version = await repos["knowledge_document_version"].get(version_id)
            legacy_generation = await repos["knowledge_index_generation"].get(version.active_generation_id)
            manifest = dict(legacy_generation.manifest_json)
            manifest["adopted_from_batch"] = manifest.pop("rebuild_id")
            legacy_generation.manifest_json = manifest
            version.active_generation_id = version.published_parse_run_id = None
            version.status = "PARSED"
            document = await repos["knowledge_document"].get(version.document_id)
            document.status = "PARSED"
            await uow.commit()
        self.assertEqual((await self.business.index_rebuild(manager, self.library_id, version_id=version_id))["batch"]["status"], "READY")
        await worker.recover_ready_rebuilds()
        self.assertEqual((await current())["status"], "PUBLISHED")

    async def test_deterministic_index_failure_stops_and_manual_index_retry_keeps_parse_and_publishes(self):
        from unittest.mock import AsyncMock
        from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker
        from knowledge_retrieval_app.application.knowledge_worker import KnowledgeBusinessWorker

        manager = self.fixture.scope
        profile = projection_profile(max_tokens=32)
        async def original(content):
            return content
        service = self.fixture.projections
        pipeline = KnowledgePipelineWorker(ingestion=self.fixture.service, projection=service,
            builder_factory=lambda *_: ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), None),
            profile=profile, allowed_policy=original, policy_revision="synthetic/v1", worker_id="retry-split-test")
        worker = KnowledgeBusinessWorker(uow_factory=self.uow_factory, pipeline=pipeline,
            worker_id="retry-split-test", poll_seconds=.1)
        await worker.process_once()
        receipt = await self.upload(key="retry-index-budget", raw=("内部控制程序验证" * 20).encode())
        version_id = self.fixture.version = UUID(receipt["version_id"])
        job_id = self.fixture.job = UUID(receipt["job_id"])
        await self.business.review(manager, self.library_id, version_id, decision="APPROVE",
                                   comment=None, parse_strategy="TEXT")
        await self.fixture.parse()
        original_run = (await self.fixture.service.get_job(manager, job_id)).identity.parse_run_ref
        fake = AsyncMock(side_effect=AlgorithmError(ErrorCode.RESOURCE_LIMIT, "合成旧版切片失败", stage="split"))
        with patch.object(service, "prepare", fake):
            self.assertTrue(await worker.process_once())
            self.assertFalse(await worker.process_once())
        fake.assert_awaited_once()

        async def row():
            directory = await self.business.files(manager, self.library_id, version_id=str(version_id), view="processing")
            return directory["items"][0]["versions"][str(version_id)]
        failed = await row()
        self.assertEqual((failed["processing_status"], failed["parse_status"]), ("INDEX_FAILED", "PARSED"))
        self.assertIn("retry-index", failed["allowed_actions"])
        queued = await self.business.retry_index(manager, self.library_id, version_id, expected_job_id=job_id)
        self.assertEqual(queued["job_id"], str(job_id))
        self.assertTrue(await worker.process_once())
        self.assertEqual((await row())["status"], "PUBLISHED")
        self.assertEqual((await self.fixture.service.get_job(manager, job_id)).identity.parse_run_ref, original_run)

    async def test_old_index_is_resplit_from_frozen_view_and_automatically_replaces_active_generation(self):
        from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker
        from knowledge_retrieval_app.application.knowledge_worker import KnowledgeBusinessWorker
        from knowledge_algorithms.projections.splitting import REVISION

        manager = self.fixture.scope
        profile = projection_profile(max_tokens=140)
        async def original(content):
            return content
        pipeline = KnowledgePipelineWorker(ingestion=self.fixture.service, projection=self.fixture.projections,
            builder_factory=lambda *_: ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), None),
            profile=profile, allowed_policy=original, policy_revision="synthetic/v1", worker_id="old-split-test")
        worker = KnowledgeBusinessWorker(uow_factory=self.uow_factory, pipeline=pipeline,
            worker_id="old-split-test", poll_seconds=.1)
        await worker.process_once()
        receipt = await self.upload(key="old-split", raw=("内部控制程序验证" * 12).encode())
        version_id = self.fixture.version = UUID(receipt["version_id"])
        self.fixture.job = UUID(receipt["job_id"])
        await self.business.review(manager, self.library_id, version_id, decision="APPROVE",
                                   comment=None, parse_strategy="TEXT")
        await self.fixture.parse()
        self.assertTrue(await worker.process_once())
        lease = await self.fixture.service.get_job(manager, self.fixture.job)
        async with self.uow_factory() as uow:
            repos = uow.knowledge_ingestion._repos(manager)
            version = await repos["knowledge_document_version"].get(version_id)
            original_id = version.active_generation_id
            parse_run = version.published_parse_run_id
            old = await repos["knowledge_index_generation"].get(original_id)
            self.assertEqual(old.manifest_json["unit_count"], 1)
            old.manifest_json = {k: v for k, v in old.manifest_json.items() if k != "split_revision"}
            await uow.commit()

        async def resolve(_access, _models):
            return replace(profile, max_tokens=20)
        self.business.resolve_index_profile = resolve
        await self.business.index_rebuild(manager, self.library_id, action="START",
            version_id=version_id, expected_generation_id=original_id)
        async with self.uow_factory() as uow:
            version = await uow.knowledge_ingestion._repos(manager)["knowledge_document_version"].get(version_id)
            self.assertEqual(version.active_generation_id, original_id)
        self.assertTrue(await worker.process_once())
        async with self.uow_factory() as uow:
            version = await uow.knowledge_ingestion._repos(manager)["knowledge_document_version"].get(version_id)
            self.assertNotEqual(version.active_generation_id, original_id)
            self.assertEqual(version.published_parse_run_id, parse_run)
            status, new = await uow.knowledge_ingestion.projection_manifest(lease, version.active_generation_id)
            self.assertEqual(status, "READY")
            self.assertEqual(new["split_revision"], REVISION)
            self.assertEqual(new["planned_batch_count"], new["batch_count"])
            self.assertGreater(new["unit_count"], 1)

    async def test_rebuild_uses_existing_parse_and_switches_only_in_successful_batch_transaction(self):
        from unittest.mock import patch
        from knowledge_algorithm_fixtures import FakeModel, spec
        from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker
        from knowledge_retrieval_app.application.knowledge_worker import KnowledgeBusinessWorker
        from knowledge_retrieval_app.repositories.knowledge_retrieval import KnowledgeRetrievalRepository

        manager = self.fixture.scope
        dense_id, visual_id, replacement_id = str(uuid4()), str(uuid4()), str(uuid4())
        dense = replace(spec(), model_ref=dense_id, inference_ref="dense", binding_revision="a" * 64)
        visual = replace(spec("MULTIVECTOR", "IMAGE", "IMAGE_TO_IMAGE"), model_ref=visual_id,
                         inference_ref="visual", binding_revision="b" * 64)
        next_dense = replace(dense, model_ref=replacement_id, binding_revision="c" * 64)
        model = FakeModel()
        stored = await self.business.update(manager, self.library_id, expected_revision=1,
                                            changes={"index_models": {"embedding": dense_id}})
        async def original(content):
            return content
        pipeline = KnowledgePipelineWorker(ingestion=self.fixture.service, projection=self.fixture.projections,
            builder_factory=lambda *_: ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), model),
            profile=projection_profile(representations=(dense,)), allowed_policy=original,
            policy_revision="synthetic/v1", worker_id="index-rebuild-test")
        worker = KnowledgeBusinessWorker(uow_factory=self.uow_factory, pipeline=pipeline,
            worker_id="index-rebuild-test", poll_seconds=.1)
        await worker.process_once()  # 恢复共享 fixture 已就绪的初始 outbox
        versions, original_ids = [], []
        for key in ("first", "second"):
            receipt = await self.upload(key=key)
            self.fixture.version, self.fixture.job = UUID(receipt["version_id"]), UUID(receipt["job_id"])
            versions.append(self.fixture.version)
            await self.business.review(manager, self.library_id, self.fixture.version, decision="APPROVE",
                                       comment=None, parse_strategy="TEXT")
            await self.fixture.parse()
            self.assertTrue(await worker.process_once())
            directory = await self.business.files(manager, self.library_id, version_id=receipt["version_id"])
            generation_id = UUID(directory["items"][0]["versions"][receipt["version_id"]]["generation_id"])
            original_ids.append(str(generation_id))
        async def resolve(_access, models):
            return projection_profile(representations=tuple(entry for entry in (dense, visual, next_dense)
                                                              if entry.model_ref in models.values()))
        self.business.resolve_index_profile = resolve
        model.calls.clear()
        started = await self.business.update(manager, self.library_id, expected_revision=stored["row_version"],
            changes={"index_models": {"embedding": dense_id, "visual_embedding": visual_id}, "rebuild_indexes": True})
        batch_id = UUID(started["index_maintenance"]["batch"]["batch_id"])
        self.assertEqual((await self.business.index_rebuild(manager, self.library_id,
            version_id=versions[0]))["batch"]["batch_id"], str(batch_id))
        with self.assertRaisesRegex(LibraryConflict, "^KNOWLEDGE_INDEX_BATCH_ACTIVE$"):
            await self.business.index_rebuild(manager, self.library_id, action="START", version_id=versions[0],
                                              expected_generation_id=UUID(original_ids[0]))
        with self.assertRaisesRegex(AlgorithmError, "未登记索引批次操作"):
            await self.business.index_rebuild(manager, self.library_id, action="PUBLISH", batch_id=batch_id)
        self.assertTrue(await worker.process_once())
        async def pointers():
            async with self.uow_factory() as uow:
                return await uow.knowledge_retrieval.version_records(manager, tuple(versions))
        before = await pointers()
        self.assertCountEqual([row["generation_id"] for row in before], original_ids)
        publish = KnowledgeRetrievalRepository.publish
        count = 0
        async def fail_second(repo, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("模拟中途失败")
            await publish(repo, *args, **kwargs)
            await repo._session.flush()
        with patch.object(KnowledgeRetrievalRepository, "publish", fail_second):
            self.assertTrue(await worker.process_once())
        self.assertEqual(count, 2)
        self.assertEqual(await pointers(), before)
        from datetime import timedelta
        self.fixture.now += timedelta(seconds=10)
        self.assertTrue(await worker.process_once())
        self.assertFalse(await worker.process_once())
        final = await self.business.index_rebuild(manager, self.library_id)
        after = await pointers()
        self.assertEqual(final["batch"]["status"], "PUBLISHED")
        self.assertTrue(all(item["visual_not_applicable"] for item in final["batch"]["items"]))
        self.assertEqual(model.calls, [])
        self.assertTrue(all(old["generation_id"] != new["generation_id"] and old["parse_run_id"] == new["parse_run_id"]
                            for old, new in zip(before, after)))
        # 仅保存后另行发起。模拟其中一项用完重试额度，不替换任何已发布指针。
        revision = (await self.business.get(manager, self.library_id))["row_version"]
        await self.business.update(manager, self.library_id, expected_revision=revision,
            changes={"index_models": {"embedding": replacement_id, "visual_embedding": visual_id}})
        pending = await self.business.index_rebuild(manager, self.library_id)
        self.assertEqual(pending["pending_roles"], ["embedding"])
        self.assertEqual(pending["batch"]["status"], "PUBLISHED")
        old_by_version = {record["version_id"]: record for record in after}
        first = old_by_version[str(versions[0])]
        single = (await self.business.index_rebuild(manager, self.library_id, action="START",
            version_id=versions[0],
            expected_generation_id=UUID(first["generation_id"])))["batch"]
        self.assertEqual((single["scope"], single["source_published"]), ("FILE", True))
        bulk_view = await self.business.index_rebuild(manager, self.library_id)
        with self.assertRaisesRegex(LibraryConflict, "^KNOWLEDGE_INDEX_BATCH_ACTIVE$"):
            await self.business.index_rebuild(manager, self.library_id, action="START",
                                             expected_revision=bulk_view["library_revision"])
        self.assertTrue(await worker.process_once())
        activated = await self.business.index_rebuild(manager, self.library_id, version_id=versions[0])
        self.assertEqual(activated["batch"]["batch_id"], single["batch_id"])
        self.assertEqual(activated["batch"]["status"], "PUBLISHED")
        self.assertEqual(activated["pending_roles"], ["embedding"])
        after = await pointers()
        new_by_version = {record["version_id"]: record for record in after}
        self.assertEqual(new_by_version[str(versions[1])], old_by_version[str(versions[1])])
        self.assertEqual(new_by_version[str(versions[0])]["parse_run_id"], first["parse_run_id"])
        self.assertNotEqual(new_by_version[str(versions[0])]["generation_id"], first["generation_id"])
        status = await self.business.index_rebuild(manager, self.library_id, action="START",
                                                 expected_revision=activated["library_revision"])
        new_batch_id = UUID(status["batch"]["batch_id"])
        target_version = versions[1]  # 仍使用旧模型的文件才会触发模型调用
        async with self.uow_factory() as uow:
            row = await uow.library(manager).authorized(self.library_id, "MANAGE")
            ref = row.configuration_json["index_rebuild"]["members"][str(target_version)]["outbox_id"]
            outbox = await uow.knowledge_ingestion._repos(manager)["knowledge_outbox"].get(UUID(ref))
            outbox.attempt = 4
            await uow.commit()
        from unittest.mock import AsyncMock
        from knowledge_algorithms.contracts.errors import ErrorCode
        timeout = AlgorithmError(ErrorCode.TIMEOUT, "合成超时", stage="index_rebuild_model")
        with patch.object(model, "represent", AsyncMock(side_effect=timeout)):
            self.assertTrue(await worker.process_once())
            self.assertTrue(await worker.process_once())
        failed = (await self.business.index_rebuild(manager, self.library_id))["batch"]
        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual({item["status"] for item in failed["items"]}, {"FAILED", "READY"})
        self.assertEqual(await pointers(), after)
        await self.business.index_rebuild(manager, self.library_id, action="RETRY", batch_id=new_batch_id,
                                          version_id=target_version)
        model.calls.clear()
        self.assertTrue(await worker.process_once())
        self.assertFalse(await worker.process_once())
        self.assertTrue(model.calls)
        self.assertTrue(all(call.kind == "TEXT" and call.input_role == "DOCUMENT" for call in model.calls))
        self.assertEqual((await self.business.index_rebuild(manager, self.library_id))["batch"]["status"], "PUBLISHED")
        # 同库两个文件分别排队，交付一个文件只会切换自己的索引。
        sources = {record["version_id"]: record["generation_id"] for record in await pointers()}
        one, two = versions
        async def submit(ref):
            return await self.business.index_rebuild(manager, self.library_id, action="START", version_id=ref,
                                                     expected_generation_id=UUID(sources[str(ref)]))
        async def file_batch(ref):
            return (await self.business.index_rebuild(manager, self.library_id, version_id=ref))["batch"]
        async def directory_progress():
            directory = await self.business.files(manager, self.library_id, view="processing")
            return {key: value["progress_stage"] for entry in directory["items"]
                    for key, value in entry["versions"].items() if key in sources}
        first_result = await submit(one)
        self.assertIsNone(await file_batch(two))
        with self.assertRaisesRegex(LibraryConflict, "^KNOWLEDGE_FILE_INDEX_BATCH_ACTIVE$"):
            await submit(one)
        second_result = await submit(two)
        self.assertNotEqual(first_result["batch"]["batch_id"], second_result["batch"]["batch_id"])
        self.assertEqual(second_result["active_file_rebuild_count"], 2)
        bulk_view = await self.business.index_rebuild(manager, self.library_id)
        self.assertEqual(bulk_view["batch"]["batch_id"], str(new_batch_id))
        self.assertEqual(await directory_progress(), dict.fromkeys(sources, "REBUILD_QUEUED"))
        cancelled = await self.business.index_rebuild(manager, self.library_id, action="DISCARD",
            batch_id=UUID(first_result["batch"]["batch_id"]))
        self.assertEqual((cancelled["batch"]["status"], cancelled["active_file_rebuild_count"]), ("DISCARDED", 1))
        self.assertEqual(await file_batch(two), second_result["batch"])
        first_result = await submit(one)
        self.assertNotEqual(first_result["batch"]["batch_id"], cancelled["batch"]["batch_id"])
        self.assertTrue(await worker.process_once())
        progress = await directory_progress()
        self.assertCountEqual(progress.values(), ["REBUILD_COMPLETE", "REBUILD_QUEUED"])
        for record in await pointers():
            ref = record["version_id"]
            self.assertEqual(record["generation_id"] != sources[ref], progress[ref] == "REBUILD_COMPLETE")
        self.assertTrue(await worker.process_once())
        self.assertFalse(await worker.process_once())
        for ref, receipt in ((one, first_result), (two, second_result)):
            batch = await file_batch(ref)
            self.assertEqual(batch["batch_id"], receipt["batch"]["batch_id"])
            self.assertEqual(batch["status"], "PUBLISHED")
        self.assertEqual(await directory_progress(), dict.fromkeys(sources, "REBUILD_COMPLETE"))
        self.assertEqual((await self.business.index_rebuild(manager, self.library_id))["active_file_rebuild_count"], 0)
        # 解绑即时屏蔽向量，已发布代次与关键词仍可用。
        self.service.published_model_spaces = True
        bound_scope = await self.service.create_scope(manager, resource_ref="library:" + str(self.library_id))
        _, profile = await self.service.backend(manager, bound_scope, "default")
        self.assertEqual([channel.representation_spec.model_ref for channel in profile.channels
                          if channel.kind == "DENSE"], [replacement_id])
        revision = (await self.business.get(manager, self.library_id))["row_version"]
        await self.business.update(manager, self.library_id, expected_revision=revision, changes={"index_models": {}})
        fresh_scope = await self.service.create_scope(manager, resource_ref="library:" + str(self.library_id))
        _, profile = await self.service.backend(manager, fresh_scope, "default")
        self.assertEqual([channel.kind for channel in profile.channels], ["LEXICAL"])
        revision = (await self.business.get(manager, self.library_id))["row_version"]
        await self.business.update(manager, self.library_id, expected_revision=revision,
                                   changes={"index_models": {"embedding": dense_id}})
        reattached = await self.service.create_scope(manager, resource_ref="library:" + str(self.library_id))
        _, profile = await self.service.backend(manager, reattached, "default")
        self.assertEqual([channel.kind for channel in profile.channels], ["LEXICAL"])
        revision = (await self.business.get(manager, self.library_id))["row_version"]
        queued = await self.business.index_rebuild(manager, self.library_id, action="START",
                                                 expected_revision=revision)
        obsolete = UUID(queued["batch"]["batch_id"])
        await self.business.withdraw(manager, self.library_id, versions[0])
        self.assertEqual((await self.business.index_rebuild(manager, self.library_id))["batch"]["status"], "STALE")
        discarded = await self.business.index_rebuild(manager, self.library_id, action="DISCARD", batch_id=obsolete)
        self.assertEqual(discarded["batch"]["status"], "DISCARDED")
        self.assertFalse(await worker.process_once())

    async def test_public_query_rejects_scope_models_and_unregistered_policy(self):
        from pydantic import ValidationError
        for key in ("scope", "profile_ref", "model_id", "owner_binding_ref"):
            with self.assertRaises(ValidationError):
                KnowledgeSearchRequest.model_validate({"text": "test", key: "injected"})
        with self.assertRaises(ValidationError):
            KnowledgeSearchRequest(text="x", mode="FILTERED_ORDER")
        with self.assertRaises(AlgorithmError):
            await self.business.create(self.fixture.scope, display_name="非法配置",
                                       description=None, profile_ref="unregistered")

    async def test_library_can_clear_text_embedding_and_enable_without_it(self):
        manager = self.fixture.scope
        current = await self.business.get(manager, self.library_id, management=True)
        self.assertEqual((current["status"], current["index_models"]), ("ACTIVE", {}))
        bound = await self.business.update(manager, self.library_id, expected_revision=current["row_version"],
                                           changes={"index_models": {"embedding": str(uuid4())}, "status": "DISABLED"})
        enabled = await self.business.update(manager, self.library_id, expected_revision=bound["row_version"],
                                             changes={"index_models": {"embedding": None}, "status": "ACTIVE"})
        self.assertEqual(enabled["status"], "ACTIVE")
        self.assertEqual(enabled["index_models"], {})


if __name__ == "__main__":
    unittest.main()
