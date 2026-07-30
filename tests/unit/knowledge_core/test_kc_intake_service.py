"""State-machine checks for the final KC intake database transaction."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

from knowledge_core.application.intake import (
    AbortIntakeCommand, AcceptKmAssetCommand, IntakeConflictError,
    KnowledgeCoreIntakeService, PublishedAttachment, PublishedManifest,
    ReserveIntakeCommand, ReviewIntakeCommand,
    PreparePublishCommand,
)
from knowledge_core.domain.intake import KmAssetIntakeManifest
from knowledge_core.domain.manifest import render_bundle_manifest
from platform_core.identity import uuid7


def manifest():
    return KmAssetIntakeManifest.model_validate({
        "bundle": {"source_id": "A-1", "source_revision": "r1", "title": "Asset", "security_level": 2},
        "documents": [{"part_name": "attachment_0", "external_document_id": "doc-1", "declared_mime_type": "text/plain", "ordinal": 0, "byte_size": 3, "content_sha256": "a" * 64}],
    })


class Repository:
    def __init__(self, values=None):
        self.values, self.added, self.deleted = values or {}, [], []
    async def get_by_scope_key(self, **kwargs): return self.values.get("collection")
    async def get_by_id(self, **kwargs): return self.values.get("collection")
    async def get_by_idempotency_key(self, **kwargs): return self.values.get("receipt")
    async def get_by_source(self, **kwargs): return self.values.get("bundle")
    async def get_by_source_revision(self, **kwargs): return self.values.get("revision")
    async def get_by_external_id(self, **kwargs): return self.values.get("document")
    async def get_by_content_hash(self, **kwargs): return self.values.get("version")
    async def get_by_input(self, **kwargs): return self.values.get("parse_view")
    async def next_revision_no(self, **kwargs): return 1
    async def next_version_no(self, **kwargs): return 1
    async def add(self, entity):
        self.added.append(entity)
        for field, value in (("bundle_id", 10), ("bundle_revision_id", 20), ("document_id", 30), ("document_version_id", 40), ("parse_view_id", 50)):
            if hasattr(entity, field) and getattr(entity, field) is None: setattr(entity, field, value)
        return entity
    async def delete_reserved_unreferenced(self, **kwargs):
        self.deleted.append(("documents", kwargs))
    async def delete_empty_unreferenced(self, **kwargs):
        self.deleted.append(("bundle", kwargs))


class Uow:
    def __init__(self, values):
        self.collections = Repository(values); self.receipts = Repository(values); self.bundles = Repository(values)
        self.revisions = Repository(values); self.documents = Repository(values); self.versions = Repository(values)
        self.members = Repository(values); self.jobs = Repository(values); self.commit = AsyncMock()
        self.parse_views = Repository(values)
        self.session = SimpleNamespace(flush=AsyncMock())
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return None


class IntakeServiceTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _collection(*, parser_vlm_model_id=None):
        models = {
            "parser_llm": str(uuid7()),
            "retrieval_llm": str(uuid7()),
            "embedding": str(uuid7()),
        }
        if parser_vlm_model_id:
            models["parser_vlm"] = str(parser_vlm_model_id)
        return SimpleNamespace(
            collection_id=1,
            status="ACTIVE",
            models_json=models,
        )

    def test_parser_policy_cannot_override_collection_embedding_model(self):
        with self.assertRaisesRegex(ValueError, "retrieval embeddings"):
            KnowledgeCoreIntakeService(
                receipt_ttl_seconds=60,
                uow_factory=lambda: None,
                parse_policy_overrides={"embedding_model_id": 17},
            )

    def _command(self):
        intake_manifest = manifest()
        rendered = render_bundle_manifest(intake_manifest.bundle)
        return AcceptKmAssetCommand(1, "assets", "svc:portal", "k1", intake_manifest, {
            "attachment_0": PublishedAttachment("doc-1", "memory://doc-1", "text/plain"),
        }, PublishedManifest("memory://manifest", len(rendered.content), rendered.content_sha256))

    async def test_creates_processing_revision_members_and_parse_job(self):
        values = {"collection": self._collection()}
        uow = Uow(values)
        service = KnowledgeCoreIntakeService(receipt_ttl_seconds=60, uow_factory=lambda: uow)

        accepted = await service.accept_published(self._command())

        self.assertEqual(10, accepted.bundle_id)
        self.assertEqual("PROCESSING", uow.revisions.added[0].status)
        self.assertEqual("MANIFEST", uow.members.added[0].document_role)
        self.assertEqual("RECEIVED", uow.members.added[1].member_status)
        self.assertEqual(2, len(uow.jobs.added))
        self.assertEqual(50, uow.jobs.added[0].parse_view_id)
        self.assertEqual("ACCEPTED", uow.receipts.added[0].receipt_status)
        uow.commit.assert_awaited_once()

    async def test_user_upload_waits_for_review_without_parse_job(self):
        values = {"collection": self._collection()}
        uow = Uow(values)
        service = KnowledgeCoreIntakeService(
            receipt_ttl_seconds=60,
            uow_factory=lambda: uow,
        )

        accepted = await service.accept_published(
            replace(self._command(), approval_required=True)
        )

        self.assertEqual("PENDING_REVIEW", accepted.acceptance_status)
        self.assertEqual("PENDING_REVIEW", uow.revisions.added[0].status)
        self.assertEqual("PENDING", uow.revisions.added[0].approval_status)
        self.assertEqual([], uow.jobs.added)
        self.assertEqual([], uow.parse_views.added)
        self.assertEqual(
            "PENDING_REVIEW",
            uow.receipts.added[0].receipt_status,
        )

    async def test_approval_atomically_creates_parse_job(self):
        revision_id = uuid7()
        collection = self._collection()
        collection.collection_key = "assets"
        revision = SimpleNamespace(
            collection_id=collection.collection_id,
            bundle_id=uuid7(),
            bundle_revision_id=revision_id,
            source_revision="r1",
            title="User Upload",
            approval_status="PENDING",
            status="PENDING_REVIEW",
            reviewed_by=None,
            reviewed_at=None,
            review_comment=None,
            updated_by=None,
            completed_at=None,
        )
        version = SimpleNamespace(
            document_version_id=uuid7(),
            detected_mime_type="text/plain",
            content_hash="a" * 64,
        )
        member = SimpleNamespace(
            member_status="RECEIVED",
            document_version_id=version.document_version_id,
            document_role="CONTENT",
            declared_name="guide.txt",
            external_document_id="doc-1",
        )
        receipt = SimpleNamespace(
            receipt_status="PENDING_REVIEW",
            updated_by=None,
        )
        values = {"collection": collection}
        uow = Uow(values)
        uow.session = SimpleNamespace(flush=AsyncMock())
        uow.revisions.get_by_id = AsyncMock(return_value=revision)
        uow.collections.get_by_id_scope = AsyncMock(
            return_value=collection
        )
        uow.members.list_by_revision = AsyncMock(return_value=[member])
        uow.versions.get_by_id = AsyncMock(return_value=version)
        uow.receipts.list_by_revision = AsyncMock(
            return_value=[receipt]
        )
        service = KnowledgeCoreIntakeService(
            receipt_ttl_seconds=60,
            uow_factory=lambda: uow,
        )

        result = await service.review(
            ReviewIntakeCommand(
                domain_id=1,
                collection_key="assets",
                bundle_revision_id=revision_id,
                decision="APPROVE",
                actor_id="user:reviewer",
                comment="内容有效",
            )
        )

        self.assertEqual("APPROVED", result.approval_status)
        self.assertEqual("PROCESSING", result.revision_status)
        self.assertEqual(1, len(uow.jobs.added))
        self.assertEqual("PENDING", uow.jobs.added[0].job_status)
        self.assertEqual("ACCEPTED", receipt.receipt_status)
        self.assertEqual("user:reviewer", revision.reviewed_by)
        uow.commit.assert_awaited_once()

    async def test_pdf_with_auto_vlm_uses_hybrid_parse_view(self):
        values = {
            "collection": self._collection(
                parser_vlm_model_id=uuid7()
            )
        }
        uow = Uow(values)
        service = KnowledgeCoreIntakeService(
            receipt_ttl_seconds=60,
            uow_factory=lambda: uow,
            parse_policy_overrides={"parse_strategy": "AUTO"},
        )
        version = SimpleNamespace(
            document_version_id=40,
            detected_mime_type="application/pdf",
            content_hash="a" * 64,
        )

        await service._enqueue_parse(
            uow,
            collection_id=1,
            bundle_revision_id=20,
            version=version,
            actor_id="svc:test",
        )

        self.assertEqual("HYBRID", uow.parse_views.added[0].view_kind)
        self.assertEqual(
            "kc-adaptive-visual-pipeline",
            uow.parse_views.added[0].parser_name,
        )

    async def test_deepseek_ocr_disables_docling_builtin_ocr(self):
        values = {"collection": self._collection()}
        uow = Uow(values)
        service = KnowledgeCoreIntakeService(
            receipt_ttl_seconds=60,
            uow_factory=lambda: uow,
            parse_policy_overrides={
                "ocr_model": "deepseek-ocr-2",
            },
        )
        version = SimpleNamespace(
            document_version_id=40,
            detected_mime_type="application/pdf",
            content_hash="a" * 64,
        )

        await service._enqueue_parse(
            uow,
            collection_id=1,
            bundle_revision_id=20,
            version=version,
            actor_id="svc:test",
        )

        policy = uow.parse_views.added[0].parse_config_json
        self.assertFalse(policy["do_ocr"])
        self.assertEqual("DEEPSEEK_OCR", policy["ocr_provider"])
        self.assertEqual(
            "kc-deepseek-ocr-pipeline",
            uow.parse_views.added[0].parser_name,
        )

    async def test_rejects_changed_snapshot_for_same_source_revision(self):
        values = {
            "collection": self._collection(),
            "bundle": SimpleNamespace(bundle_id=10, collection_id=1),
            "revision": SimpleNamespace(snapshot_fingerprint="different", bundle_revision_id=20, source_revision="r1"),
        }
        uow = Uow(values)
        service = KnowledgeCoreIntakeService(receipt_ttl_seconds=60, uow_factory=lambda: uow)

        with self.assertRaisesRegex(IntakeConflictError, "SOURCE_REVISION_CONFLICT"):
            await service.accept_published(self._command())
        uow.commit.assert_not_awaited()

    async def test_reserves_receipt_before_staging_files(self):
        values = {"collection": self._collection()}
        uow = Uow(values)
        original_add = uow.receipts.add

        async def add_receipt(receipt):
            receipt.ingestion_receipt_id = 77
            return await original_add(receipt)
        uow.receipts.add = add_receipt
        service = KnowledgeCoreIntakeService(receipt_ttl_seconds=60, uow_factory=lambda: uow)

        reservation = await service.reserve(ReserveIntakeCommand(1, "assets", "svc:portal", "k2", manifest()))

        self.assertEqual(77, reservation.receipt_id)
        self.assertEqual("RECEIVING", reservation.receipt_status)
        self.assertTrue(reservation.newly_created)
        uow.commit.assert_awaited_once()

    async def test_preparation_allocates_stable_document_ids_without_revision(self):
        receipt = SimpleNamespace(ingestion_receipt_id=77, request_fingerprint=manifest().fingerprint(), receipt_status="RECEIVING", bundle_id=None)
        values = {"collection": self._collection(), "receipt": receipt}
        uow = Uow(values)
        uow.session = SimpleNamespace(flush=AsyncMock())
        original_bundle_add, original_document_add = uow.bundles.add, uow.documents.add

        async def add_bundle(entity): entity.bundle_id = 10; return await original_bundle_add(entity)
        counter = iter((31, 32))
        async def add_document(entity): entity.document_id = next(counter); return await original_document_add(entity)
        uow.bundles.add, uow.documents.add = add_bundle, add_document
        service = KnowledgeCoreIntakeService(receipt_ttl_seconds=60, uow_factory=lambda: uow)

        prepared = await service.prepare_publish(PreparePublishCommand(1, "assets", "svc:portal", "k2", manifest(), 77))

        self.assertEqual(10, prepared.bundle_id)
        self.assertEqual({"__manifest__": 31, "doc-1": 32}, prepared.document_ids)
        self.assertEqual("STAGED", receipt.receipt_status)

    async def test_user_preparation_does_not_allocate_manifest_document(self):
        receipt = SimpleNamespace(
            ingestion_receipt_id=77,
            request_fingerprint=manifest().fingerprint(),
            receipt_status="RECEIVING",
            bundle_id=None,
        )
        values = {"collection": self._collection(), "receipt": receipt}
        uow = Uow(values)
        original_bundle_add = uow.bundles.add
        original_document_add = uow.documents.add

        async def add_bundle(entity):
            entity.bundle_id = 10
            return await original_bundle_add(entity)

        async def add_document(entity):
            entity.document_id = 31
            return await original_document_add(entity)

        uow.bundles.add = add_bundle
        uow.documents.add = add_document
        service = KnowledgeCoreIntakeService(
            receipt_ttl_seconds=60,
            uow_factory=lambda: uow,
        )

        prepared = await service.prepare_publish(
            PreparePublishCommand(
                1,
                "assets",
                "svc:portal",
                "k2",
                manifest(),
                77,
                generate_manifest=False,
            )
        )

        self.assertEqual({"doc-1": 31}, prepared.document_ids)
        self.assertEqual(
            ["doc-1"],
            [
                document.external_document_id
                for document in uow.documents.added
            ],
        )

    async def test_abort_marks_receipt_failed_and_deletes_only_created_ids(self):
        bundle_id = uuid7()
        document_id = uuid7()
        receipt = SimpleNamespace(
            receipt_status="STAGED",
            bundle_id=None,
            bundle_revision_id=None,
            staging_manifest_json={
                "bundle_id": str(bundle_id),
                "created_bundle": True,
                "created_document_ids": [str(document_id)],
            },
            failure_code=None,
            failure_message=None,
            updated_by=None,
        )
        values = {
            "collection": self._collection(),
            "receipt": receipt,
        }
        uow = Uow(values)
        service = KnowledgeCoreIntakeService(
            receipt_ttl_seconds=60,
            uow_factory=lambda: uow,
        )

        await service.abort(
            AbortIntakeCommand(
                domain_id=1,
                collection_key="assets",
                actor_id="svc:portal",
                idempotency_key="failed-key",
                failure_code="IntegrityError",
                failure_message="约束冲突",
            )
        )

        self.assertEqual("FAILED", receipt.receipt_status)
        self.assertIsNone(receipt.staging_manifest_json)
        self.assertEqual("IntegrityError", receipt.failure_code)
        self.assertEqual(
            [document_id],
            uow.documents.deleted[0][1]["document_ids"],
        )
        self.assertEqual(
            bundle_id,
            uow.bundles.deleted[0][1]["bundle_id"],
        )
        uow.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
