"""State-machine checks for the final KC intake database transaction."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from knowledge_core.application.intake import (
    AcceptKmAssetCommand, IntakeConflictError, KnowledgeCoreIntakeService, PublishedAttachment, PublishedManifest,
    ReserveIntakeCommand,
    PreparePublishCommand,
)
from knowledge_core.domain.intake import KmAssetIntakeManifest
from knowledge_core.domain.manifest import render_bundle_manifest


def manifest():
    return KmAssetIntakeManifest.model_validate({
        "bundle": {"source_id": "A-1", "source_revision": "r1", "title": "Asset", "security_level": 2},
        "documents": [{"part_name": "attachment_0", "external_document_id": "doc-1", "declared_mime_type": "text/plain", "ordinal": 0, "byte_size": 3, "content_sha256": "a" * 64}],
    })


class Repository:
    def __init__(self, values=None): self.values, self.added = values or {}, []
    async def get_by_scope_key(self, **kwargs): return self.values.get("collection")
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


class Uow:
    def __init__(self, values):
        self.collections = Repository(values); self.receipts = Repository(values); self.bundles = Repository(values)
        self.revisions = Repository(values); self.documents = Repository(values); self.versions = Repository(values)
        self.members = Repository(values); self.jobs = Repository(values); self.commit = AsyncMock()
        self.parse_views = Repository(values)
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return None


class IntakeServiceTest(unittest.IsolatedAsyncioTestCase):
    def test_parser_policy_cannot_override_collection_embedding_model(self):
        with self.assertRaisesRegex(ValueError, "retrieval embeddings"):
            KnowledgeCoreIntakeService(
                app_id=1,
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
        values = {"collection": SimpleNamespace(collection_id=1, status="ACTIVE")}
        uow = Uow(values)
        service = KnowledgeCoreIntakeService(app_id=1, receipt_ttl_seconds=60, uow_factory=lambda: uow)

        accepted = await service.accept_published(self._command())

        self.assertEqual(10, accepted.bundle_id)
        self.assertEqual("PROCESSING", uow.revisions.added[0].status)
        self.assertEqual("MANIFEST", uow.members.added[0].document_role)
        self.assertEqual("RECEIVED", uow.members.added[1].member_status)
        self.assertEqual(2, len(uow.jobs.added))
        self.assertEqual(50, uow.jobs.added[0].parse_view_id)
        self.assertEqual("ACCEPTED", uow.receipts.added[0].receipt_status)
        uow.commit.assert_awaited_once()

    async def test_rejects_changed_snapshot_for_same_source_revision(self):
        values = {
            "collection": SimpleNamespace(collection_id=1, status="ACTIVE"),
            "bundle": SimpleNamespace(bundle_id=10, collection_id=1),
            "revision": SimpleNamespace(snapshot_fingerprint="different", bundle_revision_id=20, source_revision="r1"),
        }
        uow = Uow(values)
        service = KnowledgeCoreIntakeService(app_id=1, receipt_ttl_seconds=60, uow_factory=lambda: uow)

        with self.assertRaisesRegex(IntakeConflictError, "SOURCE_REVISION_CONFLICT"):
            await service.accept_published(self._command())
        uow.commit.assert_not_awaited()

    async def test_reserves_receipt_before_staging_files(self):
        values = {"collection": SimpleNamespace(collection_id=1, status="ACTIVE")}
        uow = Uow(values)
        original_add = uow.receipts.add

        async def add_receipt(receipt):
            receipt.ingestion_receipt_id = 77
            return await original_add(receipt)
        uow.receipts.add = add_receipt
        service = KnowledgeCoreIntakeService(app_id=1, receipt_ttl_seconds=60, uow_factory=lambda: uow)

        reservation = await service.reserve(ReserveIntakeCommand(1, "assets", "svc:portal", "k2", manifest()))

        self.assertEqual(77, reservation.receipt_id)
        self.assertEqual("RECEIVING", reservation.receipt_status)
        self.assertTrue(reservation.newly_created)
        uow.commit.assert_awaited_once()

    async def test_preparation_allocates_stable_document_ids_without_revision(self):
        receipt = SimpleNamespace(ingestion_receipt_id=77, request_fingerprint=manifest().fingerprint(), receipt_status="RECEIVING", bundle_id=None)
        values = {"collection": SimpleNamespace(collection_id=1, status="ACTIVE"), "receipt": receipt}
        uow = Uow(values)
        uow.session = SimpleNamespace(flush=AsyncMock())
        original_bundle_add, original_document_add = uow.bundles.add, uow.documents.add

        async def add_bundle(entity): entity.bundle_id = 10; return await original_bundle_add(entity)
        counter = iter((31, 32))
        async def add_document(entity): entity.document_id = next(counter); return await original_document_add(entity)
        uow.bundles.add, uow.documents.add = add_bundle, add_document
        service = KnowledgeCoreIntakeService(app_id=1, receipt_ttl_seconds=60, uow_factory=lambda: uow)

        prepared = await service.prepare_publish(PreparePublishCommand(1, "assets", "svc:portal", "k2", manifest(), 77))

        self.assertEqual(10, prepared.bundle_id)
        self.assertEqual({"__manifest__": 31, "doc-1": 32}, prepared.document_ids)
        self.assertEqual("STAGED", receipt.receipt_status)


if __name__ == "__main__":
    unittest.main()
