"""The multipart orchestrator orders staging, allocation, publishing and final commit."""
import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path

from knowledge_core.application.intake import IntakeAcceptance, IntakeReservation, PublishPreparation
from knowledge_core.application.multipart import KnowledgeCoreMultipartOrchestrator, MultipartIntakeCommand
from knowledge_core.domain.intake import KmAssetIntakeManifest
from knowledge_core.ports.object_store import StoredObject
from platform_core.identity import uuid7


COLLECTION_ID = uuid7()
BUNDLE_ID = uuid7()
MANIFEST_DOCUMENT_ID = uuid7()
ATTACHMENT_DOCUMENT_ID = uuid7()
REVISION_ID = uuid7()
RECEIPT_ID = uuid7()


class FakeStore:
    def __init__(self): self.calls = []
    async def stage_file(self, **kwargs):
        self.calls.append(("stage", kwargs["part_name"]))
        return StoredObject(f"stage://{kwargs['part_name']}", kwargs["expected_size"], kwargs["expected_sha256"], kwargs["detected_mime_type"])
    async def publish_staged(self, **kwargs):
        self.calls.append(("publish", kwargs["document_id"]))
        item = kwargs["staged"]
        return StoredObject(f"published://{kwargs['document_id']}", item.byte_size, item.content_sha256, item.detected_mime_type)
    async def delete(self, uri): self.calls.append(("delete", uri))


class FakeIntake:
    def __init__(self): self.calls = []
    async def reserve(self, command):
        self.calls.append("reserve")
        return IntakeReservation(RECEIPT_ID, "RECEIVING", True)
    async def prepare_publish(self, command):
        self.calls.append("prepare")
        return PublishPreparation(
            COLLECTION_ID,
            BUNDLE_ID,
            {
                "__manifest__": MANIFEST_DOCUMENT_ID,
                "doc-1": ATTACHMENT_DOCUMENT_ID,
            },
        )
    async def accept_published(self, command):
        self.calls.append("accept")
        return IntakeAcceptance(BUNDLE_ID, REVISION_ID, "r1")


class MultipartOrchestratorTest(unittest.TestCase):
    def test_runs_all_phases_in_order(self):
        payload = b"abc"
        data = {
            "bundle": {"source_id": "A-1", "source_revision": "r1", "title": "Asset", "security_level": 1},
            "documents": [{"part_name": "attachment_0", "external_document_id": "doc-1", "declared_mime_type": "text/plain", "ordinal": 0, "byte_size": len(payload), "content_sha256": hashlib.sha256(payload).hexdigest()}],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "a.txt"
            source.write_bytes(payload)
            intake, store = FakeIntake(), FakeStore()
            result = asyncio.run(KnowledgeCoreMultipartOrchestrator(intake_service=intake, object_store=store).accept(
                MultipartIntakeCommand(1, "assets", "svc:portal", "key", KmAssetIntakeManifest.model_validate(data), {"attachment_0": source})
            ))
        self.assertEqual(REVISION_ID, result.bundle_revision_id)
        self.assertEqual(["reserve", "prepare", "accept"], intake.calls)
        self.assertEqual(["stage", "stage", "publish", "publish"], [call[0] for call in store.calls])


if __name__ == "__main__":
    unittest.main()
