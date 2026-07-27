"""Contract tests shared by the KC endpoint and Portal bundle adapter."""
import unittest

from knowledge_core.domain.intake import IntakeValidationError, KmAssetIntakeManifest


def valid_payload():
    return {
        "bundle": {
            "source_id": "ASSET-1", "source_revision": "2026-07-22T10:00:00Z",
            "title": "Example", "security_level": 1, "facet": {"product": "db"}, "metadata": {},
        },
        "documents": [{
            "part_name": "attachment_0", "external_document_id": "driveitem:d:i",
            "declared_mime_type": "application/pdf", "ordinal": 0,
            "byte_size": 3, "content_sha256": "a" * 64,
        }],
        "document_failures": [],
    }


class IntakeContractTest(unittest.TestCase):
    def test_accepts_portal_shape_and_is_fingerprint_stable(self):
        manifest = KmAssetIntakeManifest.model_validate(valid_payload())
        manifest.validate_declarations({"attachment_0"})
        self.assertEqual(manifest.fingerprint(), KmAssetIntakeManifest.model_validate(valid_payload()).fingerprint())

    def test_rejects_missing_or_unexpected_file_part(self):
        manifest = KmAssetIntakeManifest.model_validate(valid_payload())
        with self.assertRaises(IntakeValidationError):
            manifest.validate_declarations({"wrong_part"})

    def test_rejects_overlap_between_file_and_explicit_failure(self):
        payload = valid_payload()
        payload["document_failures"] = [{
            "external_document_id": "driveitem:d:i", "ordinal": 0, "failure_code": "SOURCE_DOWNLOAD_FAILED",
        }]
        manifest = KmAssetIntakeManifest.model_validate(payload)
        with self.assertRaises(IntakeValidationError):
            manifest.validate_declarations({"attachment_0"})

    def test_rejects_client_manifest(self):
        payload = valid_payload()
        payload["documents"][0]["external_document_id"] = "__manifest__"
        manifest = KmAssetIntakeManifest.model_validate(payload)
        with self.assertRaises(IntakeValidationError):
            manifest.validate_declarations({"attachment_0"})


if __name__ == "__main__":
    unittest.main()
