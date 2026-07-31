"""Contract tests for ordinary user-file declarations."""
import unittest
from unittest.mock import MagicMock, patch
from platform_core.identity import uuid7

from knowledge_core.api.intake_router import (
    UserBundleDeclaration,
    UserFileDeclaration,
    _record_user_file_failure,
    _user_manifest,
)


class UserIntakeContractTest(unittest.TestCase):
    def test_manifest_does_not_leak_transport_client_file_id(self):
        item = UserFileDeclaration(
            part_name="upload-1", client_file_id="file-1", display_name="report.xlsx",
            declared_mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            byte_size=10, content_sha256="a" * 64,
        )
        manifest = _user_manifest(
            UserBundleDeclaration(client_bundle_id="bundle-1", title="Report"),
            [item], source_revision="r1",
        )
        self.assertEqual("file-1", manifest.documents[0].external_document_id)
        self.assertEqual("report.xlsx", manifest.documents[0].declared_name)
        self.assertFalse(hasattr(manifest.documents[0], "client_file_id"))

    def test_failure_returns_error_id_and_records_searchable_context(self):
        bound_logger = MagicMock()
        with patch(
            "knowledge_core.api.intake_router.logger.bind",
            return_value=bound_logger,
        ) as bind:
            collection_id = uuid7()
            result = _record_user_file_failure(
                exc=RuntimeError("数据库约束冲突"),
                request_id="request-1",
                domain_id=7,
                collection_id=collection_id,
                client_file_id="file-1",
                file_name="guide.pdf",
            )

        self.assertEqual(
            result["error_code"], "USER_FILE_INGESTION_FAILED"
        )
        self.assertTrue(result["error_id"])
        self.assertEqual(result["error_type"], "RuntimeError")
        bind.assert_called_once_with(
            event_type="USER_FILE_INGESTION_FAILED",
            error_id=result["error_id"],
            request_id="request-1",
            domain_id=7,
            collection_id=str(collection_id),
            client_file_id="file-1",
            file_name="guide.pdf",
        )
        bound_logger.error.assert_called_once()
        log_arguments = bound_logger.error.call_args.args
        self.assertIn("RuntimeError", log_arguments)
        self.assertIn("数据库约束冲突", log_arguments)


if __name__ == "__main__":
    unittest.main()
