"""Contract tests for ordinary user-file declarations."""
import unittest

from knowledge_core.api.intake_router import UserBundleDeclaration, UserFileDeclaration, _user_manifest


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


if __name__ == "__main__":
    unittest.main()
