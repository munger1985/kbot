"""KM Asset App 公开 BFF 契约单元测试。"""

import unittest
from uuid import UUID

from pydantic import ValidationError

from main_api.api.km_asset_app import (
    AgentCreatePayload,
    AssetReferencePreview,
    ConversationTurnPayload,
    _asset_attachment,
    _asset_reference_fields,
    _km_turn_receipt,
)


class KmAssetAppContractTest(unittest.TestCase):
    def test_public_agent_create_rejects_caller_supplied_capabilities(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            AgentCreatePayload(
                source_id=UUID("01900000-0000-7000-8000-000000000001"),
                display_name="KM Agent",
                enabled_capabilities=["conversation", "document", "data_query"],
            )

        self.assertEqual("extra_forbidden", raised.exception.errors()[0]["type"])
        self.assertEqual(("enabled_capabilities",), raised.exception.errors()[0]["loc"])

    def test_turn_rejects_caller_supplied_security_level(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            ConversationTurnPayload(
                input="查询 Asset",
                expected_conversation_version=1,
                security_level=0,
            )

        self.assertEqual("extra_forbidden", raised.exception.errors()[0]["type"])
        self.assertEqual(("security_level",), raised.exception.errors()[0]["loc"])

    def test_turn_receipt_uses_km_owned_event_stream(self) -> None:
        run_id = UUID("019ff999-fb22-7d92-8e87-49a20b1d18fa")
        upstream = {
            "run_id": str(run_id),
            "status": "RUNNING",
            "event_cursor": 1,
            "events_url": (
                "/api/v1/apps/knowledge-retrieval/runs/"
                f"{run_id}/events"
            ),
        }

        receipt = _km_turn_receipt(upstream)

        self.assertEqual(
            f"/api/v1/apps/km-asset/runs/{run_id}/events",
            receipt["events_url"],
        )
        self.assertIn("knowledge-retrieval", upstream["events_url"])

    def test_turn_receipt_without_run_keeps_events_url_empty(self) -> None:
        receipt = _km_turn_receipt({"run_id": None, "events_url": None})

        self.assertIsNone(receipt["events_url"])

    def test_asset_reference_fields_match_bundle_without_internal_ids(self) -> None:
        bundle_id = UUID("01900000-0000-7000-8000-000000000011")
        fields = _asset_reference_fields(
            {
                "query_results": [{
                    "rows": [{
                        "asset_id": "ASSET-1",
                        "bundle_id": str(bundle_id),
                        "bundle_revision_id": (
                            "01900000-0000-7000-8000-000000000012"
                        ),
                        "title": "OAC Fraud Asset",
                        "product": "OAC",
                        "solution": "Financial Fraud",
                    }],
                }],
            },
            bundle_id=bundle_id,
        )

        self.assertEqual("OAC Fraud Asset", fields["title"])
        self.assertEqual("OAC", fields["product"])
        self.assertNotIn("asset_id", fields)
        self.assertNotIn("bundle_id", fields)

    def test_asset_attachment_excludes_manifest_and_keeps_evidence_locator(self) -> None:
        run_id = UUID("01900000-0000-7000-8000-000000000021")
        document_version_id = UUID(
            "01900000-0000-7000-8000-000000000022"
        )
        manifest = _asset_attachment(
            run_id=run_id,
            citation_label="C1",
            item={
                "document_role": "MANIFEST",
                "document_version_id": str(document_version_id),
                "preview_available": True,
            },
            evidence_document_version_id=document_version_id,
            locator=(3, 4, None),
        )
        attachment = _asset_attachment(
            run_id=run_id,
            citation_label="C1",
            item={
                "document_role": "ATTACHMENT",
                "document_version_id": str(document_version_id),
                "declared_name": "fraud.pdf",
                "detected_mime_type": "application/pdf",
                "preview_available": True,
            },
            evidence_document_version_id=document_version_id,
            locator=(3, 4, None),
        )

        self.assertIsNone(manifest)
        self.assertIsNotNone(attachment)
        self.assertTrue(attachment.evidence_source)
        self.assertEqual(3, attachment.page_no)
        self.assertIn(f"/files/{document_version_id}/content", attachment.content_url)

    def test_asset_reference_can_exist_without_attachments(self) -> None:
        preview = AssetReferencePreview(
            citation_label="C1",
            title="Metadata-only Asset",
            revision_no=1,
            status="READY",
            approval_status="APPROVED",
            is_current_revision=True,
            asset_content_available=True,
        )

        self.assertEqual("ASSET", preview.reference_type)
        self.assertEqual((), preview.attachments)


if __name__ == "__main__":
    unittest.main()
