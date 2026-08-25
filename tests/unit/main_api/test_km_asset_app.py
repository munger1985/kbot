"""KM Asset App 公开 BFF 契约单元测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError

from main_api.api.km_asset_app import (
    AgentCreatePayload,
    AssetReferencePreview,
    ConversationTurnPayload,
    _manifest_asset_fields,
    _validated_collection_models,
    _asset_attachment,
    _km_turn_receipt,
    _runtime_auth_context,
)
from platform_core.contracts import (
    AuthContext,
    IdentityEntryKind,
    PrincipalKind,
)


class KmAssetAppContractTest(unittest.TestCase):
    def test_manifest_source_metadata_has_priority_over_legacy_headers(
        self,
    ) -> None:
        fields = _manifest_asset_fields(
            "# Legacy Title\n\n"
            "Source ID: LEGACY/100\n\n"
            "## Source metadata\n"
            '{"external_asset_id":"METADATA/100",'
            '"asset_id":"LOWER-PRIORITY/100",'
            '"asset_title":"Metadata Title",'
            '"title":"Lower-priority Title"}\n'
        )

        self.assertEqual("METADATA/100", fields["asset_id"])
        self.assertEqual("Metadata Title", fields["asset_title"])

    def test_manifest_legacy_headers_are_missing_metadata_fallbacks(
        self,
    ) -> None:
        fields = _manifest_asset_fields(
            "# Legacy Title\n\n"
            "Source ID: LEGACY/100\n\n"
            "## Source metadata\n"
            '{"author_mail":"author@example.com"}\n'
        )

        self.assertEqual("LEGACY/100", fields["asset_id"])
        self.assertEqual("Legacy Title", fields["asset_title"])
        self.assertEqual("author@example.com", fields["author_mail"])

    def test_app_api_client_uses_bound_user_portal_runtime_context(self) -> None:
        agent_id = UUID("01900000-0000-7000-8000-000000000031")
        source = AuthContext(
            principal_kind=PrincipalKind.APP_API_CLIENT,
            client_id="kmadmin-api-client",
            request_id="request-1",
            trace_id="trace-1",
            api_key_id="key-1",
            entry_kind=IdentityEntryKind.BUSINESS,
            app_id="km_asset",
            domain_id="43",
            asserted_user_id="kmadmin",
            roles=("km_user",),
            scopes=("km:conversation:write",),
            authorized_agent_ids=(agent_id,),
        )
        request = SimpleNamespace(
            state=SimpleNamespace(auth_context=source)
        )

        projected = _runtime_auth_context(request)

        self.assertEqual(PrincipalKind.PORTAL, projected.principal_kind)
        self.assertEqual("kmadmin", projected.asserted_user_id)
        self.assertEqual("43", projected.domain_id)
        self.assertEqual("km_asset", projected.app_id)
        self.assertEqual((agent_id,), projected.authorized_agent_ids)
        self.assertEqual("kmadmin-api-client", projected.delegated_by)
        self.assertEqual(PrincipalKind.APP_API_CLIENT, source.principal_kind)

    def test_portal_runtime_context_is_not_rewritten(self) -> None:
        source = AuthContext(
            principal_kind=PrincipalKind.PORTAL,
            client_id="user-session",
            request_id="request-2",
            trace_id="trace-2",
            api_key_id="portal-session-1",
            entry_kind=IdentityEntryKind.BUSINESS,
            app_id="km_asset",
            domain_id="43",
            asserted_user_id="kmadmin",
        )
        request = SimpleNamespace(
            state=SimpleNamespace(auth_context=source)
        )

        self.assertIs(source, _runtime_auth_context(request))

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


class KmKnowledgeCoreContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_collection_models_are_validated_against_active_catalog(self):
        embedding_id = UUID("01900000-0000-7000-8000-000000000032")
        visual_id = UUID("01900000-0000-7000-8000-000000000033")
        vlm_id = UUID("01900000-0000-7000-8000-000000000034")
        catalog = [
            {"model_id": str(embedding_id), "category": 2},
            {"model_id": str(visual_id), "category": 3},
            {"model_id": str(vlm_id), "category": 5},
        ]

        with patch(
            "main_api.api.km_asset_app.load_model_catalog",
            new=AsyncMock(return_value=catalog),
        ):
            models = await _validated_collection_models(
                SimpleNamespace(),
                parser_vlm=vlm_id,
                embedding=embedding_id,
                visual_embedding=visual_id,
            )

        self.assertEqual(str(vlm_id), models["parser_vlm"])
        self.assertEqual(str(embedding_id), models["embedding"])
        self.assertEqual(str(visual_id), models["visual_embedding"])

    async def test_collection_model_category_mismatch_is_rejected(self):
        embedding_id = UUID("01900000-0000-7000-8000-000000000042")
        vlm_id = UUID("01900000-0000-7000-8000-000000000043")
        catalog = [
            {"model_id": str(embedding_id), "category": 2},
            {"model_id": str(vlm_id), "category": 2},
        ]

        with patch(
            "main_api.api.km_asset_app.load_model_catalog",
            new=AsyncMock(return_value=catalog),
        ), self.assertRaises(HTTPException) as raised:
            await _validated_collection_models(
                SimpleNamespace(),
                parser_vlm=vlm_id,
                embedding=embedding_id,
                visual_embedding=None,
            )

        self.assertEqual(422, raised.exception.status_code)
        self.assertEqual(
            "KM_COLLECTION_MODEL_CATEGORY_INVALID",
            raised.exception.detail["code"],
        )


if __name__ == "__main__":
    unittest.main()
