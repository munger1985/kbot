"""Slack 4.0 接入契约的单元测试。"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from km_asset_app.application.slack_assets import (
    _mapping_asset_fields,
    assemble_slack_asset_cards as _assemble_slack_asset_cards,
    extract_answer_asset_cards,
    parse_manifest_asset_fields,
)
from km_asset_app.application.slack_dispatch import SlackDispatchService
from km_asset_app.application.slack_intake import (
    SlackIntakeService,
    SlackWebhookError,
    parse_message_event,
    verify_slack_signature,
)
from km_asset_app.application.slack_rendering import (
    build_callback_payload,
    processing_failure_message,
    render_slack_replies,
    render_slack_reply,
    slack_visible_payload,
    waiting_message,
)
from km_asset_app.config import (
    SlackExternalCallbackConfig,
    SlackIntegrationConfig,
    SlackReplyConfig,
    SlackWorkspaceConfig,
)


class SlackSignatureTest(unittest.TestCase):
    def test_verifies_exact_raw_body(self):
        body = b'{"type":"event_callback"}'
        timestamp = "1720000000"
        secret = "test-secret"
        base = b"v0:" + timestamp.encode() + b":" + body
        signature = "v0=" + hmac.new(
            secret.encode(), base, hashlib.sha256
        ).hexdigest()
        self.assertTrue(
            verify_slack_signature(
                signing_secret=secret,
                timestamp=timestamp,
                signature=signature,
                raw_body=body,
                now=1720000000,
            )
        )
        self.assertFalse(
            verify_slack_signature(
                signing_secret=secret,
                timestamp=timestamp,
                signature=signature,
                raw_body=body + b" ",
                now=1720000000,
            )
        )

    def test_rejects_expired_request(self):
        self.assertFalse(
            verify_slack_signature(
                signing_secret="secret",
                timestamp="1",
                signature="v0=invalid",
                raw_body=b"{}",
                now=1000,
            )
        )


class SlackWaitingMessageTest(unittest.TestCase):
    def test_waiting_and_failure_messages_match_current_message_language(self):
        samples = {
            "显示所有与 ChatBI 相关的资产": (
                "KM 助手正在搜集材料并分析您的问题，请稍候。",
                "KBot 无法处理此请求，请稍后重试。",
            ),
            "ChatBIに関連するすべてのアセットを表示する": (
                "KM アシスタントが資料を収集して質問を分析しています。しばらくお待ちください。",
                "KBot はこのリクエストを処理できませんでした。後でもう一度お試しください。",
            ),
            "ChatBI 관련 자산을 모두 표시해 주세요": (
                "KM 어시스턴트가 자료를 수집하고 질문을 분석하고 있습니다. 잠시만 기다려 주세요.",
                "KBot이 이 요청을 처리하지 못했습니다. 나중에 다시 시도해 주세요.",
            ),
            "Show all assets related to ChatBI": (
                "KM Assistant is gathering materials and analyzing your question, please wait.",
                "KBot was unable to process this request. Please try again later.",
            ),
        }
        for question, (waiting, failure) in samples.items():
            with self.subTest(question=question):
                self.assertEqual(waiting, waiting_message(question))
                self.assertEqual(failure, processing_failure_message(question))


class SlackEventParsingTest(unittest.TestCase):
    def test_accepts_message_and_mention(self):
        for event_type in ("message", "app_mention"):
            payload = {
                "type": "event_callback",
                "team_id": "T1",
                "event_id": f"E-{event_type}",
                "event": {
                    "type": event_type,
                    "user": "U1",
                    "channel": "C1",
                    "text": "hello",
                    "event_ts": "1.001",
                },
            }
            parsed = parse_message_event(payload)
            self.assertEqual(event_type, parsed["event_type"])
            self.assertEqual("1.001", parsed["root_thread_ts"])

    def test_mailto_wrapper_is_restored_to_user_visible_email(self):
        payload = {
            "type": "event_callback",
            "team_id": "T1",
            "event_id": "E-email",
            "event": {
                "type": "message",
                "user": "U1",
                "channel": "C1",
                "text": (
                    "any assets of "
                    "<mailto:madhumitha.k@oracle.com|"
                    "madhumitha.k@oracle.com>；"
                ),
                "event_ts": "1.001",
            },
        }

        parsed = parse_message_event(payload)

        self.assertEqual(
            "any assets of madhumitha.k@oracle.com；",
            parsed["message_text"],
        )
        self.assertIn("<mailto:", payload["event"]["text"])

    def test_message_and_mention_share_message_identity(self):
        message = {
            "type": "event_callback",
            "team_id": "T1",
            "event_id": "E-message",
            "event": {
                "type": "message",
                "user": "U1",
                "channel": "C1",
                "text": "<@BOT> hello",
                "ts": "1723880000.123456",
                "event_ts": "1723880000.223456",
                "client_msg_id": "client-message-id",
            },
        }
        mention = json.loads(json.dumps(message))
        mention["event_id"] = "E-mention"
        mention["event"]["type"] = "app_mention"
        mention["event"]["event_ts"] = "1723880000.323456"
        mention["event"].pop("client_msg_id")

        parsed_message = parse_message_event(message)
        parsed_mention = parse_message_event(mention)

        self.assertEqual(
            parsed_message["message_identity"],
            parsed_mention["message_identity"],
        )
        self.assertEqual(
            "1723880000.123456",
            parsed_message["message_identity"],
        )

    def test_rejects_bot_and_edited_messages(self):
        base = {
            "type": "event_callback",
            "team_id": "T1",
            "event_id": "E1",
            "event": {
                "type": "message",
                "user": "U1",
                "channel": "C1",
                "text": "hello",
                "event_ts": "1.001",
            },
        }
        for subtype in ("bot_message", "message_changed", "message_deleted"):
            payload = json.loads(json.dumps(base))
            payload["event"]["subtype"] = subtype
            self.assertIsNone(parse_message_event(payload))

    def test_rejects_non_message_event(self):
        payload = {
            "type": "event_callback",
            "team_id": "T1",
            "event_id": "E1",
            "event": {
                "type": "reaction_added",
                "user": "U1",
                "channel": "C1",
                "text": "hello",
                "event_ts": "1.001",
            },
        }
        self.assertIsNone(parse_message_event(payload))


class SlackUrlVerificationTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _signed_request(secret: str, payload: dict):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        timestamp = str(int(time.time()))
        signature = "v0=" + hmac.new(
            secret.encode("utf-8"),
            b"v0:" + timestamp.encode("utf-8") + b":" + body,
            hashlib.sha256,
        ).hexdigest()
        return body, {
            "x-slack-request-timestamp": timestamp,
            "x-slack-signature": signature,
        }

    @staticmethod
    def _config():
        return SlackIntegrationConfig(
            enabled=True,
            workspaces=[
                {
                    "workspace_id": "T1",
                    "domain_id": 1001,
                    "agent_id": "019fcbe0-e46c-7d33-907b-9d1621a2998f",
                    "signing_secret_env": "KBOT_SLACK_TEST_SIGNING_SECRET",
                }
            ],
        )

    async def test_accepts_signed_challenge_without_team_id(self):
        body, headers = self._signed_request(
            "test-secret",
            {
                "type": "url_verification",
                "token": "deprecated-token",
                "challenge": "challenge-value",
            },
        )
        with patch.dict(
            os.environ,
            {"KBOT_SLACK_TEST_SIGNING_SECRET": "test-secret"},
            clear=True,
        ):
            result = await SlackIntakeService(
                uow_factory=None,
                slack_config=self._config(),
            ).receive(raw_body=body, headers=headers)

        self.assertTrue(result.accepted)
        self.assertEqual("challenge-value", result.challenge)

    async def test_rejects_challenge_with_invalid_signature(self):
        body, headers = self._signed_request(
            "wrong-secret",
            {
                "type": "url_verification",
                "challenge": "challenge-value",
            },
        )
        with patch.dict(
            os.environ,
            {"KBOT_SLACK_TEST_SIGNING_SECRET": "test-secret"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                SlackWebhookError,
                "签名无效",
            ):
                await SlackIntakeService(
                    uow_factory=None,
                    slack_config=self._config(),
                ).receive(raw_body=body, headers=headers)


class SlackAssetExtractionTest(unittest.TestCase):
    def test_maps_main_api_preview_asset_fields(self):
        fields = _mapping_asset_fields(
            {
                "asset_id": "593C6847F5EE8D1DE0630B427364FE2F",
                "asset_title": "Deep Data Security",
                "author_email": "HYSUN.HE@ORACLE.COM",
                "briefing": "An end-to-end demo.",
                # 故意把更新时间放在发布时间前，验证优先级不受 JSON 键顺序影响。
                "last_update_time": "2026-08-18T10:46:00Z",
                "publish_time": "2026-08-17T10:46:33Z",
                "product": "Database Security,Database -> SelectAI",
                "solution": "ChatBI,AI / Machine Learning,RAG",
            }
        )

        self.assertEqual(
            "593C6847F5EE8D1DE0630B427364FE2F", fields["asset_id"]
        )
        self.assertEqual("Deep Data Security", fields["asset_title"])
        self.assertEqual("HYSUN.HE@ORACLE.COM", fields["author_mail"])
        self.assertEqual("An end-to-end demo.", fields["solution_briefing"])
        self.assertEqual("2026-08-17T10:46:33Z", fields["create_time"])
        self.assertNotIn("product", fields)
        self.assertNotIn("solution", fields)

    def test_preview_time_falls_back_to_last_update_time(self):
        fields = _mapping_asset_fields(
            {
                "publish_time": "",
                "last_update_time": "2026-08-17T10:46:00Z",
            }
        )

        self.assertEqual("2026-08-17T10:46:00Z", fields["create_time"])

    def test_extracts_4_0_answer_fields_without_citations(self):
        cards = extract_answer_asset_cards(
            "**资产名称**：Conversational Banking [C1]\n"
            "**解决方案简介**：第一行 [C1]\n"
            "第二行说明 [C1]\n\n"
            "**作者邮箱**：AUTHOR@EXAMPLE.COM [C1]"
        )

        self.assertEqual(1, len(cards))
        self.assertEqual("Conversational Banking", cards[0]["asset_title"])
        self.assertEqual(
            "第一行 第二行说明",
            cards[0]["solution_briefing"],
        )
        self.assertEqual("AUTHOR@EXAMPLE.COM", cards[0]["author_mail"])
        self.assertEqual("C1", cards[0]["citation_label"])

    def test_parses_only_manifest_source_metadata_fields(self):
        fields = parse_manifest_asset_fields(
            "# Manifest Title\n\n"
            "Source ID: ASSET/100\n"
            "Source revision: R1\n\n"
            "## Facets\n{}\n\n"
            "## Source metadata\n"
            '{"external_asset_id":"METADATA/100",'
            '"asset_id":"LOWER-PRIORITY/100",'
            '"asset_title":"Metadata Title",'
            '"title":"Lower-priority Title",'
            '"solution_briefing":"Metadata Briefing",'
            '"author_mail":"author@example.com",'
            '"create_time":"2026-08-17",'
            '"product":"Database -> SelectAI",'
            '"solution":"ChatBI,RAG",'
            '"secret":"must-not-leak"}\n'
        )

        self.assertEqual("METADATA/100", fields["asset_id"])
        self.assertEqual("Metadata Title", fields["asset_title"])
        self.assertEqual("Metadata Briefing", fields["solution_briefing"])
        self.assertEqual("author@example.com", fields["author_mail"])
        self.assertEqual("2026-08-17", fields["create_time"])
        self.assertNotIn("product", fields)
        self.assertNotIn("solution", fields)
        self.assertNotIn("secret", fields)


class _ManifestClient:
    def __init__(self, manifests: dict[str, bytes]):
        self._manifests = manifests
        self.previewed_revisions: list[str] = []

    async def get_bundle_revision_preview(self, **kwargs):
        revision_id = str(kwargs["bundle_revision_id"])
        self.previewed_revisions.append(revision_id)
        content = self._manifests[revision_id]
        return {
            "files": [
                {
                    "document_role": "MANIFEST",
                    "declared_name": "manifest.md",
                    "preview_available": True,
                    "document_version_id": revision_id,
                    "byte_size": len(content),
                    "declared_mime_type": "text/markdown",
                }
            ]
        }

    async def stream_source_file(self, **kwargs):
        content = self._manifests[str(kwargs["bundle_revision_id"])]

        async def stream():
            yield content

        return SimpleNamespace(status_code=200, body=stream())


class _PublicReferencePreviewAdapter:
    """把旧测试夹具投影为 Slack 实际使用的 Main API 预览契约。"""

    def __init__(self, artifact: dict, knowledge_core_client):
        payload = artifact.get("payload") or {}
        self._references = {
            str(item.get("citation_label") or "").upper(): item
            for item in payload.get("references", [])
            if isinstance(item, dict)
        }
        self._knowledge_core_client = knowledge_core_client

    async def get_reference_preview(self, *, run_id, citation_label):
        reference = self._references[citation_label.upper()]
        client = self._knowledge_core_client
        preview = await client.get_bundle_revision_preview(
            bundle_revision_id=reference["bundle_revision_id"]
        )
        manifest = next(
            item
            for item in preview.get("files", [])
            if str(item.get("document_role") or "").upper() == "MANIFEST"
        )
        response = await client.stream_source_file(
            bundle_revision_id=reference["bundle_revision_id"],
            document_version_id=manifest["document_version_id"],
        )
        body = bytearray()
        async for chunk in response.body:
            body.extend(chunk)
        fields = parse_manifest_asset_fields(body.decode("utf-8-sig"))
        return {
            "title": fields.get("asset_title", ""),
            "asset_fields": fields,
        }


async def assemble_slack_asset_cards(
    *,
    artifact,
    limit,
    main_api_client=None,
    run_id=None,
    knowledge_core_client=None,
    **_legacy,
):
    """兼容旧夹具，同时始终验证生产代码的公开 Main API 契约。"""
    client = main_api_client or _PublicReferencePreviewAdapter(
        artifact, knowledge_core_client
    )
    return await _assemble_slack_asset_cards(
        artifact=artifact,
        main_api_client=client,
        run_id=run_id or UUID("01900000-0000-7000-8000-000000000001"),
        limit=limit,
    )


class SlackAssetManifestFallbackTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _manifest(asset_id: str, title: str) -> bytes:
        return (
            f"# {title}\n\n"
            f"Source ID: {asset_id}\n\n"
            "## Source metadata\n"
            + json.dumps(
                {
                    "asset_title": title,
                    "solution_briefing": f"{title} Briefing",
                    "author_mail": "author@example.com",
                    "create_time": "2026-08-17",
                }
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _reference(label: str, index: int) -> dict[str, str]:
        suffix = f"{index:012d}"
        return {
            "reference_type": "DOCUMENT",
            "citation_label": label,
            "collection_id": f"01900000-0000-7000-8000-{suffix}",
            "bundle_id": f"01910000-0000-7000-8000-{suffix}",
            "bundle_revision_id": f"01920000-0000-7000-8000-{suffix}",
            "document_version_id": f"01930000-0000-7000-8000-{suffix}",
        }

    async def test_main_api_preview_populates_complete_slack_template_fields(self):
        reference = self._reference("C1", 104)
        title = (
            "Deep Data Security with IAM in Agentic Application Demo "
            "(OCI Database Tools MCP Server / Agent Skill)"
        )
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": f"1. **{title}** [C1]",
                "status": "READY",
                "used_citation_labels": ["C1"],
                "references": [reference],
            },
        }
        preview = {
            "reference_type": "ASSET",
            "citation_label": "C1",
            "title": title,
            "asset_fields": {
                "asset_id": "593C6847F5EE8D1DE0630B427364FE2F",
                "asset_title": title,
                "author_email": "HYSUN.HE@ORACLE.COM",
                "briefing": "An end-to-end demo.",
                "publish_time": "2026-08-17T10:46:33Z",
                "last_update_time": "2026-08-18T10:46:00Z",
                "product": "Database Security,Database -> SelectAI",
                "solution": "ChatBI,AI / Machine Learning,RAG",
            },
        }
        main_api_client = SimpleNamespace(
            get_reference_preview=AsyncMock(return_value=preview)
        )

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            main_api_client=main_api_client,
            run_id=UUID("01a036ad-bc1c-76da-862f-33027466d09a"),
            limit=10,
        )

        self.assertEqual(
            [
                {
                    "asset_id": "593C6847F5EE8D1DE0630B427364FE2F",
                    "asset_title": title,
                    "solution_briefing": "An end-to-end demo.",
                    "author_mail": "HYSUN.HE@ORACLE.COM",
                    "create_time": "2026-08-17T10:46:33Z",
                }
            ],
            cards,
        )
        main_api_client.get_reference_preview.assert_awaited_once_with(
            run_id=UUID("01a036ad-bc1c-76da-862f-33027466d09a"),
            citation_label="C1",
        )

    async def test_answer_without_asset_does_not_create_reference_templates(self):
        reference = self._reference("C1", 99)
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": "当前资料不足，无法确认相关资产。[C1]",
                "status": "INSUFFICIENT_EVIDENCE",
                "used_citation_labels": ["C1"],
                "references": [reference],
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=_ManifestClient(
                {
                    reference["bundle_revision_id"]: self._manifest(
                        "ASSET/BACKGROUND", "仅作为参考的 Asset"
                    )
                }
            ),
            domain_id=1001,
            auth_context=None,
            limit=10,
        )

        self.assertEqual([], cards)

    async def test_natural_language_asset_title_builds_only_named_template(self):
        references = [
            self._reference("C1", 101),
            self._reference("C2", 102),
        ]
        title = "Oracle Analytics Cloud March 2026 New Features"
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    f'The asset titled "{title}" is listed as a supporting '
                    "asset. The available excerpt does not include specific "
                    "feature details. [C1]"
                ),
                "status": "READY",
                "used_citation_labels": ["C1", "C2"],
                "references": references,
            },
        }
        client = _ManifestClient(
            {
                references[0]["bundle_revision_id"]: self._manifest(
                    "ASSET/OAC", title
                ),
                references[1]["bundle_revision_id"]: self._manifest(
                    "ASSET/BACKGROUND", "Background Evidence Asset"
                ),
            }
        )

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=client,
            domain_id=1001,
            auth_context=None,
            limit=10,
        )

        self.assertEqual(1, len(cards))
        self.assertEqual("ASSET/OAC", cards[0]["asset_id"])
        self.assertEqual(title, cards[0]["asset_title"])

    async def test_template_allows_missing_solution_briefing(self):
        reference = self._reference("C1", 103)
        title = "A K3s HA environment operations guides"
        manifest = (
            f"# {title}\n\n"
            "Source ID: ASSET/K3S\n\n"
            "## Source metadata\n"
            + json.dumps(
                {
                    "asset_title": title,
                    "author_mail": "author@example.com",
                    "create_time": "2026-08-17T08:30:00Z",
                }
            )
            + "\n"
        ).encode("utf-8")
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": f"1. **{title}** [C1]",
                "status": "READY",
                "used_citation_labels": ["C1"],
                "references": [reference],
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=_ManifestClient(
                {reference["bundle_revision_id"]: manifest}
            ),
            domain_id=1001,
            auth_context=None,
            limit=10,
        )

        self.assertEqual(1, len(cards))
        self.assertEqual("ASSET/K3S", cards[0]["asset_id"])
        self.assertEqual(title, cards[0]["asset_title"])
        self.assertNotIn("solution_briefing", cards[0])
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact=artifact,
            reply_config=SlackReplyConfig(
                km_portal_base_url="https://km.example.com/assets/",
                max_references=10,
            ),
            asset_cards=cards,
        )
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertIn(f"*Asset Title:* {title}", rendered)
        self.assertNotIn("*Solution Briefing:*", rendered)
        self.assertIn("https://km.example.com/assets/ASSET%2FK3S", rendered)

    async def test_query_result_without_document_does_not_build_templates(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "**Assets created on 2026-08-18 (2 assets)**\n"
                    "- **Asset B**: B details.\n"
                    "- **Asset A**: A details. [Q1]\n"
                    "**Note**: dates use the original publish date. [Q1]"
                ),
                "status": "READY",
                "used_citation_labels": ["Q1"],
                "references": [
                    {
                        "reference_type": "QUERY_RESULT",
                        "citation_label": "Q1",
                        "query_result_id": (
                            "01900000-0000-7000-8000-000000000501"
                        ),
                        "provider": "SEMANTIC",
                        "row_count": 2,
                    }
                ],
                "query_results": [
                    {
                        "schema": "QUERY_RESULT.v1",
                        "query_result_id": (
                            "01900000-0000-7000-8000-000000000501"
                        ),
                        "provider": "SEMANTIC",
                        "columns": [],
                        "rows": [
                            {
                                "ASSET_ID": "ASSET/A",
                                "ASSET_TITLE": "Asset A",
                                "AUTHOR_MAIL": "a@example.com",
                                "PUBLISH_DATE": "2026-08-18",
                                "NORMALIZED_METADATA_JSON": json.dumps(
                                    {"solution_briefing": "Asset A Briefing"}
                                ),
                            },
                            {
                                "ASSET_ID": "ASSET/B",
                                "ASSET_TITLE": "Asset B",
                                "NORMALIZED_METADATA_JSON": {
                                    "solution_briefing": "Asset B Briefing"
                                },
                            },
                        ],
                        "row_count": 2,
                    }
                ],
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=None,
            domain_id=1001,
            auth_context=None,
            limit=10,
        )

        self.assertEqual([], cards)

    async def test_query_result_without_document_skips_local_enrichment(self):
        row = SimpleNamespace(
            km_asset_id=UUID("01900000-0000-7000-8000-000000000601"),
            external_asset_id="ASSET/LOCAL",
            asset_title="Local Asset",
            author_mail=None,
            publish_date=None,
            normalized_metadata_json={
                "solution_briefing": "Local Asset Briefing"
            },
        )

        class Assets:
            async def list_assets_for_slack_templates(self, **kwargs):
                return [row]

        class Uow:
            def __init__(self):
                self.assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": "- **Local Asset**: details. [Q1]",
                "status": "READY",
                "used_citation_labels": ["Q1"],
                "references": [],
                "query_results": [
                    {
                        "schema": "QUERY_RESULT.v1",
                        "rows": [{"ASSET_TITLE": "Local Asset"}],
                    }
                ],
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=None,
            domain_id=1001,
            auth_context=None,
            limit=10,
            uow_factory=Uow,
        )

        self.assertEqual([], cards)

    async def test_table_answer_does_not_build_asset_templates(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "Here are the assets.\n\n"
                    "| Asset ID | Title | Author | Solution |\n"
                    "|---|---|---|---|\n"
                    "| ASSET/B | Asset B | b@example.com | RAG |\n"
                    "| ASSET/A | Asset A | a@example.com | AI | [Q1]"
                ),
                "status": "READY",
                "used_citation_labels": ["Q1"],
                "references": [],
                "query_results": [
                    {
                        "schema": "QUERY_RESULT.v1",
                        "rows": [
                            {
                                "ASSET_ID": "ASSET/B",
                                "ASSET_TITLE": "Asset B",
                                "NORMALIZED_METADATA_JSON": {
                                    "solution_briefing": "Asset B Briefing"
                                },
                            },
                            {
                                "ASSET_ID": "ASSET/A",
                                "ASSET_TITLE": "Asset A",
                                "NORMALIZED_METADATA_JSON": {
                                    "solution_briefing": "Asset A Briefing"
                                },
                            },
                        ],
                    }
                ],
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=None,
            domain_id=1001,
            auth_context=None,
            limit=10,
        )

        self.assertEqual([], cards)

    async def test_missing_required_asset_field_skips_template(self):
        reference = self._reference("C1", 98)
        manifest = (
            "# Asset A\n\n"
            "## Source metadata\n"
            '{"asset_title":"Asset A"}\n'
        ).encode("utf-8")
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": "- **Asset A**: A details. [C1]",
                "status": "READY",
                "used_citation_labels": ["C1"],
                "references": [reference],
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=_ManifestClient(
                {reference["bundle_revision_id"]: manifest}
            ),
            domain_id=1001,
            auth_context=None,
            limit=10,
        )

        self.assertEqual([], cards)
        rendered = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(max_references=10),
                    asset_cards=cards,
                )
            ),
            ensure_ascii=False,
        )
        self.assertIn("Asset A", rendered)
        self.assertIn("A details", rendered)
        self.assertNotIn("*Asset Title:*", rendered)

    async def test_attachment_metadata_is_the_template_field_source(self):
        collection_id = "01900000-0000-7000-8000-000000000101"
        bundle_id = "01900000-0000-7000-8000-000000000102"
        revision_id = "01900000-0000-7000-8000-000000000103"
        document_version_id = "01900000-0000-7000-8000-000000000104"
        manifest = (
            "# Metadata Title\n\n"
            "Source ID: ASSET/100\n\n"
            "## Facets\n{}\n\n"
            "## Source metadata\n"
            '{"asset_title":"Metadata Title",'
            '"solution_briefing":"Metadata Briefing",'
            '"author_mail":"author@example.com",'
            '"create_time":"2026-08-17"}\n'
        ).encode("utf-8")

        async def stream():
            yield manifest

        client = SimpleNamespace(
            get_bundle_revision_preview=AsyncMock(
                return_value={
                    "files": [
                        {
                            "document_role": "MANIFEST",
                            "declared_name": "manifest.md",
                            "preview_available": True,
                            "document_version_id": document_version_id,
                            "byte_size": len(manifest),
                            "declared_mime_type": "text/markdown",
                        }
                    ]
                }
            ),
            stream_source_file=AsyncMock(
                return_value=SimpleNamespace(status_code=200, body=stream())
            ),
        )
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "**资产名称**：Answer Title [C1]\n"
                    "**解决方案简介**：Answer Briefing [C1]"
                ),
                "status": "READY",
                "used_citation_labels": ["C1"],
                "references": [
                    {
                        "reference_type": "DOCUMENT",
                        "citation_label": "C1",
                        "collection_id": collection_id,
                        "bundle_id": bundle_id,
                        "bundle_revision_id": revision_id,
                        "document_version_id": document_version_id,
                    }
                ],
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=client,
            domain_id=1001,
            auth_context=None,
            limit=5,
        )

        self.assertEqual(
            [
                {
                    "asset_id": "ASSET/100",
                    "asset_title": "Metadata Title",
                    "solution_briefing": "Metadata Briefing",
                    "author_mail": "author@example.com",
                    "create_time": "2026-08-17",
                }
            ],
            cards,
        )

    async def test_manifest_cards_only_include_answer_asset_sections(self):
        references = [
            self._reference("C1", 1),
            self._reference("C2", 2),
            self._reference("C3", 3),
        ]
        manifests = {
            references[0]["bundle_revision_id"]: self._manifest(
                "ASSET/X", "只作为补充证据的资产 X"
            ),
            references[1]["bundle_revision_id"]: self._manifest(
                "ASSET/B", "正文资产 B"
            ),
            references[2]["bundle_revision_id"]: self._manifest(
                "ASSET/A", "正文资产 A"
            ),
        }
        client = _ManifestClient(manifests)
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "1. **正文资产 A** 的完整说明。[C3]\n"
                    "2. **正文资产 B** 的完整说明。[C2]\n"
                    "补充证据不作为正文资产展示。[C1]"
                ),
                "status": "READY",
                "used_citation_labels": ["C1", "C2", "C3"],
                "references": references,
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=client,
            domain_id=1001,
            auth_context=None,
            limit=3,
        )

        self.assertEqual(
            ["正文资产 A", "正文资产 B"],
            [card["asset_title"] for card in cards],
        )

    async def test_document_assets_override_zero_count_query_result(self):
        references = [
            self._reference("C1", 71),
            self._reference("C2", 72),
            self._reference("C4", 74),
        ]
        titles = [
            "KM Store: SE-HUB Travel AI Agent on Oracle Analytics Cloud",
            "SE-HUB OCI GenAI Agentic Credit Card Recommendation",
            (
                "AI Agent Factory: From Cost Reports to Cost Intelligence "
                "with OCI GenAI Services for FinOps"
            ),
        ]
        manifests = {
            reference["bundle_revision_id"]: self._manifest(
                f"OAC/{index}",
                title,
            )
            for index, (reference, title) in enumerate(
                zip(references, titles, strict=True),
                start=1,
            )
        }
        answer = (
            "Based on the provided evidence, there are 3 assets related "
            "to OAC:\n\n"
            f"• **{titles[0]}**: Travel analytics details. [C1]\n\n"
            f"• **{titles[1]}**: Credit card recommendation details. [C2]\n\n"
            f"• **{titles[2]}**: FinOps analytics details. [C4]"
        )
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": answer,
                "status": "READY",
                "used_citation_labels": ["C1", "C2", "C4", "Q1"],
                "references": [
                    *references,
                    {
                        "reference_type": "QUERY_RESULT",
                        "citation_label": "Q1",
                    },
                ],
                "query_results": [{
                    "schema": "QUERY_RESULT.v1",
                    "truncated": False,
                    "rows": [{"ASSET_COUNT": 0}],
                }],
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=_ManifestClient(manifests),
            domain_id=1001,
            auth_context=None,
            limit=10,
        )
        visible = slack_visible_payload(
            render_slack_reply(
                channel_id="C1",
                user_id="U1",
                thread_ts="1.001",
                artifact=artifact,
                reply_config=SlackReplyConfig(max_references=10),
                asset_cards=cards,
            )
        )
        rendered = json.dumps(visible, ensure_ascii=False)

        self.assertEqual(titles, [card["asset_title"] for card in cards])
        for title in titles:
            self.assertIn(f"*Asset Title:* {title}", rendered)
        self.assertIn("Travel analytics details", rendered)
        self.assertNotIn("asset count of 0", rendered)

    async def test_answer_order_applies_before_limit_and_asset_dedup(self):
        references = [
            self._reference("C1", 11),
            self._reference("C2", 12),
            self._reference("C3", 13),
            self._reference("C4", 14),
        ]
        client = _ManifestClient(
            {
                references[0]["bundle_revision_id"]: self._manifest(
                    "ASSET/X", "补充资产 X"
                ),
                references[1]["bundle_revision_id"]: self._manifest(
                    "ASSET/A", "正文资产 A"
                ),
                references[2]["bundle_revision_id"]: self._manifest(
                    "ASSET/A", "正文资产 A"
                ),
                references[3]["bundle_revision_id"]: self._manifest(
                    "ASSET/B", "正文资产 B"
                ),
            }
        )
        answer = "1. 正文资产 B。[C4]\n2. 正文资产 A。[C2][C3]"
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": answer,
                "status": "READY",
                "used_citation_labels": ["C1", "C2", "C3", "C4"],
                "references": references,
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=client,
            domain_id=1001,
            auth_context=None,
            limit=2,
        )

        self.assertEqual(
            ["正文资产 B", "正文资产 A"],
            [card["asset_title"] for card in cards],
        )
        self.assertEqual(answer, artifact["payload"]["answer"])

    async def test_untitled_asset_item_uses_its_unique_manifest_reference(self):
        references = [
            self._reference("C1", 15),
            self._reference("C2", 16),
        ]
        client = _ManifestClient(
            {
                references[0]["bundle_revision_id"]: self._manifest(
                    "ASSET/CHATBI-1",
                    "Conversational Banking with Select AI Agents",
                ),
                references[1]["bundle_revision_id"]: self._manifest(
                    "ASSET/CHATBI-2",
                    "ChatBI Video Walking-Tour",
                ),
            }
        )
        answer = (
            "根据提供的证据，共有 **2 个** 与 ChatBI 相关的 assets。[C1][C2]\n"
            "- **Conversational Banking with Select AI Agents**："
            "这是一个 T_DEMO 类型的 Cloud 资产。[C1]\n"
            "- 另一个资产同样为 T_DEMO 类型（Cloud），"
            "解决方案标签包含 ChatBI 和 Video Walking-Tour。[C2]"
        )
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": answer,
                "status": "READY",
                "used_citation_labels": ["C1", "C2"],
                "references": references,
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=client,
            domain_id=1001,
            auth_context=None,
            limit=10,
        )

        self.assertEqual(
            [
                "Conversational Banking with Select AI Agents",
                "ChatBI Video Walking-Tour",
            ],
            [card["asset_title"] for card in cards],
        )
        self.assertEqual(answer, artifact["payload"]["answer"])

    async def test_candidate_order_does_not_override_answer_order(self):
        references = [
            self._reference(label, index)
            for index, label in enumerate(
                ("C1", "C2", "C3", "C4", "C5", "C6", "C7"),
                start=21,
            )
        ]
        titles = {
            "C1": "Conversational Banking with Select AI Agents",
            "C2": "DemoWalker: Turning AI Ideas into Reusable Business Value",
            "C3": (
                "From Text to Trends: Dynamic Visual AI Agent with Oracle "
                "Agent Factory and APEX Integration"
            ),
            "C4": "Selecting Insurance Plans with an Apex AI Agent",
            "C5": (
                "SE-HUB The Future is Agentic, Meet the Agents: "
                "AI Lakehouse, Data Science Agent & Private Agent Factory"
            ),
            "C6": "SE-HUB OCI GenAI Agentic Credit Card Recommendation",
            "C7": (
                "From Requirement to Reality: Generate a Complete Oracle "
                "Solution Pack in Minutes"
            ),
        }
        client = _ManifestClient(
            {
                reference["bundle_revision_id"]: self._manifest(
                    f"ASSET/{reference['citation_label']}",
                    titles[reference["citation_label"]],
                )
                for reference in references
            }
        )
        answer = (
            "Here are the assets related to AI，背景资料见 [C7]：\n\n"
            "**From Text to Trends: Dynamic Visual AI Agent with Oracle "
            "Agent Factory and APEX Integration**\n"
            "动态可视化方案。[C3]\n\n"
            "**Conversational Banking with Select AI Agents**\n"
            "银行对话式方案。[C1]\n\n"
            "**SE-HUB OCI GenAI Agentic Credit Card Recommendation**\n"
            "信用卡推荐方案。[C6]\n\n"
            "**DemoWalker: Turning AI Ideas into Reusable Business Value**\n"
            "资产复用方案。[C2]\n\n"
            "**SE-HUB The Future is Agentic, Meet the Agents: AI Lakehouse, "
            "Data Science Agent & Private Agent Factory**\n"
            "Agent Factory 演示方案。[C5]\n\n"
            "**Selecting Insurance Plans with an Apex AI Agent**\n"
            "保险方案。[C4]"
        )
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": answer,
                "status": "READY",
                # 候选引用顺序与回答 Asset 顺序不一致，
                # C7 仅作为背景证据，不应生成独立 Asset Template。
                "used_citation_labels": [
                    "C1",
                    "C2",
                    "C3",
                    "C4",
                    "C5",
                    "C6",
                    "C7",
                ],
                "references": references,
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=client,
            domain_id=1001,
            auth_context=None,
            # max_references 只属于旧参考资料配置，不得截断正文 Asset。
            limit=2,
        )

        self.assertEqual(
            [
                titles["C3"],
                titles["C1"],
                titles["C6"],
                titles["C2"],
                titles["C5"],
                titles["C4"],
            ],
            [card["asset_title"] for card in cards],
        )
        self.assertNotIn(titles["C7"], [card["asset_title"] for card in cards])
        self.assertEqual(answer, artifact["payload"]["answer"])

    async def test_single_or_escaped_markdown_builds_templates_in_answer_order(self):
        references = [
            self._reference("C1", 31),
            self._reference("C2", 32),
            self._reference("C3", 33),
        ]
        titles = {
            "C1": "Asset A",
            "C2": "Canonical Asset B",
            "C3": "Asset C",
        }
        client = _ManifestClient(
            {
                reference["bundle_revision_id"]: self._manifest(
                    f"ASSET/{reference['citation_label']}",
                    titles[reference["citation_label"]],
                )
                for reference in references
            }
        )
        answer = (
            "Here are the assets:\n"
            "- *Asset C*: C details.\n"
            "- \\*\\*Displayed Asset B\\*\\*: B details.\n"
            "- **Asset A**: A details."
        )
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": answer,
                "status": "READY",
                # Bundle 候选顺序与正文顺序不同，
                # 且正文不带逐项标签。
                "used_citation_labels": ["C1", "C2", "C3"],
                "references": references,
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=client,
            domain_id=1001,
            auth_context=None,
            limit=3,
        )
        self.assertEqual(
            ["Asset C", "Canonical Asset B", "Asset A"],
            [card["asset_title"] for card in cards],
        )

        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact=artifact,
            reply_config=SlackReplyConfig(
                km_portal_base_url="https://km.example.com/assets/",
                max_references=3,
            ),
            asset_cards=cards,
        )
        template_titles = [
            block["text"]["text"].removeprefix("*Asset Title:* ")
            for block in payload["blocks"]
            if block.get("type") == "section"
            and isinstance(block.get("text"), dict)
            and str(block["text"].get("text") or "").startswith(
                "*Asset Title:* "
            )
        ]
        self.assertEqual(
            ["Asset C", "Canonical Asset B", "Asset A"],
            template_titles,
        )
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("参考资料", rendered)
        self.assertNotIn("manifest.md", rendered)

    async def test_local_asset_metadata_is_not_used_when_manifest_fails(self):
        references = [
            self._reference("C1", 41),
            self._reference("C2", 42),
            self._reference("C3", 43),
        ]
        rows = {
            references[0]["bundle_revision_id"]: SimpleNamespace(
                external_asset_id="ASSET/A",
                asset_title="Asset A",
                author_mail="a@example.com",
                publish_date="2026-03-01",
                normalized_metadata_json={
                    "solution_briefing": "Asset A Briefing",
                    "create_time": "2026-03-01",
                },
            ),
            references[1]["bundle_revision_id"]: SimpleNamespace(
                external_asset_id="ASSET/B",
                asset_title="Asset B",
                author_mail="b@example.com",
                publish_date="2026-03-02",
                normalized_metadata_json={
                    "solution_briefing": "Asset B Briefing",
                    "create_time": "2026-03-02",
                },
            ),
            references[2]["bundle_revision_id"]: SimpleNamespace(
                external_asset_id="ASSET/X",
                asset_title="Background Asset X",
                author_mail="x@example.com",
                publish_date="2026-03-03",
                normalized_metadata_json={
                    "solution_briefing": "Background Briefing",
                    "create_time": "2026-03-03",
                },
            ),
        }

        class Assets:
            async def get_asset_by_kc_bundle_revision(
                self, *, domain_id, bundle_revision_id
            ):
                self.domain_id = domain_id
                return rows.get(str(bundle_revision_id))

        class Uow:
            def __init__(self):
                self.assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        class UnavailableKnowledgeCore:
            async def get_bundle_revision_preview(self, **kwargs):
                raise RuntimeError("preview unavailable")

        answer = (
            "Here are the assets:\n"
            "- **Asset B**: B details. [C2]\n"
            "- **Asset A**: A details. [C1]\n"
            "Background evidence is not an answer asset. [C3]"
        )
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": answer,
                "status": "READY",
                "used_citation_labels": ["C1", "C2", "C3"],
                "references": references,
            },
        }

        cards = await assemble_slack_asset_cards(
            artifact=artifact,
            knowledge_core_client=UnavailableKnowledgeCore(),
            domain_id=1001,
            auth_context=None,
            limit=10,
            uow_factory=Uow,
        )
        # Slack 不得绕过 Main API 回退读取 km_asset_app 本地仓储。
        self.assertEqual([], cards)

        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact=artifact,
            reply_config=SlackReplyConfig(
                km_portal_base_url="https://km.example.com/assets/",
                max_references=10,
            ),
            asset_cards=cards,
        )
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("参考资料", rendered)
        self.assertNotIn("*Asset Title:* Asset B", rendered)
        self.assertNotIn("*Solution Briefing:* Asset B Briefing", rendered)
        self.assertNotIn("*Asset Title:* Asset A", rendered)
        self.assertNotIn("Background Asset X", rendered)


class SlackRenderingAndConfigurationTest(unittest.TestCase):
    def test_query_only_field_list_preserves_kbot_answer_without_template(self):
        answer = (
            "The query returned 1 asset authored by "
            "madhumitha.k@oracle.com:\n\n"
            "**Title:** Selecting Insurance Plans with an Apex AI Agent\n"
            "**Asset ID:** 4996DC40D76BE6F8E0630D427364C968\n"
            "**Product:** Data Management -> Application Express (Apex)\n"
            "**Solution:** Oracle ChatBot\n"
            "**Asset Status:** Published\n"
            "**Ingestion Status:** READY\n"
            "**Asset Date:** 2026-08-18\n"
            "**Category:** Not specified\n"
            "**Industry:** Not specified\n"
            "The result is complete (not truncated). [Q1]"
        )
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": answer,
                "status": "READY",
                "used_citation_labels": ["Q1"],
                "references": [
                    {
                        "reference_type": "QUERY_RESULT",
                        "citation_label": "Q1",
                        "query_result_id": (
                            "01900000-0000-7000-8000-000000000701"
                        ),
                    }
                ],
                "query_results": [
                    {
                        "schema": "QUERY_RESULT.v1",
                        "rows": [
                            {
                                "ASSET_ID": (
                                    "4996DC40D76BE6F8E0630D427364C968"
                                ),
                                "ASSET_TITLE": (
                                    "Selecting Insurance Plans with an "
                                    "Apex AI Agent"
                                ),
                            }
                        ],
                    }
                ],
            },
        }

        payload = slack_visible_payload(
            render_slack_reply(
                channel_id="C1",
                user_id="U1",
                thread_ts="1.001",
                artifact=artifact,
                reply_config=SlackReplyConfig(
                    km_portal_base_url="https://km.example.com/assets/"
                ),
                # 即使调用方误传卡片，无文档回答也只能显示无 Template 正文。
                asset_cards=[
                    {
                        "asset_id": "SHOULD-NOT-RENDER",
                        "asset_title": "Should Not Render",
                        "solution_briefing": "Should Not Render",
                    }
                ],
            )
        )
        rendered = json.dumps(payload, ensure_ascii=False)

        self.assertIn(
            "*Title:* Selecting Insurance Plans with an Apex AI Agent",
            rendered,
        )
        self.assertIn(
            "*Product:* Data Management -&gt; Application Express (Apex)",
            rendered,
        )
        self.assertIn("*Solution:* Oracle ChatBot", rendered)
        self.assertIn("*Asset Status:* Published", rendered)
        self.assertIn("*Ingestion Status:* READY", rendered)
        self.assertIn("*Asset Date:* 2026-08-18", rendered)
        self.assertIn("*Category:* Not specified", rendered)
        self.assertIn("*Industry:* Not specified", rendered)
        self.assertIn("*Asset ID:* 4996DC40D76BE6F8E0630D427364C968", rendered)
        self.assertNotIn("complete (not truncated)", rendered)
        self.assertNotIn("[Q1]", rendered)
        self.assertNotIn("*Asset Title:*", rendered)
        self.assertNotIn("*Solution Briefing:*", rendered)
        self.assertNotIn("KM Link", rendered)
        self.assertNotIn("参考资料", rendered)

    def test_query_only_completion_suffix_is_removed_from_intro(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "The query returned 1 asset, and the results are "
                    "complete (not truncated). [Q1]\n\n"
                    "**Title:** Asset A"
                ),
                "status": "READY",
                "used_citation_labels": ["Q1"],
                "references": [
                    {
                        "reference_type": "QUERY_RESULT",
                        "citation_label": "Q1",
                    }
                ],
            },
        }

        payload = slack_visible_payload(
            render_slack_reply(
                channel_id="C1",
                user_id="U1",
                thread_ts="1.001",
                artifact=artifact,
                reply_config=SlackReplyConfig(),
            )
        )
        rendered = json.dumps(payload, ensure_ascii=False)

        self.assertIn("The query returned 1 asset", rendered)
        self.assertIn("*Title:* Asset A", rendered)
        self.assertNotIn("complete (not truncated)", rendered)

    def test_completion_boilerplate_is_removed_from_document_answer(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "完整回答。[C1]\n\n"
                    "The result is complete and not truncated."
                ),
                "status": "READY",
                "used_citation_labels": ["C1"],
                "references": [{
                    "reference_type": "DOCUMENT",
                    "citation_label": "C1",
                }],
            },
        }

        rendered = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(),
                )
            ),
            ensure_ascii=False,
        )

        self.assertIn("完整回答", rendered)
        self.assertNotIn("complete and not truncated", rendered)

    def test_query_warnings_and_truncation_do_not_render_hint_section(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "当前结果\n\n"
                    "## 提示\n"
                    "• 相关 Asset 超过 3 个，回答已按请求数量截断"
                ),
                "status": "READY",
                "used_citation_labels": ["Q1"],
                "references": [{
                    "reference_type": "QUERY_RESULT",
                    "citation_label": "Q1",
                }],
                "query_results": [{"truncated": True}],
                "warnings": ["问数结果已按服务端上限截断"],
            },
        }

        rendered = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(),
                )
            ),
            ensure_ascii=False,
        )

        self.assertNotIn("The result limit was exceeded", rendered)
        self.assertNotIn("问数结果已按服务端上限截断", rendered)
        self.assertNotIn("相关 Asset 超过 3 个", rendered)
        self.assertNotIn('"text": "提示"', rendered)

    def test_max_references_exceeded_does_not_render_hint_section(self):
        references = [
            {
                "reference_type": "DOCUMENT",
                "citation_label": f"C{index}",
            }
            for index in range(1, 4)
        ]
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": "完整回答",
                "status": "READY",
                "used_citation_labels": ["C1", "C2", "C3"],
                "references": references,
            },
        }

        exceeded = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(
                        max_references=2,
                        show_warnings=False,
                    ),
                )
            ),
            ensure_ascii=False,
        )
        at_limit = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(
                        max_references=3,
                        show_warnings=False,
                    ),
                )
            ),
            ensure_ascii=False,
        )

        self.assertNotIn("The result limit was exceeded", exceeded)
        self.assertNotIn('"text": "提示"', exceeded)
        self.assertNotIn("The result limit was exceeded", at_limit)

    def test_asset_query_table_preserves_kbot_answer_without_reconstruction(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "Here are all assets created since 01/01/2026.\n\n"
                    "| # | Title | Author | Product | Solution | "
                    "Industry | Asset Status | Ingestion Status | Asset Date |\n"
                    "|---|---|---|---|---|---|---|---|---|\n"
                    "| 1 | Asset B | b@example.com | ADW | RAG | "
                    "Financial Services | Published | READY | 2026-08-18 |\n"
                    "| 2 | Asset A | a@example.com | APEX | AI | "
                    "Public Sector | Published | FAILED | 2026-08-17 | [Q1]"
                ),
                "status": "READY",
                "used_citation_labels": ["Q1"],
                "references": [],
                "query_results": [
                    {
                        "schema": "QUERY_RESULT.v1",
                        "rows": [
                            {
                                "ASSET_TITLE": "Asset B",
                                "AUTHOR_MAIL": "b@example.com",
                                "ASSET_PRODUCT": "ADW",
                                "ASSET_SOLUTION": "RAG",
                                "INDUSTRY_ID": "Financial Services",
                                "ASSET_STATUS": "Published",
                                "INGESTION_STATUS": "READY",
                                "ASSET_DATE_VALUE": "2026-08-18",
                            },
                            {
                                "ASSET_TITLE": "Asset A",
                                "AUTHOR_MAIL": "a@example.com",
                                "ASSET_PRODUCT": "APEX",
                                "ASSET_SOLUTION": "AI",
                                "ASSET_STATUS": "Published",
                                "INGESTION_STATUS": "FAILED",
                                "ASSET_DATE_VALUE": "2026-08-17",
                            },
                        ],
                    }
                ],
            },
        }
        payload = slack_visible_payload(
            render_slack_reply(
                channel_id="C1",
                user_id="U1",
                thread_ts="1.001",
                artifact=artifact,
                reply_config=SlackReplyConfig(),
                asset_cards=[
                    {
                        "asset_id": "ASSET/B",
                        "asset_title": "Asset B",
                        "solution_briefing": "Asset B Briefing",
                    },
                    {
                        "asset_id": "ASSET/A",
                        "asset_title": "Asset A",
                        "solution_briefing": "Asset A Briefing",
                    },
                ],
            )
        )
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertIn("| # | Title | Author | Product | Solution |", rendered)
        self.assertIn("| 1 | Asset B | b@example.com | ADW | RAG |", rendered)
        self.assertIn("| 2 | Asset A | a@example.com | APEX | AI |", rendered)
        self.assertNotIn("<mailto:b@example.com|b@example.com>", rendered)
        self.assertNotIn("*Asset Title:*", rendered)
        self.assertNotIn("*Solution Briefing:*", rendered)
        self.assertNotIn("KM Link", rendered)

    def test_query_summary_does_not_append_query_result_fields(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "The user madhumitha.k@oracle.com has 1 asset "
                    "associated with their account."
                ),
                "status": "READY",
                "used_citation_labels": ["Q1"],
                "references": [{
                    "reference_type": "QUERY_RESULT",
                    "citation_label": "Q1",
                }],
                "query_results": [{
                    "schema": "QUERY_RESULT.v1",
                    "truncated": False,
                    "rows": [{
                        "ASSET_ID": "4996DC40D76BE6F8E0630D427364C968",
                        "ASSET_TITLE": (
                            "Selecting Insurance Plans with an Apex AI Agent"
                        ),
                        "AUTHOR_MAIL": "madhumitha.k@oracle.com",
                        "ASSET_PRODUCT": (
                            "Data Management -> Application Express (Apex)"
                        ),
                        "ASSET_SOLUTION": "Oracle ChatBot",
                        "ASSET_STATUS": "Published",
                        "INGESTION_STATUS": "READY",
                        "ASSET_DATE_VALUE": "2026-08-18",
                    }],
                }],
            },
        }

        rendered = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(max_references=5),
                )
            ),
            ensure_ascii=False,
        )

        self.assertIn("has 1 asset associated", rendered)
        self.assertNotIn(
            "Selecting Insurance Plans with an Apex AI Agent",
            rendered,
        )
        self.assertNotIn("<mailto:madhumitha.k@oracle.com", rendered)
        self.assertNotIn("*Solution:* Oracle ChatBot", rendered)
        self.assertNotIn("*Asset Status:* Published", rendered)
        self.assertNotIn("*Asset Date:* 2026-08-18", rendered)
        self.assertNotIn("4996DC40D76BE6F8E0630D427364C968", rendered)
        self.assertNotIn("*Asset Title:*", rendered)
        self.assertNotIn("KM Link", rendered)

    def test_query_rows_do_not_change_or_truncate_kbot_answer(self):
        rows = [
            {
                "ASSET_ID": f"ASSET-{index}",
                "ASSET_TITLE": f"Asset {index}",
                "AUTHOR_MAIL": f"author{index}@example.com",
                "ASSET_SOLUTION": "AI",
            }
            for index in range(1, 4)
        ]
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": "The query returned 3 assets.",
                "status": "READY",
                "used_citation_labels": ["Q1"],
                "references": [{
                    "reference_type": "QUERY_RESULT",
                    "citation_label": "Q1",
                }],
                "query_results": [{
                    "schema": "QUERY_RESULT.v1",
                    "truncated": False,
                    "rows": rows,
                }],
            },
        }

        rendered = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(max_references=2),
                )
            ),
            ensure_ascii=False,
        )

        self.assertIn("The query returned 3 assets.", rendered)
        self.assertNotIn("*1. Asset 1*", rendered)
        self.assertNotIn("*2. Asset 2*", rendered)
        self.assertNotIn("Asset 3", rendered)
        self.assertNotIn("The result limit was exceeded", rendered)

    def test_document_metadata_takes_precedence_over_query_rendering(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "Based on the provided evidence:\n\n"
                    "• **OAC Travel Asset**: Document-grounded details. [C1]"
                ),
                "status": "READY",
                "used_citation_labels": ["C1", "Q1"],
                "references": [
                    {
                        "reference_type": "DOCUMENT",
                        "citation_label": "C1",
                    },
                    {
                        "reference_type": "QUERY_RESULT",
                        "citation_label": "Q1",
                    },
                ],
                "query_results": [{
                    "schema": "QUERY_RESULT.v1",
                    "truncated": True,
                    "rows": [{
                        "ASSET_ID": "QUERY-ASSET",
                        "ASSET_TITLE": "Query Asset Must Not Override",
                        "AUTHOR_MAIL": "query@example.com",
                    }],
                }],
            },
        }

        rendered = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(max_references=5),
                    asset_cards=[{
                        "asset_id": "OAC-TRAVEL",
                        "asset_title": "OAC Travel Asset",
                        "solution_briefing": "OAC travel solution briefing",
                        "author_mail": "owner@example.com",
                        "create_time": "2026-08-18T12:00:00Z",
                    }],
                )
            ),
            ensure_ascii=False,
        )

        self.assertIn("Document-grounded details", rendered)
        self.assertIn("*Asset Title:* OAC Travel Asset", rendered)
        self.assertIn(
            "*Solution Briefing:* OAC travel solution briefing",
            rendered,
        )
        self.assertIn("KM Link", rendered)
        self.assertNotIn("Query Asset Must Not Override", rendered)
        self.assertNotIn("The result limit was exceeded", rendered)

    def test_document_without_template_does_not_fallback_to_query_rows(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": (
                    "Here are 1 matching assets; content evidence was used "
                    "for semantic conditions or preferences.\n\n"
                    "1. **A K3s HA environment operations guides** [C1]"
                ),
                "status": "READY",
                "used_citation_labels": ["C1", "Q1"],
                "references": [
                    {
                        "reference_type": "DOCUMENT",
                        "citation_label": "C1",
                    },
                    {
                        "reference_type": "QUERY_RESULT",
                        "citation_label": "Q1",
                    },
                ],
                "query_results": [{
                    "schema": "QUERY_RESULT.v1",
                    "truncated": True,
                    "rows": [
                        {
                            "ASSET_ID": f"QUERY-{index}",
                            "ASSET_TITLE": f"Unrelated Query Asset {index}",
                            "AUTHOR_MAIL": f"owner{index}@example.com",
                        }
                        for index in range(1, 11)
                    ],
                }],
            },
        }

        rendered = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(max_references=9),
                    asset_cards=[],
                )
            ),
            ensure_ascii=False,
        )

        self.assertIn("Here are 1 matching assets", rendered)
        self.assertIn("A K3s HA environment operations guides", rendered)
        self.assertNotIn("Unrelated Query Asset", rendered)
        self.assertNotIn("*Author:*", rendered)
        self.assertNotIn("The result limit was exceeded", rendered)

    def test_template_replaces_ooxml_carriage_return_marker(self):
        artifact = {
            "artifact_type": "GROUNDED_ANSWER",
            "schema_version": "GroundedAnswer.v1",
            "payload": {
                "answer": "• **Credit Card Asset**: Details. [C1]",
                "status": "READY",
                "used_citation_labels": ["C1"],
                "references": [{
                    "reference_type": "DOCUMENT",
                    "citation_label": "C1",
                }],
            },
        }

        rendered = json.dumps(
            slack_visible_payload(
                render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact=artifact,
                    reply_config=SlackReplyConfig(max_references=5),
                    asset_cards=[{
                        "asset_id": "CREDIT-CARD",
                        "asset_title": "Credit Card Asset",
                        "solution_briefing": (
                            "Two demo flows._x000D_\n"
                            "1) New customer._X000D_2) Existing customer."
                        ),
                    }],
                )
            ),
            ensure_ascii=False,
        )

        self.assertNotIn("_x000D_", rendered)
        self.assertNotIn("_X000D_", rendered)
        self.assertIn(r"Two demo flows.\n1) New customer.", rendered)
        self.assertIn(r"\n2) Existing customer.", rendered)

    def test_renders_latest_grounded_answer_without_internal_details(self):
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "GROUNDED_ANSWER",
                "schema_version": "GroundedAnswer.v1",
                "payload": {
                    "answer": "这是回答 [D1]。",
                    "status": "PARTIAL",
                    "used_citation_labels": ["Q1", "D1"],
                    "references": [
                        {
                            "reference_type": "DOCUMENT",
                            "citation_label": "D1",
                            "title": "安装手册",
                            "document_id": "internal-document-id",
                            "locator": {
                                "pages": [
                                    {
                                        "page_no": 3,
                                        "bbox": [0.1, 0.2, 0.3, 0.4],
                                    },
                                    {"page_no": 5},
                                ]
                            },
                            "resource_url": "https://private.example.com",
                        },
                        {
                            "reference_type": "QUERY_RESULT",
                            "citation_label": "Q1",
                            "query_result_id": "internal-query-id",
                            "provider": "MCP",
                            "row_count": 12,
                        },
                        {
                            "reference_type": "DOCUMENT",
                            "citation_label": "D2",
                            "title": "未使用文档",
                            "locator": {"pages": [{"page_no": 9}]},
                        },
                    ],
                    "query_results": [{"password": "raw-query-secret"}],
                    "visualizations": [{"options": "raw-chart-options"}],
                    "warnings": ["数据截至昨日"],
                }
            },
            reply_config=SlackReplyConfig(),
        )
        self.assertEqual("C1", payload["channel"])
        self.assertEqual("1.001", payload["thread_ts"])
        self.assertTrue(payload["text"].startswith("<@U1> 这是回答"))
        self.assertNotIn("Asset问答助手", json.dumps(payload, ensure_ascii=False))
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertIn(":information_source: 部分回答", rendered)
        self.assertNotIn("回答状态：", rendered)
        self.assertIn("这是回答 [D1]。", rendered)
        self.assertNotIn("参考资料", rendered)
        self.assertNotIn("来源：MCP · 12 行", rendered)
        self.assertNotIn("安装手册", rendered)
        self.assertNotIn("第 3、5 页", rendered)
        self.assertIn("数据截至昨日", rendered)
        self.assertIn("本次回答包含 1 个可视化结果", rendered)
        for private_value in (
            "internal-document-id",
            "internal-query-id",
            "bbox",
            "raw-query-secret",
            "raw-chart-options",
            "private.example.com",
            "未使用文档",
        ):
            self.assertNotIn(private_value, rendered)
        self.assertNotIn('"accessory"', rendered)

    def test_non_ready_statuses_are_rendered_in_english_without_prefix(self):
        expected_labels = {
            "CLARIFICATION_REQUIRED": "Additional information required",
            "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
            "PARTIAL": "Partial answer",
            "UNKNOWN": "Answer not fully ready",
        }
        for status, expected_label in expected_labels.items():
            with self.subTest(status=status):
                payload = render_slack_reply(
                    channel_id="C1",
                    user_id="U1",
                    thread_ts="1.001",
                    artifact={
                        "artifact_type": "GROUNDED_ANSWER",
                        "schema_version": "GroundedAnswer.v1",
                        "payload": {
                            "answer": "Which field should be used for sorting?",
                            "status": status,
                            "used_citation_labels": [],
                            "references": [],
                        },
                    },
                    reply_config=SlackReplyConfig(),
                )

                rendered = json.dumps(payload, ensure_ascii=False)
                self.assertIn(expected_label, rendered)
                self.assertNotIn("回答状态：", rendered)
                self.assertNotIn("需要补充信息", rendered)
                self.assertNotIn("现有资料不足", rendered)
                self.assertNotIn("部分回答", rendered)

    def test_status_uses_current_follow_up_language(self):
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "GROUNDED_ANSWER",
                "schema_version": "GroundedAnswer.v1",
                "payload": {
                    "answer": "No sufficient citable evidence was found.",
                    "status": "INSUFFICIENT_EVIDENCE",
                    "used_citation_labels": [],
                    "references": [],
                },
            },
            reply_config=SlackReplyConfig(),
            message_text="只需要过去一年发布的资产。",
        )

        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertIn(":information_source: 现有资料不足", rendered)
        self.assertNotIn("Insufficient evidence", rendered)

    def test_reply_options_limit_and_hide_optional_summaries(self):
        references = [
            {
                "reference_type": "DOCUMENT",
                "citation_label": f"D{index}",
                "title": f"文档 {index}",
                "locator": {"pages": [{"page_no": index + 1}]},
            }
            for index in range(4)
        ]
        references.append(
            {
                "reference_type": "QUERY_RESULT",
                "citation_label": "Q1",
                "provider": "SEMANTIC",
                "row_count": 3,
            }
        )
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "GROUNDED_ANSWER",
                "schema_version": "GroundedAnswer.v1",
                "payload": {
                    "answer": "完整回答",
                    "status": "READY",
                    "used_citation_labels": ["Q1", "D3", "D2", "D1"],
                    "references": references,
                    "warnings": ["不显示的警告"],
                    "visualizations": [{"type": "bar"}],
                },
            },
            reply_config=SlackReplyConfig(
                assistant_name="定制助手",
                max_references=2,
                show_warnings=False,
                show_query_result_summary=False,
                show_visualization_notice=False,
            ),
        )
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("定制助手", rendered)
        self.assertNotIn("参考资料", rendered)
        self.assertNotIn("文档 3", rendered)
        self.assertNotIn("文档 2", rendered)
        self.assertNotIn("[D1]", rendered)
        self.assertNotIn("[Q1]", rendered)
        self.assertNotIn("不显示的警告", rendered)
        self.assertNotIn("visualization(s)", rendered)
        self.assertNotIn("回答状态", rendered)

    def test_invalid_artifact_returns_fixed_safe_message(self):
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "INTERNAL_RESULT",
                "schema_version": "Internal.v1",
                "payload": {"answer": "sensitive internal answer"},
            },
            reply_config=SlackReplyConfig(),
        )
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertIn("answer format returned by KBot is temporarily unavailable", rendered)
        self.assertNotIn("sensitive internal answer", rendered)

    def test_answer_does_not_create_unintended_slack_mentions(self):
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "GROUNDED_ANSWER",
                "schema_version": "GroundedAnswer.v1",
                "payload": {
                    "answer": "请勿触发 <!channel>",
                    "status": "READY",
                    "used_citation_labels": [],
                    "references": [],
                },
            },
            reply_config=SlackReplyConfig(),
        )
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("<!channel>", rendered)
        self.assertIn("&lt;!channel&gt;", rendered)

    def test_answer_is_normalized_to_slack_mrkdwn(self):
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "GROUNDED_ANSWER",
                "schema_version": "GroundedAnswer.v1",
                "payload": {
                    "answer": (
                        "## ChatBI 资产\n\n"
                        "**资产名称**：Conversational Banking [C1]\n\n"
                        "**关键信息**：\n"
                        "- 使用 Select AI Agents\n"
                        "- 支持 RAG 工作流\n\n"
                        "**文档链接**："
                        '<a target="_blank" '
                        'href="https://oracle.example.com/document.docx?web=1&amp;e=x">'
                        "documentation-select-ai-agent</a> [C1]"
                    ),
                    "status": "READY",
                    "used_citation_labels": ["C1"],
                    "references": [
                        {
                            "reference_type": "DOCUMENT",
                            "citation_label": "C1",
                            "title": "manifest.md",
                            "locator": {},
                        }
                    ],
                },
            },
            reply_config=SlackReplyConfig(),
        )

        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertIn("*ChatBI 资产*", rendered)
        self.assertIn("*资产名称*：Conversational Banking [C1]", rendered)
        self.assertIn("• 使用 Select AI Agents", rendered)
        self.assertIn(
            "<https://oracle.example.com/document.docx?web=1&e=x|"
            "documentation-select-ai-agent>",
            rendered,
        )
        for original_format in ("**资产名称**", "<a ", "</a>", "&lt;a"):
            self.assertNotIn(original_format, rendered)

    def test_asset_blocks_replace_reference_blocks_after_original_answer(self):
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "GROUNDED_ANSWER",
                "schema_version": "GroundedAnswer.v1",
                "payload": {
                    "answer": "原始 KBot 回答 [C1]。",
                    "status": "READY",
                    "used_citation_labels": ["C1"],
                    "references": [
                        {
                            "reference_type": "DOCUMENT",
                            "citation_label": "C1",
                            "title": "manifest.md",
                            "locator": {},
                        }
                    ],
                },
            },
            reply_config=SlackReplyConfig(
                km_portal_base_url="https://km.example.com/assets/"
            ),
            asset_cards=[
                {
                    "asset_id": "ASSET/100",
                    "asset_title": "Claim Prediction Architecture",
                    "solution_briefing": "Real-time insights",
                    "author_mail": "AUTHOR@example.com",
                    "create_time": "2026-08-17T19:50:11Z",
                }
            ],
        )

        blocks = payload["blocks"]
        self.assertEqual("原始 KBot 回答 [C1]。", blocks[1]["text"]["text"])
        self.assertEqual({"type": "divider"}, blocks[2])
        self.assertEqual(
            "*Asset Title:* Claim Prediction Architecture",
            blocks[3]["text"]["text"],
        )
        self.assertEqual(
            "*Solution Briefing:* Real-time insights",
            blocks[4]["text"]["text"],
        )
        self.assertEqual(
            "<mailto:author@example.com|author@example.com> | 2026-08-17",
            blocks[5]["text"]["text"],
        )
        self.assertEqual(
            "https://km.example.com/assets/ASSET%2F100",
            blocks[5]["accessory"]["url"],
        )
        self.assertEqual(
            "KM Link (VPN)",
            blocks[5]["accessory"]["text"]["text"],
        )
        self.assertEqual("open_km_resource", blocks[5]["accessory"]["action_id"])
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("参考资料", rendered)
        self.assertNotIn("manifest.md", rendered)
        self.assertNotIn("Contributor", rendered)
        self.assertNotIn("Publish\\_date", rendered)
        self.assertNotIn("VPN required", rendered)

    def test_optional_asset_metadata_is_visually_empty_but_keeps_km_link(self):
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "GROUNDED_ANSWER",
                "schema_version": "GroundedAnswer.v1",
                "payload": {
                    "answer": "**Asset A**: A details. [C1]",
                    "status": "READY",
                    "used_citation_labels": ["C1"],
                    "references": [
                        {
                            "reference_type": "DOCUMENT",
                            "citation_label": "C1",
                        }
                    ],
                },
            },
            reply_config=SlackReplyConfig(
                km_portal_base_url="https://km.example.com/assets/"
            ),
            asset_cards=[
                {
                    "asset_id": "ASSET/A",
                    "asset_title": "Asset A",
                    "solution_briefing": "A briefing",
                }
            ],
        )

        self.assertEqual("\u200b", payload["blocks"][5]["text"]["text"])
        self.assertEqual(
            "https://km.example.com/assets/ASSET%2FA",
            payload["blocks"][5]["accessory"]["url"],
        )

    def test_max_asset_templates_fit_one_final_message_in_order(self):
        cards = [
            {
                "asset_id": f"ASSET/{index}",
                "asset_title": f"Asset {index}",
                "solution_briefing": f"Briefing {index}",
            }
            for index in range(1, 11)
        ]
        payloads = render_slack_replies(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "GROUNDED_ANSWER",
                "schema_version": "GroundedAnswer.v1",
                "payload": {
                    "answer": "\n\n".join(
                        f"**Asset {index}**: Briefing {index}"
                        for index in range(1, 11)
                    ),
                    "status": "READY",
                    "used_citation_labels": ["C1"],
                    "references": [
                        {
                            "reference_type": "DOCUMENT",
                            "citation_label": "C1",
                        }
                    ],
                },
            },
            reply_config=SlackReplyConfig(
                km_portal_base_url="https://km.example.com/assets/"
            ),
            asset_cards=cards,
        )

        self.assertEqual(1, len(payloads))
        self.assertTrue(all(len(payload["blocks"]) <= 50 for payload in payloads))
        template_titles = [
            block["text"]["text"].removeprefix("*Asset Title:* ")
            for payload in payloads
            for block in payload["blocks"]
            if isinstance(block, dict)
            and block.get("type") == "section"
            and isinstance(block.get("text"), dict)
            and str(block["text"].get("text") or "").startswith(
                "*Asset Title:* "
            )
        ]
        self.assertEqual(
            [f"Asset {index}" for index in range(1, 11)],
            template_titles,
        )

    def test_references_never_render_when_asset_cards_are_empty(self):
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "artifact_type": "GROUNDED_ANSWER",
                "schema_version": "GroundedAnswer.v1",
                "payload": {
                    "answer": "No matching asset was found. [C1]",
                    "status": "INSUFFICIENT_EVIDENCE",
                    "used_citation_labels": ["C1"],
                    "references": [
                        {
                            "reference_type": "DOCUMENT",
                            "citation_label": "C1",
                            "title": "manifest.md",
                        }
                    ],
                },
            },
            reply_config=SlackReplyConfig(),
            asset_cards=[],
        )

        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("参考资料", rendered)
        self.assertNotIn("manifest.md", rendered)
        self.assertNotIn("*Asset Title:*", rendered)

    def test_callback_requires_url_only_when_enabled(self):
        SlackExternalCallbackConfig(enabled=False, url="")
        with self.assertRaises(ValueError):
            SlackExternalCallbackConfig(enabled=True, url="")

    def test_callback_payload_keeps_3_3_fields(self):
        payload = build_callback_payload(
            user_id="U1",
            username="User",
            user_email="user@example.com",
            user_question="Question",
            request_date=date(2026, 8, 4),
        )
        self.assertEqual(
            {
                "user_id": "U1",
                "username": "User",
                "user_email": "user@example.com",
                "user_question": "Question",
                "request_time": "2026-08-04",
            },
            payload,
        )

    def test_slack_is_disabled_by_default(self):
        config = SlackIntegrationConfig()
        self.assertFalse(config.enabled)
        self.assertFalse(config.debug.callback_payload_log_enabled)
        self.assertFalse(config.debug.slack_reply_dump_enabled)
        self.assertEqual("Asset问答助手", config.reply.assistant_name)
        self.assertEqual(10, config.reply.max_references)
        self.assertEqual(
            "https://apex.oraclecorp.com/pls/apex/"
            "f?p=2018:130:::::P130_SUB,P130_ASSET_ID:SP,",
            config.reply.km_portal_base_url,
        )

    def test_reply_rejects_invalid_portal_url(self):
        with self.assertRaises(ValueError):
            SlackReplyConfig(km_portal_base_url="apex.invalid/path")

    def test_reply_rejects_more_than_ten_references(self):
        with self.assertRaises(ValueError):
            SlackReplyConfig(max_references=11)

    def test_workspace_reads_secrets_from_named_environment_variables(self):
        config = SlackWorkspaceConfig(
            workspace_id="T1",
            domain_id=1001,
            agent_id="019fcbe0-e46c-7d33-907b-9d1621a2998f",
            signing_secret_env="KBOT_SLACK_TEST_SIGNING_SECRET",
            bot_token_env="KBOT_SLACK_TEST_BOT_TOKEN",
        )
        with patch.dict(
            os.environ,
            {
                "KBOT_SLACK_TEST_SIGNING_SECRET": "signing-secret",
                "KBOT_SLACK_TEST_BOT_TOKEN": "bot-token",
            },
            clear=True,
        ):
            self.assertEqual("signing-secret", config.require_signing_secret())
            self.assertEqual("bot-token", config.require_bot_token())

    def test_debug_files_use_restricted_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SlackIntegrationConfig(
                debug={
                    "slack_reply_dump_enabled": True,
                    "slack_reply_dump_dir": directory,
                }
            )
            callback_log = Path(directory) / "callback.log"
            service = SlackDispatchService(
                uow_factory=None,
                main_api_client=None,
                slack_config=config,
                worker_id="test-worker",
                http_session=None,
                callback_debug_log_path=callback_log,
            )

            service._dump_slack_payload(
                {"channel": "C1", "text": "answer"},
                "T1",
                "E1",
                "final",
            )
            service._append_callback_debug(
                {"callback_payload": {"user_id": "U1"}}
            )

            reply_file = next(
                path
                for path in Path(directory).glob("*.json")
            )
            self.assertEqual(0o600, os.stat(reply_file).st_mode & 0o777)
            self.assertEqual(0o600, os.stat(callback_log).st_mode & 0o777)

    def test_visible_payload_hides_citations_without_mutating_raw_payload(self):
        raw_payload = {
            "channel": "C1",
            "text": "Answer [C1]. More [C2][C10]! Keep [Apex].",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "[D1] Document details [Q1]",
                    },
                }
            ],
            "metadata": "internal [C1]",
            "url": "https://example.com/files/[C1]",
        }

        visible = slack_visible_payload(raw_payload)

        self.assertEqual(
            "Answer. More! Keep [Apex].",
            visible["text"],
        )
        self.assertEqual(
            "Document details",
            visible["blocks"][0]["text"]["text"],
        )
        self.assertEqual("internal [C1]", visible["metadata"])
        self.assertEqual("https://example.com/files/[C1]", visible["url"])
        self.assertEqual(
            "Answer [C1]. More [C2][C10]! Keep [Apex].",
            raw_payload["text"],
        )

    def test_visible_reply_splits_bulleted_assets_into_sections(self):
        answer = (
            "Assets found:\n"
            "• *Asset One*：First details [C1]\n"
            "• *Asset Two*：Second details [C2]"
        )
        split_at = answer.index("Asset Two") + len("Asset Tw")
        raw_payload = {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": answer[:split_at],
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": answer[split_at:],
                    },
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Asset Title:* Asset One",
                    },
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Asset Title:* Asset Two",
                    },
                },
            ]
        }

        visible = slack_visible_payload(raw_payload)

        self.assertEqual(
            "Assets found:",
            visible["blocks"][0]["text"]["text"],
        )
        self.assertEqual(
            "• *Asset One*：First details",
            visible["blocks"][1]["text"]["text"],
        )
        self.assertEqual(
            "• *Asset Two*：Second details",
            visible["blocks"][2]["text"]["text"],
        )
        self.assertEqual(
            answer,
            raw_payload["blocks"][0]["text"]["text"]
            + raw_payload["blocks"][1]["text"]["text"],
        )
        self.assertEqual(
            "*Asset Title:* Asset Two",
            visible["blocks"][6]["text"]["text"],
        )


class SlackVisibleDeliveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_dump_keeps_citations_while_slack_receives_hidden_copy(self):
        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def json(self):
                return {"ok": True, "ts": "1.001"}

        class Session:
            def __init__(self):
                self.kwargs = None

            def post(self, url, **kwargs):
                self.kwargs = kwargs
                return Response()

        with tempfile.TemporaryDirectory() as directory:
            session = Session()
            service = SlackDispatchService(
                uow_factory=None,
                main_api_client=None,
                slack_config=SlackIntegrationConfig(
                    debug={
                        "slack_reply_dump_enabled": True,
                        "slack_reply_dump_dir": directory,
                    }
                ),
                worker_id="test-worker",
                http_session=session,
            )
            raw_payload = {
                "channel": "C1",
                "text": "Answer [C1].",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Details [C2]"},
                    }
                ],
            }

            await service._post_slack(
                bot_token="token",
                payload=raw_payload,
                workspace_id="T1",
                event_key="E1",
                delivery_type="final",
            )

            dump_file = next(Path(directory).glob("*.json"))
            dumped = json.loads(dump_file.read_text(encoding="utf-8"))
            self.assertEqual("Answer [C1].", dumped["text"])
            self.assertEqual("Details [C2]", dumped["blocks"][0]["text"]["text"])
            self.assertEqual("Answer.", session.kwargs["json"]["text"])
            self.assertEqual(
                "Details",
                session.kwargs["json"]["blocks"][0]["text"]["text"],
            )
            self.assertEqual("Answer [C1].", raw_payload["text"])


class SlackCallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_callback_uses_exact_payload_without_auth_header(self):
        class Response:
            status = 204

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        class Session:
            def __init__(self):
                self.url = None
                self.kwargs = None

            def post(self, url, **kwargs):
                self.url = url
                self.kwargs = kwargs
                return Response()

        session = Session()
        config = SlackIntegrationConfig(
            external_callback={
                "enabled": True,
                "url": "https://callback.example.com/events",
            }
        )
        service = SlackDispatchService(
            uow_factory=None,
            main_api_client=None,
            slack_config=config,
            worker_id="test-worker",
            http_session=session,
        )
        service._fetch_user_info = AsyncMock(
            return_value=("User", "user@example.com")
        )
        await service._send_external_callback(
            bot_token="token",
            slack_user_id="U1",
            message_text="Question",
            workspace_id="T1",
            event_id="E1",
        )

        self.assertEqual(
            "https://callback.example.com/events",
            session.url,
        )
        self.assertEqual(
            {"Content-Type": "application/json"},
            session.kwargs["headers"],
        )
        self.assertEqual(
            {
                "user_id",
                "username",
                "user_email",
                "user_question",
                "request_time",
            },
            set(session.kwargs["json"]),
        )


class SlackDispatchMainApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_new_conversation_sends_original_question_to_main_api(self):
        agent_id = UUID("019ff999-6789-799b-97c3-500879812f7b")
        inbox_id = UUID("01a00e17-084d-7370-935e-5d8702b26ad1")
        conversation_id = UUID("01a00e17-084d-7370-935e-5d8702b26ad2")

        class ExpiringInbox(SimpleNamespace):
            _protected = {
                "workspace_id",
                "channel_id",
                "slack_user_id",
                "root_thread_ts",
                "message_text",
                "event_id",
                "callback_sent_at",
            }

            def __getattribute__(self, name):
                if name in object.__getattribute__(self, "_protected"):
                    if not object.__getattribute__(self, "_active"):
                        raise AssertionError(
                            f"UoW 退出后访问了 Inbox 属性：{name}"
                        )
                return object.__getattribute__(self, name)

        inbox = ExpiringInbox(
            _active=False,
            inbox_id=inbox_id,
            workspace_id="T1",
            channel_id="C1",
            slack_user_id="U1",
            root_thread_ts="1723880000.123456",
            message_text="any assets of madhumitha.k@oracle.com；",
            event_id="E1",
            callback_sent_at=None,
        )
        slack_repository = SimpleNamespace(
            get_inbox=AsyncMock(return_value=inbox),
            get_delivery=AsyncMock(return_value=None),
            add_delivery=AsyncMock(),
            get_thread=AsyncMock(side_effect=[None, None, None]),
            add_thread=AsyncMock(),
        )

        class UnitOfWork:
            slack = slack_repository
            commit = AsyncMock()

            async def __aenter__(self):
                inbox._active = True
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                inbox._active = False
                return None

        main_api_client = SimpleNamespace(
            create_conversation=AsyncMock(
                return_value={"conversation_id": str(conversation_id)}
            ),
            get_conversation=AsyncMock(return_value={"row_version": 1}),
            create_conversation_turn=AsyncMock(
                return_value={
                    "turn_id": "01a00e17-084d-7370-935e-5d8702b26ad3",
                    "run_id": "01a00e17-084d-7370-935e-5d8702b26ad4",
                }
            ),
        )
        config = SlackIntegrationConfig(
            enabled=True,
            workspaces=[
                {
                    "workspace_id": "T1",
                    "domain_id": 1001,
                    "agent_id": str(agent_id),
                }
            ],
        )
        service = SlackDispatchService(
            uow_factory=UnitOfWork,
            main_api_client=main_api_client,
            slack_config=config,
            worker_id="test-worker",
            http_session=None,
        )

        await service._start_run(inbox_id)

        create_payload = main_api_client.create_conversation.await_args.kwargs[
            "payload"
        ]
        self.assertNotIn("execution_spec", create_payload)
        turn_payload = main_api_client.create_conversation_turn.await_args.kwargs[
            "payload"
        ]
        self.assertNotIn("execution_spec", turn_payload)
        self.assertEqual(
            "any assets of madhumitha.k@oracle.com；",
            turn_payload["input"],
        )
        self.assertNotIn("collection_ids", turn_payload)
        self.assertNotIn("security_level", turn_payload)
        self.assertNotIn("route_type", turn_payload)
        self.assertNotIn("task_type", turn_payload)

    async def test_run_check_does_not_read_inbox_after_uow_exit(self):
        agent_id = UUID("019ff999-6789-799b-97c3-500879812f7b")
        inbox_id = UUID("01a00e5c-6d3e-7def-88a4-b9dfc1391ecc")
        run_id = UUID("01a00e5c-6d3e-7def-88a4-b9dfc1391ecd")

        class ExpiringInbox(SimpleNamespace):
            _protected = {
                "workspace_id",
                "slack_user_id",
                "channel_id",
                "root_thread_ts",
                "run_id",
            }

            def __getattribute__(self, name):
                if name in object.__getattribute__(self, "_protected"):
                    if not object.__getattribute__(self, "_active"):
                        raise AssertionError(
                            f"UoW 退出后访问了 Inbox 属性：{name}"
                        )
                return object.__getattribute__(self, name)

        inbox = ExpiringInbox(
            _active=True,
            workspace_id="T1",
            slack_user_id="U1",
            channel_id="C1",
            root_thread_ts="1723880000.123456",
            run_id=run_id,
        )
        current = SimpleNamespace(
            status="RUNNING",
            lease_owner="test-worker",
            lease_until=None,
            updated_at=None,
        )
        first_repository = SimpleNamespace(
            get_inbox=AsyncMock(return_value=inbox)
        )
        second_repository = SimpleNamespace(
            get_inbox=AsyncMock(return_value=current),
            get_delivery=AsyncMock(return_value=None),
            add_delivery=AsyncMock(),
        )

        class FirstUnitOfWork:
            slack = first_repository

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                inbox._active = False

        class SecondUnitOfWork:
            slack = second_repository
            commit = AsyncMock()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        units = iter((FirstUnitOfWork(), SecondUnitOfWork()))
        main_api_client = SimpleNamespace(
            get_run=AsyncMock(return_value={"status": "COMPLETED"}),
            get_result=AsyncMock(
                return_value={
                    "artifact_type": "GROUNDED_ANSWER",
                    "schema_version": "GroundedAnswer.v1",
                    "payload": {
                        "answer": "完成",
                        "status": "READY",
                        "used_citation_labels": [],
                        "references": [],
                    },
                }
            ),
        )
        service = SlackDispatchService(
            uow_factory=lambda: next(units),
            main_api_client=main_api_client,
            slack_config=SlackIntegrationConfig(
                enabled=True,
                workspaces=[
                    {
                        "workspace_id": "T1",
                        "domain_id": 1001,
                        "agent_id": str(agent_id),
                    }
                ],
            ),
            worker_id="test-worker",
            http_session=None,
        )

        with patch(
            "km_asset_app.application.slack_dispatch.render_slack_replies",
            return_value=[
                {
                    "channel": "C1",
                    "thread_ts": "1723880000.123456",
                    "text": "正文",
                    "blocks": [],
                },
                {
                    "channel": "C1",
                    "thread_ts": "1723880000.123456",
                    "text": "Asset Templates（续）",
                    "blocks": [],
                },
            ],
        ):
            await service._check_run(inbox_id)

        self.assertEqual("COMPLETED", current.status)
        self.assertEqual(1, second_repository.add_delivery.await_count)
        delivery_types = [
            call.args[0].delivery_type
            for call in second_repository.add_delivery.await_args_list
        ]
        self.assertEqual(["FINAL"], delivery_types)


if __name__ == "__main__":
    unittest.main()
