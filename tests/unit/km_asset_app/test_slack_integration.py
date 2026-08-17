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

from km_asset_app.application.slack_dispatch import SlackDispatchService
from km_asset_app.application.slack_intake import (
    SlackIntakeService,
    SlackWebhookError,
    parse_message_event,
    verify_slack_signature,
)
from km_asset_app.application.slack_rendering import (
    build_callback_payload,
    render_slack_reply,
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


class SlackRenderingAndConfigurationTest(unittest.TestCase):
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
        self.assertTrue(payload["text"].startswith("<@U1> Asset问答助手："))
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertIn("回答状态：部分回答", rendered)
        self.assertLess(rendered.index("[Q1]"), rendered.rindex("[D1]"))
        self.assertIn("来源：MCP · 12 行", rendered)
        self.assertIn("[D1] 安装手册", rendered)
        self.assertIn("第 3、5 页", rendered)
        self.assertIn("数据截至昨日", rendered)
        self.assertIn("包含 1 个可视化结果", rendered)
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
        self.assertIn("定制助手", rendered)
        self.assertIn("[D3] 文档 3", rendered)
        self.assertIn("[D2] 文档 2", rendered)
        self.assertNotIn("[D1]", rendered)
        self.assertNotIn("[Q1]", rendered)
        self.assertNotIn("不显示的警告", rendered)
        self.assertNotIn("可视化结果", rendered)
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
        self.assertIn("回答格式暂不可用", rendered)
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
        self.assertEqual(5, config.reply.max_references)
        self.assertEqual(
            "https://apex.oraclecorp.com/pls/apex/"
            "f?p=2018:130:::::P130_SUB,P130_ASSET_ID:SP,",
            config.reply.km_portal_base_url,
        )

    def test_reply_rejects_invalid_portal_url(self):
        with self.assertRaises(ValueError):
            SlackReplyConfig(km_portal_base_url="apex.invalid/path")

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
                agent_client=None,
                km_asset_client=None,
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
            agent_client=None,
            km_asset_client=None,
            slack_config=config,
            worker_id="test-worker",
            http_session=session,
        )
        service._fetch_user_info = AsyncMock(
            return_value=("User", "user@example.com")
        )
        inbox = SimpleNamespace(
            slack_user_id="U1",
            message_text="Question",
            workspace_id="T1",
            event_id="E1",
        )
        workspace = SimpleNamespace(require_bot_token=lambda: "token")

        await service._send_external_callback(inbox, workspace)

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


class SlackDispatchExecutionSpecTest(unittest.IsolatedAsyncioTestCase):
    async def test_new_conversation_uses_authoritative_execution_spec(self):
        agent_id = UUID("019ff999-6789-799b-97c3-500879812f7b")
        inbox_id = UUID("01a00e17-084d-7370-935e-5d8702b26ad1")
        conversation_id = UUID("01a00e17-084d-7370-935e-5d8702b26ad2")
        inbox = SimpleNamespace(
            inbox_id=inbox_id,
            workspace_id="T1",
            channel_id="C1",
            slack_user_id="U1",
            root_thread_ts="1723880000.123456",
            message_text="<@BOT> hello",
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
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        execution_spec = {
            "agent_id": str(agent_id),
            "domain_id": 1001,
            "model_bindings": {},
        }
        km_asset_client = SimpleNamespace(
            execution_spec=AsyncMock(return_value=execution_spec)
        )
        agent_client = SimpleNamespace(
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
            agent_client=agent_client,
            km_asset_client=km_asset_client,
            slack_config=config,
            worker_id="test-worker",
            http_session=None,
        )

        await service._start_run(inbox_id)

        km_call = km_asset_client.execution_spec.await_args.kwargs
        self.assertEqual(agent_id, km_call["agent_id"])
        self.assertEqual(1001, km_call["domain_id"])
        self.assertEqual(("km_asset.slack.dispatch",), km_call["scopes"])
        create_payload = agent_client.create_conversation.await_args.kwargs[
            "payload"
        ]
        self.assertIs(execution_spec, create_payload["execution_spec"])

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
        agent_client = SimpleNamespace(
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
            agent_client=agent_client,
            km_asset_client=None,
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

        await service._check_run(inbox_id)

        self.assertEqual("COMPLETED", current.status)
        second_repository.add_delivery.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
