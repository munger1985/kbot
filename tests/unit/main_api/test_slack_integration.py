"""Slack 4.0 接入契约的单元测试。"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from main_api.application.slack_dispatch import SlackDispatchService
from main_api.application.slack_intake import (
    parse_message_event,
    verify_slack_signature,
)
from main_api.application.slack_rendering import (
    build_callback_payload,
    render_slack_reply,
)
from main_api.config import SlackExternalCallbackConfig, SlackIntegrationConfig


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


class SlackRenderingAndConfigurationTest(unittest.TestCase):
    def test_renders_valid_section_blocks_and_three_references(self):
        payload = render_slack_reply(
            channel_id="C1",
            user_id="U1",
            thread_ts="1.001",
            artifact={
                "payload": {
                    "answer": "answer",
                    "references": [
                        {
                            "citation_label": f"D{index}",
                            "title": f"doc-{index}",
                            "locator": {"page": index + 1},
                            "url": "https://example.com",
                        }
                        for index in range(4)
                    ],
                }
            },
        )
        self.assertEqual("C1", payload["channel"])
        self.assertTrue(all(block["type"] != "markdown" for block in payload["blocks"]))
        self.assertEqual(
            3,
            sum(
                1
                for block in payload["blocks"]
                if block.get("accessory", {}).get("type") == "button"
            ),
        )
        self.assertIn("[D0] doc-0", payload["blocks"][2]["text"]["text"])
        self.assertIn("page: 1", payload["blocks"][2]["text"]["text"])

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


if __name__ == "__main__":
    unittest.main()
