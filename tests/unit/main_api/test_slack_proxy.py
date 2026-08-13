"""Main API Slack 公共适配器测试。"""

from __future__ import annotations

import base64
import hashlib
import unittest
from types import SimpleNamespace
from uuid import UUID

from main_api.api.slack import receive_slack_event


class _Headers(dict):
    def items(self):
        return super().items()


class _Request:
    def __init__(self, body: bytes, client):
        self._body = body
        self.headers = _Headers(
            {
                "content-type": "application/json; charset=utf-8",
                "x-slack-request-timestamp": "1720000000",
                "x-slack-signature": "v0=test",
                "authorization": "Bearer must-not-forward",
                "X-Request-ID": "request-1",
            }
        )
        self.client = SimpleNamespace(host="127.0.0.1")
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                service_name="kbot-main-api",
                main_api_settings=SimpleNamespace(
                    integrations=SimpleNamespace(
                        slack_public_requests_per_minute=120,
                        slack_public_max_webhook_bytes=1024 * 1024,
                    )
                ),
                km_asset_client=client,
            )
        )

    async def stream(self):
        yield self._body[:5]
        yield self._body[5:]


class _KmAssetClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.envelope = None
        self.auth_context = None

    async def intake_slack_event(self, *, envelope, auth_context):
        self.envelope = envelope
        self.auth_context = auth_context
        return self.payload


class SlackProxyTest(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_raw_body_and_only_forwards_slack_headers(self):
        body = b'{"type":"event_callback", "event_id":"E1"}'
        receipt_id = UUID("019fcbe0-e46c-7d33-907b-9d1621a2998f")
        client = _KmAssetClient(
            {
                "schema_version": "slack.webhook.internal.v1",
                "receipt_id": str(receipt_id),
                "accepted": True,
                "duplicate": False,
                "ignored": False,
                "challenge": None,
            }
        )

        result = await receive_slack_event(_Request(body, client))

        self.assertEqual(receipt_id, result.receipt_id)
        self.assertEqual(
            body,
            base64.b64decode(client.envelope.raw_body_base64),
        )
        self.assertEqual(
            hashlib.sha256(body).hexdigest(),
            client.envelope.raw_body_hash,
        )
        self.assertEqual(
            {
                "x-slack-request-timestamp": "1720000000",
                "x-slack-signature": "v0=test",
            },
            client.envelope.signature_headers,
        )
        self.assertEqual("kbot-main-api", client.auth_context.calling_service)

    async def test_maps_url_verification_to_plain_text(self):
        client = _KmAssetClient(
            {
                "schema_version": "slack.webhook.internal.v1",
                "receipt_id": None,
                "accepted": True,
                "duplicate": False,
                "ignored": False,
                "challenge": "challenge-value",
            }
        )

        response = await receive_slack_event(_Request(b"{}", client))

        self.assertEqual(200, response.status_code)
        self.assertEqual(b"challenge-value", response.body)
        self.assertEqual("text/plain", response.media_type)


if __name__ == "__main__":
    unittest.main()
