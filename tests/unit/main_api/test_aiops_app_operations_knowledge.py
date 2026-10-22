"""AIOps 运维知识库 BFF 的稳定业务边界测试。"""

import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

from main_api.api.aiops_app import (
    OperationsKnowledgeConversationCreatePayload,
    create_operations_knowledge_conversation,
    delete_operations_knowledge_asset,
    operations_knowledge_overview,
    upload_operations_knowledge,
    upload_operations_manual_version,
)
from platform_core.identity import uuid7
from platform_core.contracts import AuthContext, PrincipalKind


class _AIOpsClient:
    def __init__(self):
        self.uploaded = None

    async def get_operations_knowledge_overview(self, **kwargs):
        return {"published_manuals": 2, "published_cases": 1}

    async def upload_operations_manual_version(self, **kwargs):
        self.uploaded = kwargs
        return {"version": {"version_no": 2}, "replayed": False}

    async def upload_operations_knowledge(self, **kwargs):
        self.uploaded = kwargs
        return {"grouping_mode": "SINGLE_BUNDLE", "items": []}

    async def get_operations_knowledge_execution_spec(self, **kwargs):
        return {
            "consumer_agent_id": str(uuid7()),
            "owner_app_id": "aiops",
        }

    async def delete_operations_knowledge_asset(self, asset_id, **kwargs):
        self.deleted = {"asset_id": asset_id, **kwargs}


def _context():
    return AuthContext(
        principal_kind=PrincipalKind.PORTAL,
        client_id="aiops-user-session",
        api_key_id="aiops-user-token",
        request_id="aiops-knowledge-test",
        trace_id="aiops-knowledge-test",
        app_id="aiops",
        domain_id="42",
        asserted_user_id="aiopsadmin",
    )


def _request(client, *, body: bytes = b"", headers=None):
    async def stream():
        yield body

    return SimpleNamespace(
        headers=headers or {},
        stream=stream,
        state=SimpleNamespace(auth_context=_context()),
        app=SimpleNamespace(
            state=SimpleNamespace(
                aiops_client=client,
                agent_runtime_client=AsyncMock(),
            )
        ),
    )


class AIOpsOperationsKnowledgeBffTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_conversation_only_uses_server_signed_spec(self):
        client = _AIOpsClient()
        request = _request(client)
        request.app.state.agent_runtime_client.create_conversation.return_value = {
            "owner_app_id": "aiops"
        }
        with (
            patch(
                "main_api.api.aiops_app._require",
                AsyncMock(return_value=(42, "aiopsadmin", object())),
            ),
            patch(
                "main_api.api.aiops_app._effective_security_level",
                AsyncMock(return_value=2),
            ),
        ):
            await create_operations_knowledge_conversation(
                OperationsKnowledgeConversationCreatePayload(), request
            )
        payload = (
            request.app.state.agent_runtime_client.create_conversation
            .await_args.kwargs["payload"]
        )
        self.assertEqual(payload["execution_spec"]["owner_app_id"], "aiops")
        self.assertNotIn("collection_ids", payload)
        self.assertNotIn("security_level", payload)

    async def test_overview_uses_aiops_registry_client(self):
        client = _AIOpsClient()
        with patch(
            "main_api.api.aiops_app._require",
            AsyncMock(return_value=(42, "aiopsadmin", object())),
        ):
            result = await operations_knowledge_overview(_request(client))
        self.assertEqual(2, result["published_manuals"])

    async def test_delete_asset_forwards_optimistic_row_version(self):
        client = _AIOpsClient()
        asset_id = uuid7()
        with patch(
            "main_api.api.aiops_app._require",
            AsyncMock(return_value=(42, "aiopsadmin", object())),
        ):
            response = await delete_operations_knowledge_asset(
                asset_id, _request(client), if_match='"rv-7"'
            )

        self.assertEqual(204, response.status_code)
        self.assertEqual(asset_id, client.deleted["asset_id"])
        self.assertEqual(7, client.deleted["expected_row_version"])

    async def test_manual_new_version_forwards_asset_optimistic_version(self):
        client = _AIOpsClient()
        asset_id = uuid7()
        content = b"# Oracle Data Guard v2"
        digest = hashlib.sha256(content).hexdigest()
        metadata = {"display_name": "Data Guard 手册", "security_level": 2}
        request = _request(client, body=content, headers={
            "Content-Type": "text/markdown",
            "Idempotency-Key": f"aiops-manual:{asset_id}:{digest}",
            "X-File-Name": quote("data-guard-v2.md", safe=""),
            "X-Content-SHA256": digest,
            "X-Upload-Metadata": quote(json.dumps(metadata, ensure_ascii=False), safe=""),
            "X-Expected-Asset-Row-Version": "3",
        })
        with patch(
            "main_api.api.aiops_app._require",
            AsyncMock(return_value=(42, "aiopsadmin", object())),
        ):
            result = await upload_operations_manual_version(asset_id, request)
        self.assertEqual(2, result["version"]["version_no"])
        self.assertEqual(asset_id, client.uploaded["asset_id"])
        self.assertEqual(3, client.uploaded["expected_asset_row_version"])

    async def test_multipart_upload_streams_request_with_idempotency_key(self):
        client = _AIOpsClient()
        request = _request(
            client,
            body=b"multipart-body",
            headers={
                "Content-Type": "multipart/form-data; boundary=test",
                "Idempotency-Key": "aiops-upload-test",
            },
        )
        with patch(
            "main_api.api.aiops_app._require",
            AsyncMock(return_value=(42, "aiopsadmin", object())),
        ):
            result = await upload_operations_knowledge(
                request,
                idempotency_key="aiops-upload-test",
            )

        self.assertEqual("SINGLE_BUNDLE", result["grouping_mode"])
        self.assertEqual("multipart/form-data; boundary=test", client.uploaded["content_type"])
        self.assertEqual("aiops-upload-test", client.uploaded["idempotency_key"])
        self.assertEqual(
            b"multipart-body",
            b"".join([chunk async for chunk in client.uploaded["body"]]),
        )


if __name__ == "__main__":
    unittest.main()
