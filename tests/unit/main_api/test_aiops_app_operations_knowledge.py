"""AIOps 运维知识库 BFF 的稳定业务边界测试。"""

import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

from main_api.api.aiops_app import (
    operations_knowledge_overview,
    upload_operations_manual,
    upload_operations_manual_version,
)
from platform_core.identity import uuid7
from platform_core.contracts import AuthContext, PrincipalKind


class _AIOpsClient:
    def __init__(self):
        self.uploaded = None

    async def get_operations_knowledge_overview(self, **kwargs):
        return {"published_manuals": 2, "published_cases": 1}

    async def upload_operations_manual(self, **kwargs):
        self.uploaded = kwargs
        return {"asset": {"status": "PROCESSING"}, "replayed": False}

    async def upload_operations_manual_version(self, **kwargs):
        self.uploaded = kwargs
        return {"version": {"version_no": 2}, "replayed": False}


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
        app=SimpleNamespace(state=SimpleNamespace(aiops_client=client)),
    )


class AIOpsOperationsKnowledgeBffTest(unittest.IsolatedAsyncioTestCase):
    async def test_overview_uses_aiops_registry_client(self):
        client = _AIOpsClient()
        with patch(
            "main_api.api.aiops_app._require",
            AsyncMock(return_value=(42, "aiopsadmin", object())),
        ):
            result = await operations_knowledge_overview(_request(client))
        self.assertEqual(2, result["published_manuals"])

    async def test_manual_upload_forwards_business_metadata_not_kc_contract(self):
        client = _AIOpsClient()
        content = b"# Oracle Data Guard"
        digest = hashlib.sha256(content).hexdigest()
        metadata = {
            "display_name": "Data Guard 手册",
            "security_level": 1,
            "scopes": [{"kind": "DATABASE_TYPE", "value": "ORACLE"}],
        }
        request = _request(client, body=content, headers={
            "Content-Type": "text/markdown",
            "Idempotency-Key": f"aiops-manual:{digest}",
            "X-File-Name": quote("data-guard.md", safe=""),
            "X-Content-SHA256": digest,
            "X-Upload-Metadata": quote(json.dumps(metadata, ensure_ascii=False), safe=""),
        })
        with patch(
            "main_api.api.aiops_app._require",
            AsyncMock(return_value=(42, "aiopsadmin", object())),
        ):
            result = await upload_operations_manual(request)
        self.assertEqual("PROCESSING", result["asset"]["status"])
        self.assertEqual(metadata, client.uploaded["metadata"])
        self.assertEqual(digest, client.uploaded["content_sha256"])
        self.assertNotIn("collection_id", client.uploaded)

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


if __name__ == "__main__":
    unittest.main()
