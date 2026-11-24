"""AIOps 运维知识库 BFF 的稳定业务边界测试。"""

import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

from fastapi import HTTPException, Response

from main_api.api.aiops_app import (
    router,
    delete_operations_knowledge_asset,
    update_operations_knowledge_base,
    operations_knowledge_overview,
    search_managed_operations_knowledge,
    upload_operations_knowledge,
    upload_operations_manual_version,
)
from platform_core.identity import uuid7
from platform_core.contracts.aiops.knowledge_base import KnowledgeBaseWrite
from platform_core.contracts.aiops.knowledge_search import KnowledgeManagementSearchRequest
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
        return {"grouping_mode": "EACH_FILE", "items": []}

    async def delete_operations_knowledge_asset(self, asset_id, **kwargs):
        self.deleted = {"asset_id": asset_id, **kwargs}

    async def update_operations_knowledge_base(self, knowledge_base_id, payload, **kwargs):
        return payload


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
    async def test_management_search_requires_user_and_knowledge_manage(self):
        context = _context().model_copy(update={"app_id": None})
        client = SimpleNamespace(search_managed_operations_knowledge=AsyncMock(return_value={}))
        request = _request(client)
        request.state.auth_context = context
        response = Response()
        payload = KnowledgeManagementSearchRequest(query="等待", source_kind="MANUAL")
        with patch("main_api.api.aiops_app._require", AsyncMock(side_effect=HTTPException(403))) as authorize:
            with self.assertRaises(HTTPException):
                await search_managed_operations_knowledge(payload, request, response)
            authorize.assert_awaited_once_with(request, "aiops:knowledge_manage")
        client.search_managed_operations_knowledge.assert_not_awaited()
        with patch("main_api.api.aiops_app._require", AsyncMock()):
            await search_managed_operations_knowledge(payload, request, response)
        self.assertEqual("no-store", response.headers["cache-control"])
        forwarded = client.search_managed_operations_knowledge.await_args
        self.assertEqual({"query": "等待", "source_kind": "MANUAL", "knowledge_base_id": None}, forwarded.args[0])
        self.assertEqual(context.domain_id, forwarded.kwargs["auth_context"].domain_id)
        self.assertEqual("aiops", forwarded.kwargs["auth_context"].app_id)
        self.assertEqual(("aiops:knowledge_manage",), forwarded.kwargs["auth_context"].scopes)
        request.state.auth_context = context.model_copy(update={"principal_kind": PrincipalKind.APP_API_CLIENT})
        with self.assertRaises(HTTPException) as caught:
            await search_managed_operations_knowledge(payload, request, Response())
        self.assertEqual(403, caught.exception.status_code)
        self.assertEqual(1, client.search_managed_operations_knowledge.await_count)

    async def test_knowledge_uses_unified_agent_conversation_not_fixed_assistant(self):
        paths = {route.path for route in router.routes}
        self.assertTrue(any(path.endswith("/conversations/{conversation_id}/turns") for path in paths))
        for path in paths:
            if "/operations-knowledge/conversations" in path:
                self.assertTrue(path.endswith(("/case-candidate", ":close-diagnosis")))

    async def test_overview_uses_aiops_registry_client(self):
        client = _AIOpsClient()
        with patch(
            "main_api.api.aiops_app._require",
            AsyncMock(return_value=(42, "aiopsadmin", object())),
        ):
            result = await operations_knowledge_overview(_request(client))
        self.assertEqual(2, result["published_manuals"])

    async def test_update_preserves_optional_label_model_field(self):
        client = _AIOpsClient()
        model_id = uuid7()
        for field in ({}, {"tagging_model_id": None}, {"tagging_model_id": str(model_id)}):
            payload = KnowledgeBaseWrite(display_name="手册", **field)
            with patch("main_api.api.aiops_app._require", AsyncMock(return_value=(42, "aiopsadmin", object()))):
                result = await update_operations_knowledge_base(uuid7(), payload, _request(client))
            self.assertEqual({"display_name": "手册", **field}, result)

    async def test_delete_asset_forwards_optimistic_row_version(self):
        client = _AIOpsClient()
        asset_id = uuid7()
        with patch(
            "main_api.api.aiops_app._require",
            AsyncMock(return_value=(42, "aiopsadmin", object())),
        ):
            response = await delete_operations_knowledge_asset(
                asset_id, _request(client), if_match='"rv-7"',
                purge_source_objects=True,
            )

        self.assertEqual(204, response.status_code)
        self.assertEqual(asset_id, client.deleted["asset_id"])
        self.assertEqual(7, client.deleted["expected_row_version"])
        self.assertTrue(client.deleted["purge_source_objects"])

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

        self.assertEqual("EACH_FILE", result["grouping_mode"])
        self.assertEqual("multipart/form-data; boundary=test", client.uploaded["content_type"])
        self.assertEqual("aiops-upload-test", client.uploaded["idempotency_key"])
        self.assertEqual(
            b"multipart-body",
            b"".join([chunk async for chunk in client.uploaded["body"]]),
        )


if __name__ == "__main__":
    unittest.main()
