"""管理预览使用本 App 精确文档，不授予 Agent 搜索权限。"""

import asyncio
from copy import deepcopy
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from aiops_agent.api import operations_knowledge as internal
from aiops_agent.application.knowledge_preview import preview_knowledge_version
from aiops_agent.application.case_knowledge_projection import case_knowledge_markdown
from aiops_agent.contracts.knowledge import KnowledgeSearchPreviewRequest
from aiops_agent.entities import ReportSourceEntity
from knowledge_algorithms.contracts.errors import AlgorithmError
from platform_clients.aiops import AIOpsManagementClient
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_case_knowledge_projection import profile


def fixture():
    value = profile()
    case_knowledge_markdown(title="已脱敏案例", profile=value)
    version_id, document_id = uuid7(), uuid7()
    index = {"document_version_id": str(document_id), "parse_run_id": str(uuid7()),
        "index_generation_id": str(uuid7()), "status": "READY"}
    detail = {"asset": {"asset_id": str(uuid7()), "asset_kind": "DIAGNOSIS_CASE", "row_version": 1},
        "version": {"asset_version_id": str(version_id), "status": "DRAFT", "row_version": 1,
            "profile": value.model_dump(mode="json")}, "indexes": [index]}
    record = {"parse_run_id": index["parse_run_id"], "generation_id": index["index_generation_id"],
        "source_hash":value.knowledge_projection.body_sha256, "status":"PARSED"}
    documents = SimpleNamespace(access=lambda context: context, inspect=AsyncMock(return_value=record),
        excerpts=AsyncMock(return_value={"excerpts":[{"content":"合成解析正文", "locator":{"key":"synthetic"}}],
            "warnings":[]}), retrieval=SimpleNamespace(retrieve=AsyncMock()))
    service = SimpleNamespace(get_version=AsyncMock(side_effect=lambda **_: deepcopy(detail)), _documents=documents)
    kwargs = dict(domain_id=uuid7(), asset_version_id=version_id, context=SimpleNamespace(), kc_scope={})
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("status", ["DRAFT", "REVIEW_REQUIRED", "PUBLISHED"])
def test_manager_previews_exact_indexed_version_without_agent_or_search(status):
    async def run():
        f = fixture()
        f.detail["version"]["status"] = status
        result = await preview_knowledge_version(f.service, **f.kwargs)
        assert result["usage"] == "MANAGEMENT_PREVIEW_NOT_AGENT_EVIDENCE"
        assert result["status"] == status
        assert result["documents"][0]["document_version_id"] == str(f.document_id)
        assert result["documents"][0]["fragments"][0]["content_text"] == "合成解析正文"
        f.documents.retrieval.retrieve.assert_not_awaited()
        assert f.service.get_version.await_count == 2
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["processing", "withdrawn", "wrong_parse", "wrong_hash",
    "wrong_generation", "empty_indexes", "edited_during_preview"])
def test_manager_preview_rejects_unready_withdrawn_or_changed_exact_source(failure):
    async def run():
        f = fixture()
        if failure == "processing":
            f.detail["version"]["status"] = "PROCESSING"
        elif failure == "withdrawn":
            f.record["status"] = "WITHDRAWN"
        elif failure == "wrong_parse":
            f.record["parse_run_id"] = str(uuid7())
        elif failure == "wrong_hash":
            f.record["source_hash"] = "f"*64
        elif failure == "wrong_generation":
            f.record["generation_id"] = str(uuid7())
        elif failure == "empty_indexes":
            f.detail["indexes"] = []
        else:
            async def changed(*_):
                f.detail["version"]["row_version"] += 1
                return {"excerpts":[], "warnings":[]}
            f.documents.excerpts.side_effect = changed
        with pytest.raises(AlgorithmError):
            await preview_knowledge_version(f.service, **f.kwargs)
        f.documents.retrieval.retrieve.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("has_agent", [False, True])
def test_search_preview_requires_selected_agent_across_public_client_internal_boundaries(monkeypatch, has_agent):
    async def run():
        oracle = ReportSourceEntity.__tablename__.startswith("KBOT_")
        public = importlib.import_module("main_api.api.aiops_app" if oracle else "main_api.api.ops")
        context, agent_id, target_id = SimpleNamespace(), uuid7(), uuid7()
        service = SimpleNamespace(search=AsyncMock(return_value={"status": "READY", "results": []}))
        internal_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(operations_knowledge_service=service)))
        monkeypatch.setattr(internal, "_scope", lambda *_: (1, "manager") if oracle else (uuid7(), uuid7(), uuid7()))
        client = AIOpsManagementClient.__new__(AIOpsManagementClient)
        async def dispatch(method, path, *, payload, auth_context):
            assert method == "POST" and path.endswith("operations-knowledge:search")
            return await internal.search(KnowledgeSearchPreviewRequest.model_validate(payload),
                internal_request, context=auth_context)
        client._json = AsyncMock(side_effect=dispatch)
        monkeypatch.setattr(public, "_client", lambda _: client)
        if oracle:
            monkeypatch.setattr(public, "_require", AsyncMock())
        app = FastAPI()
        app.add_api_route("/search-preview", public.preview_operations_knowledge_search, methods=["POST"])
        @app.middleware("http")
        async def set_context(req, call_next):
            req.state.auth_context = context
            return await call_next(req)
        payload = {"query": "如何排查等待", "target_id": str(target_id)}
        if has_agent:
            payload["agent_id"] = str(agent_id)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            response = await http.post("/search-preview", json=payload)
        if not has_agent:
            assert response.status_code == 422
            service.search.assert_not_awaited()
        else:
            assert response.status_code == 200
            args = service.search.await_args.kwargs
            assert args["agent_id"] == str(agent_id) and args["target_id"] == target_id
            assert args["request"].query == payload["query"]
            assert "agent_id" not in type(args["request"]).model_fields
    asyncio.run(run())
