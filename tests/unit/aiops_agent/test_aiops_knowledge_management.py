"""管理预览与 Agent 正式检索明确分开，预览不伪造 Agent，也不读取任意外部版本。"""

import asyncio
from copy import deepcopy
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from aiops_agent.api import operations_knowledge as internal
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.knowledge_preview import preview_knowledge_version
from aiops_agent.application.case_knowledge_projection import case_knowledge_markdown
from aiops_agent.contracts.knowledge import KnowledgeSearchPreviewRequest
from aiops_agent.entities import ReportSourceEntity
from platform_clients.aiops import AIOpsManagementClient
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_case_knowledge_projection import profile


def test_management_preview_uses_actual_registry_version_projection():
    import inspect
    from tests.unit.aiops_agent.test_aiops_report_case_extraction import case_fixture, extract
    async def run():
        f = await case_fixture()
        await extract(f)
        version = next(iter(f.state.versions.values()))
        version.status = "DRAFT"
        f.state.relations[version.asset_version_id].update(scopes=[], reviews=[])
        index = f.state.relations[version.asset_version_id]["indexes"][0]
        index.index_status = "READY"
        detail = await f.service.get_version(domain_id=f.domain, asset_version_id=version.asset_version_id)
        document_id = uuid7()
        f.service._kc.get_revision_status = AsyncMock(return_value={
            "bundle_revision_id": str(index.bundle_revision_id), "status": "READY",
            "approval_status": "APPROVED", "withdrawn_at": None, "members": [{
                "document_role": "CONTENT", "member_status": "READY", "document_version_id": str(document_id),
                "content_hash_sha256": version.profile_json["knowledge_projection"]["body_sha256"],
            }],
        })
        f.service._kc.get_document_preview = AsyncMock(return_value={
            "document_version_id": str(document_id), "fragments": [{"content": "实际登记版本的解析正文"}],
        })
        kwargs = dict(domain_id=f.domain, asset_version_id=version.asset_version_id, context=f.context)
        if "tenant_id" in inspect.signature(f.service.preview_version).parameters:
            kwargs["tenant_id"] = version.tenant_id
        result = await f.service.preview_version(**kwargs)
        assert result["documents"][0]["fragments"][0]["content"] == "实际登记版本的解析正文"
        assert result["bundle_revision_id"] == detail["indexes"][0]["bundle_revision_id"]
        assert version.status == "DRAFT"
    asyncio.run(run())


def test_kc_status_keeps_exact_withdrawal_instead_of_reporting_ready_only():
    from datetime import UTC, datetime
    from knowledge_core.application.status import KnowledgeCoreStatusService
    withdrawn_at = datetime.now(UTC)
    entity = SimpleNamespace(bundle_revision_id=uuid7(), revision_no=1, source_revision="source-1",
        status="READY", approval_status="APPROVED", reviewed_by=None, reviewed_at=None,
        review_comment=None, accepted_at=None, completed_at=None, withdrawn_at=withdrawn_at)
    assert KnowledgeCoreStatusService._revision(entity).withdrawn_at == withdrawn_at


def fixture():
    value = profile()
    case_knowledge_markdown(title="已脱敏案例", profile=value)
    version_id, document_id = uuid7(), uuid7()
    index = {"collection_id": str(uuid7()), "bundle_id": str(uuid7()),
        "bundle_revision_id": str(uuid7()), "status": "READY"}
    detail = {"asset": {"asset_id": str(uuid7()), "asset_kind": "DIAGNOSIS_CASE", "row_version": 1},
        "version": {"asset_version_id": str(version_id), "status": "DRAFT", "row_version": 1,
            "profile": value.model_dump(mode="json")}, "indexes": [index]}
    revision = {"bundle_revision_id": index["bundle_revision_id"], "status": "READY",
        "approval_status": "APPROVED", "withdrawn_at": None, "members": [{
            "document_version_id": str(document_id), "document_role": "CONTENT",
            "member_status": "READY", "content_hash_sha256": value.knowledge_projection.body_sha256,
        }]}
    kc = SimpleNamespace(get_revision_status=AsyncMock(return_value=revision),
        get_document_preview=AsyncMock(return_value={"document_version_id": str(document_id),
            "fragments": [{"content": "真实解析正文", "locator": {"heading_path": ["问题"]}}]}),
        discover=AsyncMock(), retrieve_evidence=AsyncMock())
    service = SimpleNamespace(get_version=AsyncMock(side_effect=lambda **_: deepcopy(detail)), _kc=kc)
    kwargs = dict(domain_id=uuid7(), asset_version_id=version_id, context=SimpleNamespace(), kc_scope={})
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("status", ["DRAFT", "REVIEW_REQUIRED", "PUBLISHED"])
def test_manager_can_preview_exact_indexed_version_without_agent_search(status):
    async def run():
        f = fixture()
        f.detail["version"]["status"] = status
        result = await preview_knowledge_version(f.service, **f.kwargs)
        assert result["usage"] == "MANAGEMENT_PREVIEW_NOT_AGENT_EVIDENCE"
        assert result["status"] == status and result["documents"][0]["fragments"]
        assert result["bundle_revision_id"] == f.index["bundle_revision_id"]
        f.kc.discover.assert_not_awaited()
        f.kc.retrieve_evidence.assert_not_awaited()
        assert f.service.get_version.await_count == 2
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["processing", "withdrawn", "wrong_revision", "wrong_hash",
    "wrong_role", "wrong_document", "edited_during_preview"])
def test_manager_preview_never_substitutes_wrong_or_changed_source(failure):
    async def run():
        f = fixture()
        if failure == "processing":
            f.detail["version"]["status"] = "PROCESSING"
        elif failure == "withdrawn":
            f.revision["withdrawn_at"] = "2026-11-03T00:00:00+00:00"
        elif failure == "wrong_revision":
            f.revision["bundle_revision_id"] = str(uuid7())
        elif failure == "wrong_hash":
            f.revision["members"][0]["content_hash_sha256"] = "f" * 64
        elif failure == "wrong_role":
            f.revision["members"][0]["document_role"] = "MANIFEST"
        elif failure == "wrong_document":
            f.kc.get_document_preview.return_value["document_version_id"] = str(uuid7())
        else:
            async def changed(**_):
                f.detail["version"]["row_version"] += 1
                return {"document_version_id": str(f.document_id), "fragments": []}
            f.kc.get_document_preview.side_effect = changed
        with pytest.raises(AIOpsApplicationError) as error:
            await preview_knowledge_version(f.service, **f.kwargs)
        assert error.value.code == "AIOPS_KNOWLEDGE_PREVIEW_UNAVAILABLE"
        f.kc.discover.assert_not_awaited()
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
