"""会话案例真实异步受理与确切结果版本；不调用模型或 KC。"""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from aiops_agent.api import operations_knowledge as knowledge_api
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.application.output_jobs import OutputJobScope, OutputJobService
from platform_clients.aiops import AIOpsManagementClient
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_conversation_case_closure import setup, preview
from tests.unit.aiops_agent.test_aiops_output_job_api import api, human_context
from tests.unit.aiops_agent.test_aiops_output_jobs import Store, MemoryUow


async def fixture():
    f = await setup()
    store = Store()
    if store.tenant is None:
        f.domain, f.actor = store.domain, str(f.actor)
        for run in f.state.runs.values():
            run.domain_id, run.actor_id = f.domain, f.actor
    store.domain, store.actor = f.domain, f.actor
    store.scope = OutputJobScope(domain_id=f.domain, actor_id=f.actor,
        tenant_id=store.tenant, trace_id="trace")
    store.root.created_by = f.actor
    store.source = store.source.model_copy(update={"source_id": f.conversation_id})
    def uow():
        value = MemoryUow(store)
        for name in ("runs", "turns", "targets", "agents", "conversations", "operations_knowledge"):
            setattr(value, name, getattr(f.uow, name))
        return value
    store.service = OutputJobService(uow_factory=uow)
    app = api(store)
    app.include_router(knowledge_api.router)
    app.state.operations_knowledge_service = f.service
    candidate = await preview(f, (f.turn_id,))
    payload = dict(included_turn_ids=candidate["included_turn_ids"],
        source_fingerprint=candidate["source_fingerprint"], display_name="冻结的案例",
        symptoms=["短时业务等待"], root_cause_summary="现有证据的人工归纳")
    return f, store, app, payload


def test_http_receipt_is_persisted_before_model_or_kc_and_duplicate_is_idempotent():
    async def run():
        f, store, app, payload = await fixture()
        path = "/internal/v1/aiops/operations-knowledge/conversations/" + str(f.conversation_id) + ":close-diagnosis"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            response = await client.post(path, json=payload)
            assert response.status_code == 428 and not store.jobs
            headers = {"Idempotency-Key": "case-click-1"}
            first = await client.post(path, json=payload, headers=headers)
            assert first.status_code == 202, first.text
            receipt = first.json()
            assert receipt["status"] == "QUEUED" and receipt["result"] is None
            assert receipt["job_type"] == "CASE_EXTRACTION"
            assert UUID(receipt["job_id"]) in store.jobs
            assert (await client.post(path, json=payload, headers=headers)).json()["job_id"] == receipt["job_id"]
            assert len(store.jobs) == len(store.events) == len(store.notifications) == 1
            # 同键也必须重新验证原件，不能返回已失效的成功受理。
            f.artifact.content_hash = "f" * 64
            assert (await client.post(path, json=payload, headers=headers)).status_code == 422
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.service._documents.stage.assert_not_awaited()
        f.service._documents.publish.assert_not_awaited()
    asyncio.run(run())


def test_public_submission_requires_and_forwards_idempotency_without_result_redirect():
    async def run():
        f, store, _, payload = await fixture()
        module = importlib.import_module("main_api.api." + ("ops" if store.tenant else "aiops_app"))
        client = SimpleNamespace(close_conversation_diagnosis=AsyncMock(return_value={"job_id": "queued"}))
        request = SimpleNamespace(headers={}, state=SimpleNamespace(auth_context=human_context(store)),
            app=SimpleNamespace(state=SimpleNamespace(aiops_client=client)))
        command = module.DiagnosisClosurePayload.model_validate(payload)
        with patch.object(module, "_conversation_with_access", new=AsyncMock()) as access:
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as caught:
                await module.close_conversation_diagnosis(f.conversation_id, command, request)
            assert caught.value.status_code == 428
            client.close_conversation_diagnosis.assert_not_awaited()
            request.headers["Idempotency-Key"] = "case-click-1"
            result = await module.close_conversation_diagnosis(f.conversation_id, command, request)
            assert result == {"job_id": "queued"} and access.await_count == 2
        assert client.close_conversation_diagnosis.await_args.kwargs["idempotency_key"] == "case-click-1"
        assert client.close_conversation_diagnosis.await_args.kwargs["payload"] == command.model_dump(mode="json")
        narrow = object.__new__(AIOpsManagementClient)
        narrow._json = AsyncMock(return_value={})
        await narrow.close_conversation_diagnosis(f.conversation_id, payload=payload,
            idempotency_key="case-click-1", auth_context=request.state.auth_context)
        assert narrow._json.await_args.kwargs["idempotency_key"] == "case-click-1"
    asyncio.run(run())


def test_case_read_selects_exact_version_and_rejects_different_case_without_latest_fallback():
    async def run():
        service = object.__new__(OperationsKnowledgeService)
        domain, asset, latest, requested = (uuid7() for _ in range(4))
        service.get_asset = AsyncMock(return_value={"asset": {"asset_kind": "DIAGNOSIS_CASE"},
            "versions": [{"asset_version_id": str(latest)}]})
        service.get_version = AsyncMock(return_value={"asset": {"asset_id": str(asset)},
            "version": {"asset_version_id": str(requested)}})
        detail = await service.get_case(domain_id=domain, asset_id=asset, asset_version_id=requested)
        assert detail["version"]["asset_version_id"] == str(requested)
        assert service.get_version.await_args.kwargs["asset_version_id"] == requested
        await service.get_case(domain_id=domain, asset_id=asset)
        assert service.get_version.await_args.kwargs["asset_version_id"] == latest
        service.get_version.return_value["asset"]["asset_id"] = str(uuid7())
        with pytest.raises(AIOpsApplicationError):
            await service.get_case(domain_id=domain, asset_id=asset, asset_version_id=requested)
        assert service.get_version.await_args.kwargs["asset_version_id"] == requested
        narrow = object.__new__(AIOpsManagementClient)
        narrow._json = AsyncMock(return_value={})
        await narrow.get_operations_diagnosis_case(asset, asset_version_id=requested, auth_context=object())
        assert narrow._json.await_args.args[1].endswith("?asset_version_id=" + str(requested))
    asyncio.run(run())
