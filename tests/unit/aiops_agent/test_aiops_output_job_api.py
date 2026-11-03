"""输出任务内部 HTTP、公共 BFF 与窄客户端合同；使用可信上下文替身。"""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from aiops_agent.api.output_jobs import router
from aiops_agent.api.dependencies import get_aiops_auth_context
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.output_job_sources import OutputJobSourceValidator
from aiops_agent.application.output_jobs import job_receipt
from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts import AuthContext, PrincipalKind, ServiceIdentity
from platform_core.contracts.aiops.output_jobs import OutputJobRetry
from platform_core.identity import uuid7
from platform_core.security import create_service_auth_context
from tests.unit.aiops_agent.test_aiops_output_jobs import Store, NOW


def human_context(store):
    context = create_service_auth_context(caller_service="main-api", trace_id="trace")
    changes = dict(principal_kind=PrincipalKind.PORTAL, domain_id=str(store.domain))
    if store.tenant is not None:
        changes.update(principal_id=store.actor, domain_id=store.domain,
            tenant_id=str(store.tenant), tenant_ids=(store.tenant,), app_id="aiops")
    else:
        changes.update(api_key_id="portal-key-reference", asserted_user_id=store.actor)
    return AuthContext.model_validate({**context.model_dump(), **changes})


def api(store, *, context=None, scopes=("aiops.run",)):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_aiops_auth_context] = lambda: context or human_context(store)
    app.state.output_job_service = store.service
    app.state.output_job_source_validator = OutputJobSourceValidator()
    @app.middleware("http")
    async def identity(request, call_next):
        request.state.service_identity = ServiceIdentity(issuer="test", subject="main-api",
            audience="aiops", scopes=scopes, issued_at=NOW,
            expires_at=NOW + timedelta(minutes=1), token_id=uuid7())
        return await call_next(request)
    @app.exception_handler(AIOpsApplicationError)
    async def error(_request, exc):
        return JSONResponse(status_code=exc.status_code, content={"code": exc.code})
    return app


def test_internal_list_and_detail_are_actor_scoped_and_never_expose_frozen_input():
    async def run():
        store = Store()
        receipt = await store.submit()
        app = api(store)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            response = await client.get("/internal/v1/aiops/output-jobs")
            assert response.status_code == 200
            assert response.json()["items"][0]["job_id"] == str(receipt.job_id)
            for key in ("input_snapshot_json", "lease_token", "processing_ref_json", "authority"):
                assert key not in str(response.json())
            response = await client.get("/internal/v1/aiops/output-jobs/" + str(receipt.job_id))
            assert response.status_code == 200
            context = human_context(store)
            other = {"principal_id": uuid7()} if store.tenant else {"asserted_user_id": "another"}
            app.dependency_overrides[get_aiops_auth_context] = lambda: context.model_copy(update=other)
            response = await client.get("/internal/v1/aiops/output-jobs/" + str(receipt.job_id))
            assert response.status_code == 404
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["service", "scope", "limit", "type"])
def test_internal_http_rejects_untrusted_actor_scope_and_invalid_filters(mode):
    async def run():
        store = Store()
        context = human_context(store)
        if mode == "service":
            context = create_service_auth_context(caller_service="main-api", trace_id="trace")
        app = api(store, context=context, scopes=("aiops.manage",) if mode == "scope" else ("aiops.run",))
        suffix = "?limit=101" if mode == "limit" else "?job_type=OTHER" if mode == "type" else ""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            response = await client.get("/internal/v1/aiops/output-jobs" + suffix)
            assert response.status_code == (422 if mode in {"limit", "type"} else 403)
        assert not store.jobs
    asyncio.run(run())


def test_internal_retry_uses_optimistic_version_and_revalidates_before_requeue():
    async def run():
        store = Store()
        receipt = await store.submit()
        job = store.jobs[receipt.job_id]
        job.status, job.stage, job.retryable = "FAILED", "FAILED", True
        job.completed_at, job.error_code = NOW, "TEMPORARY"
        app = api(store)
        authorize = AsyncMock()
        app.state.output_job_source_validator.retry = authorize
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            path = "/internal/v1/aiops/output-jobs/" + str(receipt.job_id) + ":retry"
            response = await client.post(path, json={"expected_row_version": 9})
            assert response.status_code == 409 and store.jobs[receipt.job_id].status == "FAILED"
            response = await client.post(path, json={"expected_row_version": 1})
            assert response.status_code == 202 and response.json()["status"] == "QUEUED"
            assert authorize.await_count == 2
            assert len(store.events) == len(store.notifications) == 2
    asyncio.run(run())


def test_management_client_forwards_only_supported_filters_and_retry_contract():
    async def run():
        client = object.__new__(AIOpsManagementClient)
        client._json = AsyncMock(return_value={})
        context = object()
        conversation_id, before_id, job_id = uuid7(), uuid7(), uuid7()
        await client.list_output_jobs(auth_context=context, limit=12,
            job_type="CASE_EXTRACTION", status="FAILED",
            conversation_id=conversation_id, before_id=before_id)
        args = client._json.await_args
        query = parse_qs(urlsplit(args.args[1]).query)
        assert query == {"limit": ["12"], "job_type": ["CASE_EXTRACTION"], "status": ["FAILED"],
            "conversation_id": [str(conversation_id)], "before_id": [str(before_id)]}
        await client.retry_output_job(job_id, payload={"expected_row_version": 4}, auth_context=context)
        assert client._json.await_args.kwargs["payload"] == {"expected_row_version": 4}
        assert client._json.await_args.kwargs["auth_context"] is context
        await client.get_output_job(job_id, auth_context=context)
        assert client._json.await_args.args == ("GET", "/internal/v1/aiops/output-jobs/" + str(job_id))
    asyncio.run(run())


def test_public_retry_derives_required_permission_from_authoritative_job_not_browser():
    async def run():
        store = Store()
        receipt = await store.submit()
        job = store.jobs[receipt.job_id]
        job.job_type = "CASE_EXTRACTION"
        job.source_json["kind"] = "REPORT"
        result = job_receipt(job).model_dump(mode="json")
        client = SimpleNamespace(get_output_job=AsyncMock(return_value=result),
            retry_output_job=AsyncMock(return_value=result))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(aiops_client=client)),
            state=SimpleNamespace(auth_context=human_context(store)))
        if store.tenant is not None:
            from main_api.api import ops as public
            permission = "require_cached_app_permission"
        else:
            from main_api.api import aiops_app as public
            permission = "_require"
        denied = AIOpsApplicationError(code="DENIED", message="当前无权限", status_code=403)
        if store.tenant is not None:
            guard = patch.object(public, permission, side_effect=denied)
        else:
            guard = patch.object(public, permission,
                new=AsyncMock(side_effect=[None, denied]))
        with guard as checked:
            with pytest.raises(AIOpsApplicationError):
                await public.retry_output_job(job.job_id, OutputJobRetry(expected_row_version=1), request)
            if store.tenant is not None:
                assert checked.call_args.kwargs["permissions"] == "aiops:knowledge_manage"
            else:
                assert checked.await_args.args == (request, "aiops:knowledge_manage")
        client.retry_output_job.assert_not_awaited()
    asyncio.run(run())
