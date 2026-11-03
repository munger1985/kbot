"""后台核权的当前状态与内部调用边界；不使用真实平台数据库。"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.output_authorization import OutputJobAuthority
from aiops_agent.application.output_jobs import OutputJobScope
from aiops_agent.entities import OpsOutputJobEntity
from aiops_agent.workers.output_jobs import OutputExecution
from main_api.api.output_authority import OUTPUT_AUTHORITY_PATH
from main_api.application.output_authority import OutputAuthorityDenied, OutputAuthorityService
from platform_clients.output_authority import OutputAuthorityClient, OutputAuthorityClientError
from platform_core.contracts import PrincipalKind
from platform_core.contracts.aiops.output_authority import OutputAuthorityRequest, OutputAuthoritySnapshot
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput, OutputJobSource, OutputSourceArtifact
from platform_core.identity import uuid7
from platform_core.security import (
    AUTH_CONTEXT_HEADER, SERVICE_IDENTITY_HEADER,
    build_scoped_internal_auth_headers, create_service_auth_context,
)


ORACLE = OpsOutputJobEntity.__tablename__.startswith("KBOT_")
CALLER = "kbot-aiops-worker" if ORACLE else "ammolite-aiops-worker"
AUDIENCE = "kbot-main-api" if ORACLE else "ammolite-main-api"


def authority_request(*, kind="CONVERSATION", job_type="CASE_EXTRACTION"):
    return OutputAuthorityRequest(job_id=uuid7(), job_type=job_type,
        source=OutputJobSource(kind=kind, source_id=uuid7()), origin="HUMAN",
        actor_id="operator" if ORACLE else str(uuid7()), domain_id="7" if ORACLE else str(uuid7()),
        tenant_id=None if ORACLE else uuid7(), source_agent_ids=(uuid7(),))


class Uow:
    def __init__(self):
        self.authentication = SimpleNamespace(background_actor_is_active=AsyncMock(return_value=True))
        self.app_domains = SimpleNamespace(output_job_binding=AsyncMock(return_value={"code": "ops-main"}),
            list_user_app_permissions=AsyncMock(return_value=frozenset({"aiops:use", "aiops:knowledge_manage"})))
        self.access = SimpleNamespace(get_user=AsyncMock(return_value=SimpleNamespace(status="ACTIVE", max_security_level=2)),
            permissions_for=AsyncMock(return_value={"aiops:use", "aiops:knowledge_manage"}),
            list_roles=AsyncMock(return_value=()))
        self.domains = SimpleNamespace(exists_active=AsyncMock(return_value=True))
        self.entries = 0

    async def __aenter__(self):
        self.entries += 1
        return self

    async def __aexit__(self, *_):
        pass


def test_background_authority_reads_current_permissions_each_time_without_user_session():
    async def run():
        request, uow = authority_request(kind="REPORT"), Uow()
        service = OutputAuthorityService(uow_factory=lambda: uow)
        snapshot = await service.resolve(request)
        assert snapshot.request == request
        assert set(snapshot.permissions) == {"aiops:use", "aiops:knowledge_manage"}
        assert snapshot.max_security_level == (2 if ORACLE else None)
        if ORACLE:
            uow.access.permissions_for.return_value = {"aiops:use"}
        else:
            uow.app_domains.list_user_app_permissions.return_value = {"aiops:use"}
        with pytest.raises(OutputAuthorityDenied):
            await service.resolve(request)
        assert uow.entries == 2
        assert not {"session_id", "auth_context", "jwt", "token"} & snapshot.model_dump().keys()
    asyncio.run(run())


def test_conversation_case_permission_is_not_silently_upgraded_to_knowledge_management():
    async def run():
        uow = Uow()
        uow.access.permissions_for.return_value = {"aiops:use"}
        uow.app_domains.list_user_app_permissions.return_value = {"aiops:use"}
        request = authority_request()
        assert (await OutputAuthorityService(uow_factory=lambda: uow).resolve(request)).request == request
    asyncio.run(run())


@pytest.mark.parametrize("revoked", ["user", "space", "permission", "platform_scope"])
def test_revoked_account_domain_membership_or_permission_is_denied(revoked):
    async def run():
        request, uow = authority_request(), Uow()
        if revoked == "user":
            uow.authentication.background_actor_is_active.return_value = False
            uow.access.get_user.return_value = SimpleNamespace(status="DISABLED", max_security_level=2)
        elif revoked == "space":
            uow.app_domains.output_job_binding.return_value = None
            uow.domains.exists_active.return_value = False
        elif revoked == "permission":
            uow.app_domains.list_user_app_permissions.return_value = set()
            uow.access.permissions_for.return_value = set()
        elif ORACLE:
            request = request.model_copy(update={"tenant_id": uuid7()})
        else:
            request = request.model_copy(update={"tenant_id": None})
        with pytest.raises(OutputAuthorityDenied):
            await OutputAuthorityService(uow_factory=lambda: uow).resolve(request)
    asyncio.run(run())


@pytest.mark.parametrize("bad_domain", ["", "foreign", "-1", "007"])
def test_platform_domain_format_is_validated_before_opening_transaction(bad_domain):
    async def run():
        uow = Uow()
        request = authority_request().model_copy(update={"domain_id": bad_domain})
        with pytest.raises(OutputAuthorityDenied):
            await OutputAuthorityService(uow_factory=lambda: uow).resolve(request)
        assert uow.entries == 0
    asyncio.run(run())


@pytest.mark.parametrize("credential", ["valid", "missing", "context_only", "identity_only",
    "wrong_caller", "wrong_scope", "wrong_audience", "not_service"])
def test_real_main_api_internal_boundary_requires_both_jwts_caller_scope_and_service_kind(credential):
    async def run():
        if ORACLE:
            from main_api.app import create_main_api_app
            app = create_main_api_app(enable_access_log=False)
        else:
            from main_api.app import create_ammolite_main_api_app
            app = create_ammolite_main_api_app(enable_access_log=False)
        request = authority_request()
        snapshot = OutputAuthoritySnapshot(request=request, permissions=("aiops:use",), checked_at=datetime.now(timezone.utc))
        service = SimpleNamespace(resolve=AsyncMock(return_value=snapshot))
        app.state.output_authority_service = service
        caller = "untrusted-worker" if credential == "wrong_caller" else CALLER
        context = create_service_auth_context(caller_service=caller, trace_id="trace")
        if credential == "not_service":
            context = context.model_copy(update={"principal_kind": PrincipalKind.API_CLIENT, "api_key_id": "test-key-id"})
        headers = {} if credential == "missing" else build_scoped_internal_auth_headers(
            caller_service=caller, audience="wrong-main-api" if credential == "wrong_audience" else AUDIENCE,
            scopes=("aiops.output.invalid" if credential == "wrong_scope" else "aiops.output.authorize",), context=context,
        )
        if credential == "context_only":
            headers.pop(SERVICE_IDENTITY_HEADER)
        elif credential == "identity_only":
            headers.pop(AUTH_CONTEXT_HEADER)
        assert OUTPUT_AUTHORITY_PATH not in app.openapi()["paths"]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(OUTPUT_AUTHORITY_PATH, headers=headers, json=request.model_dump(mode="json"))
        assert response.status_code == (200 if credential == "valid" else 403), response.text
        assert service.resolve.await_count == (1 if credential == "valid" else 0)
    asyncio.run(run())


def execution_fixture():
    request = authority_request(job_type="REPORT_GENERATION")
    frozen = FrozenOutputInput(source=request.source, source_fingerprint="a" * 64,
        artifacts=(OutputSourceArtifact(run_id=uuid7(), artifact_id=uuid7(), schema_version="AIOPS_TURN_RESULT.v1",
            content_hash="b" * 64, security_level=2),),
        parameters={"authority": {"origin": "HUMAN", "source_agent_ids": list(map(str, request.source_agent_ids))}},
        frozen_at=datetime.now(timezone.utc))
    scope = OutputJobScope(domain_id=request.domain_id, actor_id=request.actor_id, trace_id="trace", tenant_id=request.tenant_id)
    return OutputExecution(request.job_id, request.job_type, scope, {}, uuid7(), "VALIDATING", frozen), request


@pytest.mark.parametrize("change", ["fresh", "stale", "future", "wrong_actor", "permissions", "security"])
def test_worker_rejects_stale_foreign_or_narrowed_current_authority(change):
    async def run():
        execution, request = execution_fixture()
        fields = dict(request=request, permissions=("aiops:use",), max_security_level=2, checked_at=datetime.now(timezone.utc))
        if change == "stale":
            fields["checked_at"] -= timedelta(minutes=2)
        elif change == "future":
            fields["checked_at"] += timedelta(minutes=2)
        elif change == "wrong_actor":
            fields["request"] = request.model_copy(update={"actor_id": "foreign-actor"})
        elif change == "permissions":
            fields["permissions"] = ()
        elif change == "security":
            fields["max_security_level"] = 1
        client = SimpleNamespace(resolve=AsyncMock(return_value=OutputAuthoritySnapshot(**fields)))
        authority = OutputJobAuthority(client=client)
        if change == "fresh":
            assert (await authority.resolve(execution)).request == request
        else:
            with pytest.raises(AIOpsApplicationError) as raised:
                await authority.resolve(execution)
            assert not raised.value.retryable
    asyncio.run(run())


@pytest.mark.parametrize("status,retryable", [(403, False), (401, False), (503, True), (429, True)])
def test_authority_outage_and_revocation_have_distinct_retry_semantics(status, retryable):
    async def run():
        execution, _ = execution_fixture()
        client = SimpleNamespace(resolve=AsyncMock(side_effect=OutputAuthorityClientError(status_code=status, retryable=retryable)))
        with pytest.raises(AIOpsApplicationError) as raised:
            await OutputJobAuthority(client=client).resolve(execution)
        assert raised.value.retryable == retryable
    asyncio.run(run())


def test_missing_server_frozen_origin_cannot_gain_background_human_privileges():
    async def run():
        execution, _ = execution_fixture()
        execution.frozen.parameters.clear()
        client = SimpleNamespace(resolve=AsyncMock())
        with pytest.raises(AIOpsApplicationError) as raised:
            await OutputJobAuthority(client=client).resolve(execution)
        assert not raised.value.retryable
        client.resolve.assert_not_awaited()
    asyncio.run(run())


def test_narrow_client_creates_new_auth_each_call_and_never_saves_it_in_business_input():
    class Response:
        status = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            pass
        async def json(self):
            return OutputAuthoritySnapshot(request=request, permissions=("aiops:use",),
                checked_at=datetime.now(timezone.utc)).model_dump(mode="json")
    class Session:
        def __init__(self):
            self.requests = []
        def post(self, url, **kwargs):
            self.requests.append((url, kwargs))
            return Response()
    async def run():
        session = Session()
        client = OutputAuthorityClient(base_url="http://test", audience=AUDIENCE, caller_service=CALLER, session=session)
        await client.resolve(request, trace_id="trace")
        await client.resolve(request, trace_id="trace")
        assert session.requests[0][1]["headers"] != session.requests[1][1]["headers"]
        assert session.requests[0][1]["json"] == request.model_dump(mode="json")
        assert not {"jwt", "token", "auth_context", "session_id"} & session.requests[0][1]["json"].keys()
    request = authority_request()
    asyncio.run(run())
