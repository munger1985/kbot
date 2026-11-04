"""案例资源自动生命周期的离线回归；不连接数据库或真实模型。"""

import asyncio
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.dialects import oracle, postgresql

from knowledge_core.api.case_resource import install_case_resource_boundary
from knowledge_core.application.case_resource import CaseResourceConflict, SystemCaseResourceService
from knowledge_core.entities.collection import KcCollectionEntity
from knowledge_core.repositories.case_resource import SystemCaseResourceRepository
from main_api.application.case_resource_lifecycle import CaseResourceLifecycle
from platform_clients.case_resource import CaseResourceClientError
from platform_core.app_resources.entity import CaseResourceEntity
from platform_core.app_resources.repository import CaseResourceRepository, now
from platform_core.contracts import PrincipalKind
from platform_core.contracts.aiops.case_resource import (
    CASE_RESOURCE_PATH, CASE_RESOURCE_SCOPE, CaseResourceRequest, CaseResourceResult,
)
from platform_core.identity import uuid7
from platform_core.security import (
    AUTH_CONTEXT_HEADER, SERVICE_IDENTITY_HEADER, build_scoped_internal_auth_headers,
    create_service_auth_context,
)


ORACLE = CaseResourceEntity.__tablename__.startswith("KBOT_")
ROOT = Path(__file__).resolve().parents[3]


def request_fixture(**changes):
    values = dict(domain_id="73" if ORACLE else str(uuid7()),
        tenant_id=None if ORACLE else uuid7(), actor_id="operator" if ORACLE else str(uuid7()))
    values.update(changes)
    return CaseResourceRequest(**values)


class Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def all(self):
        return self.rows

    def mappings(self):
        return self

    def one_or_none(self):
        assert len(self.rows) <= 1
        return self.rows[0] if self.rows else None


class Session:
    def __init__(self):
        self.rows, self.collections, self.bindings, self.sql = [], [], [], []
        self.flush = AsyncMock()
        self.scope = {"eligible": 1}

    def add(self, row):
        (self.collections if isinstance(row, KcCollectionEntity) else self.rows).append(row)

    async def get(self, entity, key):
        return next((row for row in self.rows if row.domain_id == key), None)

    async def execute(self, query, params=None):
        self.sql.append((str(query), deepcopy(params)))
        if str(query).lstrip().startswith("INSERT INTO platform_resource_domain_collection"):
            self.bindings.append(dict(params, access_mode="manage"))
            return Result()
        if "platform_resource_domain_collection" in str(query):
            return Result(self.bindings)
        return Result([self.scope] if self.scope is not None else [])

    async def scalars(self, query):
        self.sql.append((str(query.compile(dialect=oracle.dialect() if ORACLE else postgresql.dialect())), {}))
        entity = query.column_descriptions[0]["entity"]
        if entity == KcCollectionEntity:
            return Result(self.collections)
        if query.column_descriptions[0]["name"] == "domain_id":
            return Result([row.domain_id for row in self.rows])
        return Result(self.rows)


class Uow:
    def __init__(self, repository):
        self.case_resources = repository
        self.commit = AsyncMock()
        self.exits = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, kind, *args):
        self.exits.append(kind)


def repositories(request):
    session = Session()
    repository = SystemCaseResourceRepository(session)
    repository._scopes.scope = AsyncMock(return_value={"eligible": 1})
    return session, repository


def test_new_case_resource_is_empty_text_collection_and_idempotent():
    async def run():
        request = request_fixture()
        session, repo = repositories(request)
        uow = Uow(repo)
        service = SystemCaseResourceService(uow_factory=lambda: uow)
        first = await service.ensure(request)
        second = await service.ensure(request)
        assert first.collection_id == second.collection_id
        assert len(session.collections) == 1
        assert session.collections[0].models_json == {}
        assert session.collections[0].parse_policy_json == {"parse_strategy": "TEXT"}
        assert first.configuration_status == "MODELS_REQUIRED"
        assert first.missing_model_roles == ["embedding", "parser_llm", "retrieval_llm"]
        assert repo._scopes.scope.await_args.kwargs["lock"] is True
        if not ORACLE:
            assert len(session.bindings) == 1
        assert uow.commit.await_count == 2
    asyncio.run(run())


@pytest.mark.parametrize("policy", ["AUTO", "TEXT", "VISUAL"])
def test_existing_resource_is_adopted_without_overwriting_identifier_models_policy_or_notes(policy):
    async def run():
        request = request_fixture()
        session, repo = repositories(request)
        collection = await repo.ensure(request)
        collection.display_name = "可调整的展示名"
        collection.models_json = {role: str(uuid7()) for role in ("parser_llm", "retrieval_llm", "embedding")}
        collection.parse_policy_json = {"parse_strategy": policy, "custom": True}
        collection.metadata_json["operator_note"] = "保留人工配置"
        before = deepcopy((collection.collection_id, collection.models_json,
            collection.parse_policy_json, collection.metadata_json))
        result = await SystemCaseResourceService(uow_factory=lambda: Uow(repo)).ensure(request)
        assert result.configuration_status == ("TEXT_POLICY_REQUIRED" if policy == "VISUAL" else "CONFIGURED")
        assert before == (collection.collection_id, collection.models_json,
            collection.parse_policy_json, collection.metadata_json)
        assert len(session.collections) == 1
    asyncio.run(run())


@pytest.mark.parametrize("conflict", ["duplicate", "disabled", "owner", "kind", "fixed", "known_missing", "known_changed"])
def test_conflicts_never_recreate_reactivate_or_merge(conflict):
    async def run():
        request = request_fixture()
        session, repo = repositories(request)
        collection = await repo.ensure(request)
        if conflict == "duplicate":
            session.collections.append(deepcopy(collection))
        elif conflict == "disabled":
            collection.status = "DISABLED"
        elif conflict in {"owner", "kind", "fixed"}:
            key = {"owner": "owner_app_id", "kind": "content_kind", "fixed": "fixed_resource"}[conflict]
            collection.metadata_json[key] = "invalid"
        elif conflict == "known_missing":
            request = request.model_copy(update={"expected_collection_id": collection.collection_id})
            session.collections.clear()
        else:
            request = request.model_copy(update={"expected_collection_id": uuid7()})
        before = len(session.collections)
        with pytest.raises(CaseResourceConflict):
            await repo.ensure(request)
        assert len(session.collections) == before
    asyncio.run(run())


@pytest.mark.parametrize("scope", [None, {"eligible": 0}])
def test_current_scope_is_checked_before_reading_or_creating_collection(scope):
    async def run():
        request = request_fixture()
        session, repo = repositories(request)
        repo._scopes.scope.return_value = scope
        with pytest.raises(CaseResourceConflict, match="SCOPE_INACTIVE"):
            await repo.ensure(request)
        assert session.sql == [] and session.collections == []
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["cross_scope", "readonly", "duplicate", "missing"])
@pytest.mark.skipif(ORACLE, reason="Oracle Collection 直接归属 Domain")
def test_pg_binding_only_repairs_missing_ownership_without_grant_expansion(failure):
    async def run():
        request = request_fixture()
        session, repo = repositories(request)
        collection = await repo.ensure(request)
        if failure == "missing":
            session.bindings.clear()
            assert (await repo.ensure(request)).collection_id == collection.collection_id
            assert len(session.bindings) == 1
            return
        if failure == "cross_scope":
            session.bindings[0]["domain_id"] = uuid7()
        elif failure == "readonly":
            session.bindings[0]["access_mode"] = "consume"
        else:
            session.bindings.append(deepcopy(session.bindings[0]))
        before = deepcopy(session.bindings)
        with pytest.raises(CaseResourceConflict, match="BINDING_CONFLICT"):
            await repo.ensure(request)
        assert session.bindings == before
    asyncio.run(run())


def test_registration_coalesces_events_and_reactivation_reuses_known_resource():
    async def run():
        request, session = request_fixture(), Session()
        repo = CaseResourceRepository(session)
        scope = dict(domain_id=request.domain_id, tenant_id=request.tenant_id, actor_id=request.actor_id)
        await repo.register(**scope)
        await repo.register(**scope)
        assert len(session.rows) == 1
        row = session.rows[0]
        known = row.collection_id = uuid7()
        session.scope = {"eligible": 0}
        await repo.register(**scope, force=True)
        assert row.status == "PAUSED" and row.collection_id == known
        session.scope = {"eligible": 1}
        await repo.register(**scope, force=True)
        assert row.status == "QUEUED" and row.collection_id == known
        assert all("FOR UPDATE" in sql for sql, _ in session.sql)
        session.scope = None
        await repo.register(**scope)
        assert len(session.rows) == 1
    asyncio.run(run())


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_completion_preserves_identity_but_rechecks_domain_and_rejects_stale_lease(late, active):
    async def run():
        request, session = request_fixture(), Session()
        repo = CaseResourceRepository(session)
        await repo.register(domain_id=request.domain_id, tenant_id=request.tenant_id, actor_id=request.actor_id)
        row = session.rows[0]
        row.status, row.lease_token, row.lease_until = "RUNNING", "current", now() + timedelta(seconds=60)
        result = CaseResourceResult(request=request, collection_id=uuid7(),
            missing_model_roles=[], configuration_status="CONFIGURED")
        session.scope = {"eligible": int(active)}
        item = {"domain_id": row.domain_id, "tenant_id": request.tenant_id, "lease_token": "old" if late else "current"}
        await repo.complete(item, result=result)
        assert row.collection_id == result.collection_id
        if late:
            assert row.lease_token == "current"
        else:
            assert row.lease_token is None
            assert row.status == ("READY" if active else "PAUSED")
    asyncio.run(run())


@pytest.mark.parametrize("retryable", [True, False])
def test_errors_have_durable_backoff_and_do_not_clear_known_collection_id(retryable):
    async def run():
        request, session = request_fixture(), Session()
        repo = CaseResourceRepository(session)
        await repo.register(domain_id=request.domain_id, tenant_id=request.tenant_id, actor_id=request.actor_id)
        row = session.rows[0]
        known = row.collection_id = uuid7()
        row.lease_token, row.attempt_count = "lease", 3
        instant = now()
        await repo.complete({"domain_id": row.domain_id, "tenant_id": request.tenant_id,
            "lease_token": "lease"}, error="SAFE_CODE", retryable=retryable)
        assert row.collection_id == known
        assert row.next_attempt_at > instant
        assert row.status == ("QUEUED" if retryable else "FAILED")
        assert row.error_code == "SAFE_CODE"
    asyncio.run(run())


@pytest.mark.parametrize("expired", [True, False])
def test_claim_recovers_expired_lease_and_never_reclaims_live_lease(expired):
    async def run():
        request, session = request_fixture(), Session()
        repo = CaseResourceRepository(session)
        await repo.register(domain_id=request.domain_id, tenant_id=request.tenant_id, actor_id=request.actor_id)
        row = session.rows[0]
        row.status, row.lease_token = "RUNNING", "crashed-worker"
        row.lease_until = now() + timedelta(seconds=-5 if expired else 60)
        item = await repo.claim()
        if expired:
            assert item["lease_token"] != "crashed-worker"
            assert row.lease_until > now()
            assert row.attempt_count == 1
        else:
            assert item is None
            assert row.lease_token == "crashed-worker"
    asyncio.run(run())


def test_discovery_uses_keyset_cursor_instead_of_repeated_first_page():
    async def run():
        repo = SimpleNamespace(discover=AsyncMock(), register=AsyncMock())
        repo.discover.side_effect = [
            [{"domain_id": str(i), "actor_id": "operator"} for i in range(100)],
            [{"domain_id": "100", "actor_id": "operator"}],
        ]
        uow = Uow(repo)
        lifecycle = CaseResourceLifecycle(uow_factory=lambda: uow, client=None)
        await lifecycle.discover()
        await lifecycle.discover()
        assert repo.discover.await_args_list[1].kwargs["after"] == "99"
        assert lifecycle._cursor is None and repo.register.await_count == 101
        assert uow.commit.await_count == 101
    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["ok", "network", "conflict"])
def test_worker_commits_claim_before_http_then_finalizes_in_fresh_transaction(outcome):
    async def run():
        request = request_fixture()
        item = dict(domain_id=request.domain_id, tenant_id=request.tenant_id, actor_id=request.actor_id,
            collection_id=None, allow_create=True, lease_token="lease")
        repo = SimpleNamespace(claim=AsyncMock(return_value=item), complete=AsyncMock())
        uow = Uow(repo)
        async def ensure(payload):
            assert uow.commit.await_count == 1
            if outcome != "ok":
                raise CaseResourceClientError("SAFE_CODE", retryable=outcome == "network")
            return CaseResourceResult(request=payload, collection_id=uuid7(),
                missing_model_roles=[], configuration_status="CONFIGURED")
        lifecycle = CaseResourceLifecycle(uow_factory=lambda: uow, client=SimpleNamespace(ensure=ensure))
        assert await lifecycle.tick()
        assert uow.commit.await_count == 2
        kwargs = repo.complete.await_args.kwargs
        assert kwargs["error"] == (None if outcome == "ok" else "SAFE_CODE")
        assert kwargs["retryable"] == (outcome != "conflict")
    asyncio.run(run())


@pytest.mark.parametrize("credential", ["valid", "missing", "context_only", "identity_only",
    "wrong_caller", "wrong_scope", "empty_scope", "wrong_audience", "not_service", "ordinary_token"])
def test_kc_boundary_requires_main_api_service_both_jwts_and_narrow_scope(credential):
    async def run():
        caller = "kbot-main-api" if ORACLE else "ammolite-main-api"
        audience = "kbot-knowledge-core-api" if ORACLE else "ammolite-knowledge-core-api"
        app = FastAPI()
        install_case_resource_boundary(app, audience=audience, caller_service=caller)
        request = request_fixture()
        result = CaseResourceResult(request=request, collection_id=uuid7(),
            missing_model_roles=[], configuration_status="CONFIGURED")
        app.state.case_resource_service = SimpleNamespace(ensure=AsyncMock(return_value=result))
        actual_caller = "foreign" if credential == "wrong_caller" else caller
        context = create_service_auth_context(caller_service=actual_caller)
        if credential == "not_service":
            context = context.model_copy(update={"principal_kind": PrincipalKind.API_CLIENT, "api_key_id": "test"})
        scope = () if credential == "empty_scope" else ("foreign" if credential == "wrong_scope" else CASE_RESOURCE_SCOPE,)
        if credential == "empty_scope":
            with pytest.raises(ValueError):
                build_scoped_internal_auth_headers(caller_service=actual_caller,
                    audience=audience, scopes=scope, context=context)
            return
        headers = build_scoped_internal_auth_headers(caller_service=actual_caller,
            audience="foreign" if credential == "wrong_audience" else audience, scopes=scope, context=context)
        if credential == "context_only":
            headers.pop(SERVICE_IDENTITY_HEADER)
        elif credential == "identity_only":
            headers.pop(AUTH_CONTEXT_HEADER)
        elif credential in {"missing", "ordinary_token"}:
            headers = {} if credential == "missing" else {"X-KBot-Internal-Token": "ordinary"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(CASE_RESOURCE_PATH, json=request.model_dump(mode="json"), headers=headers)
        assert response.status_code == (200 if credential == "valid" else 403), response.text
        assert app.state.case_resource_service.ensure.await_count == int(credential == "valid")
    asyncio.run(run())


def test_canonical_initializer_and_worker_use_durable_events_not_deployment_provisioner():
    assert not (ROOT / "scripts/deployment/provision_aiops_case_collection.py").exists()
    source = (ROOT / ("database/oracle/bootstrap/aiops/initial_admin.sql" if ORACLE
        else "scripts/deployment/init_postgresql.py")).read_text()
    assert ("MERGE INTO KBOT_PLATFORM_CASE_RESOURCE" if ORACLE else "resource_repo.register(**domain)") in source
    worker = (ROOT / ("services/main_api/src/main_api/entrypoints/notification_worker.py" if ORACLE
        else "services/main_api/src/main_api/entrypoints/api.py")).read_text()
    assert "run_case_resource_worker" in worker
    assert "provision_aiops_case_collection" not in worker


def test_system_case_collection_cannot_be_physically_deleted():
    async def run():
        from knowledge_core.application.collections import KnowledgeCoreCollectionService, CollectionInUseError
        collection = SimpleNamespace(metadata_json={"owner_app_id": "aiops",
            "fixed_resource": True, "content_kind": "diagnosis_case"})
        uow = Uow(None)
        uow.collections = SimpleNamespace(get_by_id_scope=AsyncMock(return_value=collection))
        uow.jobs = SimpleNamespace()
        uow.bindings = SimpleNamespace()
        service = KnowledgeCoreCollectionService(uow_factory=lambda: uow)
        scope = {"domain_id": 73} if ORACLE else {"tenant_id": uuid7()}
        with pytest.raises(CollectionInUseError):
            await service.request_delete(**scope, collection_id=uuid7(), actor_id="operator")
        uow.commit.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("active", [True, False])
def test_status_is_read_only_and_model_readiness_is_separate(active):
    async def run():
        request = request_fixture()
        row = SimpleNamespace(status="READY", collection_id=uuid7(),
            configuration_status="MODELS_REQUIRED", missing_model_roles=["embedding"], error_code=None)
        repository = SimpleNamespace(get=AsyncMock(return_value=row),
            scope=AsyncMock(return_value={"eligible": int(active)}), register=AsyncMock())
        uow = Uow(repository)
        lifecycle = CaseResourceLifecycle(uow_factory=lambda: uow, client=None)
        result = await lifecycle.status(domain_id=request.domain_id, tenant_id=request.tenant_id)
        assert result.resource_status == ("READY" if active else "PAUSED")
        assert result.configuration_status == "MODELS_REQUIRED"
        repository.register.assert_not_awaited()
        uow.commit.assert_not_awaited()
    asyncio.run(run())


def test_real_kc_app_keeps_system_endpoint_under_scoped_boundary():
    async def run():
        from knowledge_core.entrypoints.api import app, SERVICE_NAME
        caller = "kbot-main-api" if ORACLE else "ammolite-main-api"
        request = request_fixture()
        result = CaseResourceResult(request=request, collection_id=uuid7(),
            missing_model_roles=[], configuration_status="CONFIGURED")
        app.state.case_resource_service = SimpleNamespace(ensure=AsyncMock(return_value=result))
        context = create_service_auth_context(caller_service=caller)
        headers = build_scoped_internal_auth_headers(caller_service=caller,
            audience=SERVICE_NAME, scopes=(CASE_RESOURCE_SCOPE,), context=context)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            valid = await client.post(CASE_RESOURCE_PATH, json=request.model_dump(mode="json"), headers=headers)
            invalid = await client.post(CASE_RESOURCE_PATH, json=request.model_dump(mode="json"))
        assert valid.status_code == 200, valid.text
        assert invalid.status_code == 403
    asyncio.run(run())


def test_previously_conflicting_resource_is_not_recreated_after_removal():
    async def run():
        request, session = request_fixture(), Session()
        repo = CaseResourceRepository(session)
        await repo.register(domain_id=request.domain_id, tenant_id=request.tenant_id, actor_id=request.actor_id)
        row = session.rows[0]
        row.lease_token = "lease"
        item = {"domain_id": row.domain_id, "tenant_id": request.tenant_id, "lease_token": "lease"}
        await repo.complete(item, error="CASE_RESOURCE_CONFLICT", retryable=False)
        assert row.creation_allowed == 0
        # 后续网络异常不能把首次创建权限恢复。
        row.lease_token = "retry"
        await repo.complete(dict(item, lease_token="retry"), error="CASE_RESOURCE_UNAVAILABLE", retryable=True)
        assert row.creation_allowed == 0
        _, kc = repositories(request)
        with pytest.raises(CaseResourceConflict, match="KNOWN_COLLECTION_MISSING"):
            await kc.ensure(request.model_copy(update={"allow_create": False}))
    asyncio.run(run())


def test_one_invalid_historical_domain_does_not_block_other_discovery_events():
    async def run():
        repo = SimpleNamespace(discover=AsyncMock(return_value=[
            {"domain_id": "bad", "actor_id": None}, {"domain_id": "good", "actor_id": "operator"}]))
        async def register(**scope):
            if scope["domain_id"] == "bad":
                raise ValueError("缺少历史审计信息")
        repo.register = AsyncMock(side_effect=register)
        uow = Uow(repo)
        lifecycle = CaseResourceLifecycle(uow_factory=lambda: uow, client=None)
        await lifecycle.discover()
        assert repo.register.await_count == 2
        assert uow.commit.await_count == 1
        assert lifecycle._cursor is None
    asyncio.run(run())
