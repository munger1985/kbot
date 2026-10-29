"""精确版本撤回、原接收方回执、重复投递及跨域围栏；不替代实库运行测试。"""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.knowledge_withdrawal import (
    CaseWithdrawalCommand, WITHDRAWAL_EVENT, assert_case_not_withdrawn,
    payload_hash, queue_case_withdrawals, validate_command_source,
    validate_receipt, withdrawal_key, withdrawal_views,
)
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.entities import OperationsKnowledgeAssetEntity
from aiops_agent.repositories.messaging import OutboxRepository
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink
from knowledge_core.application.status import KnowledgeCoreStatusService, KnowledgeObjectNotFoundError
from knowledge_core.repositories.ingestion_repo import BundleRevisionRepository
from platform_clients.knowledge_core import KnowledgeCoreClientError
from platform_core.contracts.knowledge_withdrawal import KnowledgeWithdrawalReceipt, KnowledgeWithdrawalRequest
from platform_core.identity import uuid7


def fixture():
    scoped_tenant = hasattr(OperationsKnowledgeAssetEntity, "tenant_id")
    domain = uuid7() if scoped_tenant else 7
    asset = SimpleNamespace(asset_id=uuid7(), domain_id=domain, asset_kind="DIAGNOSIS_CASE",
        current_version_id=None, created_by=uuid7() if scoped_tenant else "operator")
    if scoped_tenant:
        asset.tenant_id = uuid7()
    version = SimpleNamespace(asset_id=asset.asset_id, asset_version_id=uuid7(), status="REVIEW_REQUIRED")
    index = SimpleNamespace(index_ref_id=uuid7(), collection_id=uuid7(),
        bundle_id=uuid7(), bundle_revision_id=uuid7())
    index.asset_version_id = version.asset_version_id
    repo = SimpleNamespace(get_version=AsyncMock(return_value=version),
        get_asset=AsyncMock(return_value=asset),
        get_version_relations=AsyncMock(return_value={"indexes": [index], "reviews": []}),
        get_review=AsyncMock(return_value=None), add_review=AsyncMock())
    outbox = SimpleNamespace(get_by_idempotency=AsyncMock(return_value=None), add=AsyncMock())
    uow = SimpleNamespace(operations_knowledge=repo, outbox=outbox, commit=AsyncMock())
    @asynccontextmanager
    async def factory():
        yield uow
    service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
    service._uow_factory = factory
    service._kc = SimpleNamespace(withdraw_revision=AsyncMock())
    return SimpleNamespace(asset=asset, version=version, index=index, repo=repo,
        outbox=outbox, uow=uow, factory=factory, service=service, domain=domain)


async def queued(f):
    await queue_case_withdrawals(uow=f.uow, asset=f.asset, version=f.version)
    message = f.outbox.add.await_args.args[0]
    f.outbox.get_by_idempotency.return_value = message
    return message, CaseWithdrawalCommand.model_validate(message.payload_json)


def ack(command, **changes):
    data = dict(request_id=command.request_id, withdrawal_id=command.request_id,
        collection_id=command.collection_id, bundle_id=command.bundle_id,
        bundle_revision_id=command.bundle_revision_id, status="WITHDRAWN",
        withdrawn_at=datetime.now(UTC) - timedelta(seconds=1))
    data.update(changes)
    return KnowledgeWithdrawalReceipt.model_validate(data).model_dump(mode="json")


def test_enqueue_is_atomic_immutable_and_idempotent():
    async def run():
        f = fixture()
        message, command = await queued(f)
        assert message.aggregate_id == f.version.asset_version_id
        assert message.payload_hash == payload_hash(message.payload_json)
        assert message.outbox_id == command.request_id
        assert message.event_type == WITHDRAWAL_EVENT and message.status == "PENDING"
        assert message.idempotency_key == withdrawal_key(f.index.index_ref_id)
        f.uow.commit.assert_not_awaited()
        await queue_case_withdrawals(uow=f.uow, asset=f.asset, version=f.version)
        f.outbox.add.assert_awaited_once()
        with pytest.raises(AIOpsApplicationError, match="新版本"):
            await assert_case_not_withdrawn(uow=f.uow, indexes=[f.index])
    asyncio.run(run())


def test_manual_is_not_silently_withdrawn_as_a_case():
    async def run():
        f = fixture()
        f.asset.asset_kind = "MANUAL"
        await queue_case_withdrawals(uow=f.uow, asset=f.asset, version=f.version)
        f.repo.get_version_relations.assert_not_awaited()
        f.outbox.add.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", [
    "domain", "tenant", "published", "current", "asset", "kind", "index",
    "collection", "bundle", "revision", "hash", "payload", "request", "event", "aggregate",
])
def test_withdrawal_rejects_tampered_or_stale_source(mutation):
    async def run():
        f = fixture()
        message, command = await queued(f)
        if mutation == "domain": f.asset.domain_id = 99
        if mutation == "tenant": f.asset.tenant_id = uuid7()
        if mutation == "published": f.version.status = "PUBLISHED"
        if mutation == "current": f.asset.current_version_id = f.version.asset_version_id
        if mutation == "asset": f.version.asset_id = uuid7()
        if mutation == "kind": f.asset.asset_kind = "MANUAL"
        if mutation == "index": f.index.index_ref_id = uuid7()
        if mutation in {"collection", "bundle", "revision"}:
            setattr(f.index, {"collection": "collection_id", "bundle": "bundle_id",
                "revision": "bundle_revision_id"}[mutation], uuid7())
        if mutation == "hash": message.payload_hash = "0" * 64
        if mutation == "payload": message.payload_json = {}
        if mutation == "request": message.outbox_id = uuid7()
        if mutation == "event": message.event_type = "OPS_RUN_CREATED"
        if mutation == "aggregate": message.aggregate_id = uuid7()
        with pytest.raises(AIOpsApplicationError):
            await validate_command_source(uow=f.uow, command=command, domain_id=f.domain)
    asyncio.run(run())


@pytest.mark.parametrize("field", [
    "request_id", "collection_id", "bundle_id", "bundle_revision_id",
    "status", "withdrawn_at", "missing", "extra",
])
def test_ack_must_be_exact_and_typed(field):
    async def run():
        f = fixture()
        _, command = await queued(f)
        response = ack(command)
        if field in {"request_id", "collection_id", "bundle_id", "bundle_revision_id"}:
            response[field] = str(uuid7())
        elif field == "status": response[field] = "PENDING"
        elif field == "withdrawn_at": response[field] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        elif field == "missing": response.pop("withdrawn_at")
        elif field == "extra": response["cache_cleared"] = True
        with pytest.raises(AIOpsApplicationError):
            validate_receipt(command, response)
    asyncio.run(run())


def test_delivery_persists_receipt_after_remote_commit_and_replay_does_not_call_again():
    async def run():
        f = fixture()
        message, command = await queued(f)
        f.service._kc.withdraw_revision.return_value = ack(command)
        await f.service.deliver_case_withdrawal(message.payload_json, caller_service="test-worker")
        review = f.repo.add_review.await_args.args[0]
        assert review.review_id == command.request_id
        assert review.decision == "EXTERNAL_WITHDRAWAL_ACK"
        assert review.before_status == review.after_status == f.version.status
        validate_receipt(command, json.loads(review.comment_text))
        assert f.uow.commit.await_count == 1
        f.repo.get_review.return_value = review
        await f.service.deliver_case_withdrawal(message.payload_json, caller_service="test-worker")
        f.service._kc.withdraw_revision.assert_awaited_once()
        f.repo.add_review.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["network", "wrong_ack", "commit"])
def test_no_persisted_success_when_network_receipt_or_local_commit_fails(failure):
    async def run():
        f = fixture()
        message, command = await queued(f)
        f.service._kc.withdraw_revision.return_value = ack(command)
        if failure == "network":
            f.service._kc.withdraw_revision.side_effect = KnowledgeCoreClientError(
                status_code=503, code="UNAVAILABLE", message="不可用")
        elif failure == "wrong_ack":
            f.service._kc.withdraw_revision.return_value = ack(command, bundle_id=uuid7())
        else:
            f.uow.commit.side_effect = RuntimeError("事务失败")
        with pytest.raises((AIOpsApplicationError, RuntimeError)):
            await f.service.deliver_case_withdrawal(message.payload_json, caller_service="test-worker")
        if failure != "commit":
            f.repo.add_review.assert_not_awaited()
            f.uow.commit.assert_not_awaited()
        assert message.status == "PENDING"
    asyncio.run(run())


@pytest.mark.parametrize("field", ["status", "withdrawal_id", "request_id"])
def test_receipt_requires_explicit_ack_fields(field):
    async def run():
        f = fixture()
        _, command = await queued(f)
        response = ack(command)
        response.pop(field)
        with pytest.raises(AIOpsApplicationError):
            validate_receipt(command, response)
    asyncio.run(run())


def test_malformed_command_is_permanent_failure_and_does_not_call_consumer():
    async def run():
        f = fixture()
        with pytest.raises(AIOpsApplicationError) as error:
            await f.service.deliver_case_withdrawal({"request_id": "invalid"}, caller_service="worker")
        assert error.value.retryable is False
        f.service._kc.withdraw_revision.assert_not_awaited()
    asyncio.run(run())


def test_client_uses_scoped_internal_request_and_frozen_delivery_id():
    from platform_clients.knowledge_core import KnowledgeCoreClient
    async def run():
        client = KnowledgeCoreClient.__new__(KnowledgeCoreClient)
        client._json = AsyncMock(return_value={})
        args = dict(bundle_id=uuid7(), bundle_revision_id=uuid7(), request_id=uuid7(), auth_context=object())
        scope = "tenant_id" if "tenant_id" in inspect.signature(client.withdraw_revision).parameters else "domain_id"
        args[scope] = uuid7() if scope == "tenant_id" else 7
        await client.withdraw_revision(**args)
        call = client._json.await_args
        assert call.args[0] == "POST" and str(args[scope]) in call.args[1]
        assert str(args["bundle_revision_id"]) in call.args[1]
        assert call.kwargs["payload"] == {"request_id": str(args["request_id"])}
        assert call.kwargs["auth_context"] is args["auth_context"]
    asyncio.run(run())


def test_route_enforces_scope_before_mutating_consumer(monkeypatch):
    from fastapi import HTTPException
    from knowledge_core.api import status_router
    async def run():
        service = SimpleNamespace(withdraw_revision=AsyncMock())
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(kc_status_service=service)))
        args = dict(bundle_id=uuid7(), bundle_revision_id=uuid7(),
            payload=KnowledgeWithdrawalRequest(request_id=uuid7()), request=request)
        scope = "tenant_id" if "tenant_id" in inspect.signature(status_router.withdraw_revision).parameters else "domain_id"
        args[scope] = uuid7() if scope == "tenant_id" else 7
        def reject(*args, **kwargs):
            raise HTTPException(404)
        monkeypatch.setattr(status_router, "require_tenant_match" if scope == "tenant_id" else "require_domain_match", reject)
        with pytest.raises(HTTPException):
            await status_router.withdraw_revision(**args)
        service.withdraw_revision.assert_not_awaited()
        if scope == "tenant_id":
            monkeypatch.setattr(status_router, "require_tenant_match", lambda *args: None)
            access = AsyncMock(side_effect=HTTPException(403))
            monkeypatch.setattr(status_router, "require_collection_access", access)
            with pytest.raises(HTTPException):
                await status_router.withdraw_revision(**args)
            assert access.await_args.kwargs["permission"] == "manage"
            service.withdraw_revision.assert_not_awaited()
    asyncio.run(run())


def test_canonical_model_and_deployment_artifact_preserve_nullable_withdrawal_state():
    from pathlib import Path
    from knowledge_core.entities import KcBundleRevisionEntity
    columns = KcBundleRevisionEntity.__table__.c
    assert columns.withdrawn_at.nullable and columns.withdrawal_id.nullable
    root = Path(__file__).resolve().parents[3]
    if KcBundleRevisionEntity.__tablename__.startswith("KBOT_"):
        canonical = (root / "database/oracle/knowledge_core/002_ingestion.sql").read_text()
        upgrade = (root / "database/oracle/operations/apply_kc_revision_withdrawal.sql").read_text()
        for name in ("WITHDRAWN_AT", "WITHDRAWAL_ID"):
            assert name in canonical and name in upgrade
        assert "DROP " not in upgrade.upper() and "UPDATE " not in upgrade.upper()
    else:
        from scripts.deployment.init_postgresql import _align_knowledge_withdrawal
        async def run():
            connection = SimpleNamespace(execute=AsyncMock())
            await _align_knowledge_withdrawal(connection, schema="withdrawal_test")
            sql = str(connection.execute.await_args.args[0])
            assert '"withdrawal_test".kc_bundle_revision' in sql
            assert "withdrawn_at timestamptz" in sql and "withdrawal_id uuid" in sql
            assert "UPDATE " not in sql.upper() and "DROP " not in sql.upper()
        asyncio.run(run())


def test_lost_receipt_redelivers_to_same_revision_without_touching_new_current():
    async def run():
        f = fixture()
        message, command = await queued(f)
        f.asset.current_version_id = uuid7()
        f.service._kc.withdraw_revision.return_value = ack(command)
        await f.service.deliver_case_withdrawal(message.payload_json, caller_service="test-worker")
        await f.service.deliver_case_withdrawal(message.payload_json, caller_service="test-worker")
        assert f.service._kc.withdraw_revision.await_count == 2
        assert all(call.kwargs["bundle_revision_id"] == command.bundle_revision_id
            for call in f.service._kc.withdraw_revision.await_args_list)
    asyncio.run(run())


@pytest.mark.parametrize("state", ["PENDING", "PUBLISHING", "PUBLISHED", "FAILED"])
def test_progress_never_treats_delivery_as_receipt(state):
    async def run():
        f = fixture()
        message, command = await queued(f)
        message.status, message.error_code = state, None
        result = await withdrawal_views(uow=f.uow, indexes=[f.index], reviews=[])
        assert result[0]["status"] == ("FAILED" if state == "FAILED" else "PENDING")
        assert result[0]["receipt"] is None
        review = SimpleNamespace(review_id=command.request_id, decision="EXTERNAL_WITHDRAWAL_ACK",
            comment_text=json.dumps(ack(command)))
        result = await withdrawal_views(uow=f.uow, indexes=[f.index], reviews=[review])
        assert result[0]["status"] == "ACKNOWLEDGED"
    asyncio.run(run())


def kc_fixture():
    bundle = SimpleNamespace(bundle_id=uuid7(), collection_id=uuid7(), current_revision_id=uuid7())
    revision = SimpleNamespace(bundle_id=bundle.bundle_id, collection_id=bundle.collection_id,
        bundle_revision_id=uuid7(), withdrawn_at=None, withdrawal_id=None)
    uow = SimpleNamespace(bundles=SimpleNamespace(get_by_id=AsyncMock(return_value=bundle)),
        collections=SimpleNamespace(get_by_id_scope=AsyncMock(return_value=object())),
        revisions=SimpleNamespace(get_by_id=AsyncMock(return_value=revision)), commit=AsyncMock())
    @asynccontextmanager
    async def factory():
        yield uow
    service = KnowledgeCoreStatusService(uow_factory=factory)
    scope = {"tenant_id": uuid7()} if "tenant_id" in inspect.signature(service.withdraw_revision).parameters else {"domain_id": 7}
    return SimpleNamespace(bundle=bundle, revision=revision, uow=uow, service=service,
        params=dict(**scope, bundle_id=bundle.bundle_id, bundle_revision_id=revision.bundle_revision_id, request_id=uuid7()))


def test_receiver_withdraws_old_revision_permanently_with_idempotent_persisted_ack():
    async def run():
        f = kc_fixture()
        current = f.bundle.current_revision_id
        first = await f.service.withdraw_revision(**f.params)
        second = await f.service.withdraw_revision(**f.params)
        assert first == second
        assert first.withdrawn_at == f.revision.withdrawn_at
        assert first.withdrawal_id == f.params["request_id"]
        f.params["request_id"] = uuid7()
        third = await f.service.withdraw_revision(**f.params)
        assert third.withdrawal_id == first.withdrawal_id
        assert third.request_id == f.params["request_id"]
        assert f.bundle.current_revision_id == current
        assert f.uow.commit.await_count == 3
        assert f.uow.revisions.get_by_id.await_args.kwargs["lock"] is True
    asyncio.run(run())


@pytest.mark.parametrize("missing", ["bundle", "scope", "revision", "wrong_bundle", "wrong_collection"])
def test_receiver_does_not_ack_wrong_scope(missing):
    async def run():
        f = kc_fixture()
        if missing == "bundle": f.uow.bundles.get_by_id.return_value = None
        if missing == "scope": f.uow.collections.get_by_id_scope.return_value = None
        if missing == "revision": f.uow.revisions.get_by_id.return_value = None
        if missing == "wrong_bundle": f.revision.bundle_id = uuid7()
        if missing == "wrong_collection": f.revision.collection_id = uuid7()
        with pytest.raises(KnowledgeObjectNotFoundError):
            await f.service.withdraw_revision(**f.params)
        f.uow.commit.assert_not_awaited()
    asyncio.run(run())


def test_receiver_does_not_return_receipt_when_commit_fails():
    async def run():
        f = kc_fixture()
        f.uow.commit.side_effect = RuntimeError("未提交")
        with pytest.raises(RuntimeError):
            await f.service.withdraw_revision(**f.params)
    asyncio.run(run())


def test_withdrawal_is_not_a_logging_fallback():
    async def run():
        f = fixture()
        message, _ = await queued(f)
        fallback = SimpleNamespace(publish=AsyncMock())
        sink = AIOpsDomainOutboxSink(runtime_service=None, fallback=fallback)
        with pytest.raises(RuntimeError):
            await sink.publish(WITHDRAWAL_EVENT, message.payload_json)
        fallback.publish.assert_not_awaited()
        sink = AIOpsDomainOutboxSink(runtime_service=None, fallback=fallback,
            operations_knowledge_service=f.service, caller_service="test-worker")
        f.service.deliver_case_withdrawal = AsyncMock()
        await sink.publish(WITHDRAWAL_EVENT, message.payload_json)
        f.service.deliver_case_withdrawal.assert_awaited_once_with(message.payload_json, caller_service="test-worker")
    asyncio.run(run())


def test_request_cannot_assert_ack_or_change_scope():
    with pytest.raises(ValidationError):
        KnowledgeWithdrawalRequest.model_validate({"request_id": uuid7(), "status": "WITHDRAWN"})


def test_retire_and_source_invalidation_save_withdrawal_in_same_transaction(monkeypatch):
    async def run():
        f = fixture()
        f.version.row_version = 1
        f.version.status = f.asset.status = "PUBLISHED"
        f.asset.current_version_id = f.version.asset_version_id
        f.service._actor = lambda context: f.asset.created_by
        f.service.get_version = AsyncMock(return_value={})
        args = dict(domain_id=f.domain, asset_version_id=f.version.asset_version_id,
            decision="RETIRE", expected_row_version=1, comment=None, context=object())
        if hasattr(f.asset, "tenant_id"): args["tenant_id"] = f.asset.tenant_id
        await f.service.review(**args)
        assert f.version.status == "RETIRED" and f.asset.current_version_id is None
        f.outbox.add.assert_awaited_once()
        f.uow.commit.assert_awaited_once()

        f = fixture()
        f.version.status = f.asset.status = "PUBLISHED"
        f.asset.current_version_id = f.version.asset_version_id
        f.index.last_checked_at = None
        f.repo.next_case_index_for_validation = AsyncMock(return_value=(f.index, f.domain))
        f.repo.get_version_relations.return_value["sources"] = []
        monkeypatch.setattr("aiops_agent.application.operations_knowledge.validate_case_sources",
            AsyncMock(return_value=False))
        assert await f.service.revalidate_next_case()
        assert f.version.status == f.asset.status == "REVIEW_REQUIRED"
        assert f.asset.current_version_id is None
        f.outbox.add.assert_awaited_once()
        f.uow.commit.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("method", ["resolve_current_scope", "list_current_scope"])
def test_current_scope_resolution_excludes_withdrawn_revision(method):
    async def run():
        result = SimpleNamespace(scalars=lambda: [])
        session = SimpleNamespace(execute=AsyncMock(return_value=result))
        repo = BundleRevisionRepository(session)
        args = {"collection_ids": [uuid7()]}
        if "tenant_id" in inspect.signature(getattr(repo, method)).parameters:
            args["tenant_id"] = uuid7()
        if method == "resolve_current_scope": args["bundle_revision_ids"] = [uuid7()]
        await getattr(repo, method)(**args)
        query = str(session.execute.await_args.args[0]).lower()
        assert "withdrawn_at is null" in query
    asyncio.run(run())


@pytest.mark.parametrize("channel", ["discovery_text", "discovery_vector", "fallback_text",
    "evidence_text", "evidence_vector", "citation"])
def test_all_search_and_citation_paths_filter_permanent_withdrawal(channel):
    from knowledge_core.repositories.discovery_repo import DiscoveryRepository
    from knowledge_core.repositories.ingestion_repo import EvidenceRepository
    from knowledge_core.application.evidence_retrieval import EvidenceScope
    async def run():
        result = SimpleNamespace(scalars=lambda: [], all=lambda: [], one_or_none=lambda: None)
        session = SimpleNamespace(execute=AsyncMock(return_value=result))
        scope = EvidenceScope(collection_id=uuid7(), bundle_id=uuid7(), bundle_revision_id=uuid7())
        if channel.startswith("discovery") or channel == "fallback_text":
            repo = DiscoveryRepository(session)
            args = dict(collection_id=scope.collection_id, bundle_revision_ids=[scope.bundle_revision_id])
            if channel == "discovery_text":
                await repo.search_text(**args, query="database")
            elif channel == "discovery_vector":
                await repo.search_vector(**args, vector=[0.1, 0.2])
            else:
                name = "postgresql_query" if "postgresql_query" in inspect.signature(repo._search_evidence_text).parameters else "oracle_query"
                await repo._search_evidence_text(**args, **{name: "database"}, limit=10, max_security_level=3)
        else:
            repo = EvidenceRepository(session)
            if channel == "evidence_text":
                await repo.search_text(scope=scope, query="database")
            elif channel == "evidence_vector":
                await repo.search_vector(scope=scope, vector=[0.1, 0.2])
            else:
                args = dict(citation_id=uuid7(), max_security_level=3)
                if "tenant_id" in inspect.signature(repo.get_citation_content).parameters:
                    args["tenant_id"] = uuid7()
                await repo.get_citation_content(**args)
        assert session.execute.await_count >= 1
        for call in session.execute.await_args_list:
            assert "withdrawn_at is null" in str(call.args[0]).lower()
    asyncio.run(run())


@pytest.mark.parametrize("candidate,locked", [(False, False), (True, False), (True, True)])
def test_slow_retry_is_fenced_bounded_and_keeps_attempt_history(candidate, locked):
    async def run():
        message = SimpleNamespace(attempt_count=27, status="FAILED", max_attempts=27)
        first = SimpleNamespace(scalar_one_or_none=lambda: uuid7() if candidate else None)
        second = SimpleNamespace(scalar_one_or_none=lambda: message if locked else None)
        session = SimpleNamespace(execute=AsyncMock(side_effect=[first, second]))
        repo = OutboxRepository(session)
        now = datetime.now(UTC)
        changed = await repo.rearm_case_withdrawal(now=now, before=now-timedelta(minutes=5))
        assert changed is (candidate and locked)
        queries = [str(call.args[0]).lower() for call in session.execute.await_args_list]
        assert "event_type" in queries[0] and "error_code" in queries[0]
        assert "limit" in queries[0]
        if candidate: assert "for update" in queries[1]
        if changed:
            assert message.status == "RETRY_WAIT" and message.max_attempts == 39
            assert message.attempt_count == 27 and message.available_at == now
    asyncio.run(run())
