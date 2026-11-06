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
    validate_receipt, withdrawal_key, withdrawal_views, KnowledgeWithdrawalReceipt,
)
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.entities import OperationsKnowledgeAssetEntity
from aiops_agent.repositories.messaging import OutboxRepository
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink
from platform_core.identity import uuid7


def fixture():
    scoped_tenant = hasattr(OperationsKnowledgeAssetEntity, "tenant_id")
    domain = uuid7() if scoped_tenant else 7
    asset = SimpleNamespace(asset_id=uuid7(), domain_id=domain, asset_kind="DIAGNOSIS_CASE",
        current_version_id=None, created_by=uuid7() if scoped_tenant else "operator")
    if scoped_tenant:
        asset.tenant_id = uuid7()
    version = SimpleNamespace(asset_id=asset.asset_id, asset_version_id=uuid7(), status="REVIEW_REQUIRED")
    index = SimpleNamespace(index_ref_id=uuid7(), document_id=uuid7(),
        document_version_id=uuid7(), ingestion_job_id=uuid7())
    index.asset_version_id = version.asset_version_id
    repo = SimpleNamespace(get_version=AsyncMock(return_value=version),
        get_asset=AsyncMock(return_value=asset),
        get_version_relations=AsyncMock(return_value={"indexes": [index], "reviews": []}),
        get_review=AsyncMock(return_value=None), add_review=AsyncMock())
    outbox = SimpleNamespace(get_by_idempotency=AsyncMock(return_value=None), add=AsyncMock())
    uow = SimpleNamespace(operations_knowledge=repo, outbox=outbox, commit=AsyncMock(),
        knowledge_documents=SimpleNamespace(inspect=AsyncMock(return_value={"status":"PUBLISHED","revision":1})),
        knowledge_retrieval=SimpleNamespace(withdraw=AsyncMock()))
    @asynccontextmanager
    async def factory():
        yield uow
    service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
    service._uow_factory = factory
    service._documents = SimpleNamespace(withdraw=AsyncMock(), access=lambda context: None)
    return SimpleNamespace(asset=asset, version=version, index=index, repo=repo,
        outbox=outbox, uow=uow, factory=factory, service=service, domain=domain)


async def queued(f):
    await queue_case_withdrawals(uow=f.uow, asset=f.asset, version=f.version)
    message = f.outbox.add.await_args.args[0]
    f.outbox.get_by_idempotency.return_value = message
    return message, CaseWithdrawalCommand.model_validate(message.payload_json)


def ack(command, **changes):
    data = dict(request_id=command.request_id, withdrawal_id=command.request_id,
        document_id=command.document_id, document_version_id=command.document_version_id,
        ingestion_job_id=command.ingestion_job_id, status="WITHDRAWN",
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
    "document", "version", "job", "hash", "payload", "request", "event", "aggregate",
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
        if mutation in {"document", "version", "job"}:
            setattr(f.index, {"document": "document_id", "version": "document_version_id",
                "job": "ingestion_job_id"}[mutation], uuid7())
        if mutation == "hash": message.payload_hash = "0" * 64
        if mutation == "payload": message.payload_json = {}
        if mutation == "request": message.outbox_id = uuid7()
        if mutation == "event": message.event_type = "OPS_RUN_CREATED"
        if mutation == "aggregate": message.aggregate_id = uuid7()
        with pytest.raises(AIOpsApplicationError):
            await validate_command_source(uow=f.uow, command=command, domain_id=f.domain)
    asyncio.run(run())


@pytest.mark.parametrize("field", [
    "request_id", "document_id", "document_version_id", "ingestion_job_id",
    "status", "withdrawn_at", "missing", "extra",
])
def test_ack_must_be_exact_and_typed(field):
    async def run():
        f = fixture()
        _, command = await queued(f)
        response = ack(command)
        if field in {"request_id", "document_id", "document_version_id", "ingestion_job_id"}:
            response[field] = str(uuid7())
        elif field == "status": response[field] = "PENDING"
        elif field == "withdrawn_at": response[field] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        elif field == "missing": response.pop("withdrawn_at")
        elif field == "extra": response["cache_cleared"] = True
        with pytest.raises(AIOpsApplicationError):
            validate_receipt(command, response)
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
        f.service._documents.withdraw.assert_not_awaited()
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


def test_retire_and_source_invalidation_save_withdrawal_in_same_transaction(monkeypatch):
    async def run():
        f = fixture()
        f.version.row_version = 1
        f.repo.add_review = AsyncMock()
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
def test_delivery_withdrawal_and_receipt_share_transaction_and_replay_is_idempotent():
    async def run():
        f = fixture()
        message, command = await queued(f)
        await f.service.deliver_case_withdrawal(message.payload_json, caller_service="test-worker")
        review = f.repo.add_review.await_args.args[0]
        assert review.review_id == command.request_id
        validate_receipt(command, json.loads(review.comment_text))
        f.uow.commit.assert_awaited_once()
        f.repo.get_review.return_value = review
        await f.service.deliver_case_withdrawal(message.payload_json, caller_service="test-worker")
        f.service._documents.withdraw.assert_awaited_once()
        f.repo.add_review.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["withdraw", "commit"])
def test_local_failure_does_not_persist_false_receipt(failure):
    async def run():
        f = fixture()
        message, command = await queued(f)
        if failure == "withdraw":
            f.service._documents.withdraw.side_effect = RuntimeError("合成撤回失败")
        else:
            f.uow.commit.side_effect = RuntimeError("合成提交失败")
        with pytest.raises(RuntimeError):
            await f.service.deliver_case_withdrawal(message.payload_json, caller_service="test-worker")
        if failure == "withdraw":
            f.repo.add_review.assert_not_awaited()
            f.uow.commit.assert_not_awaited()
        assert message.status == "PENDING"
    asyncio.run(run())
