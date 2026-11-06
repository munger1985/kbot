"""投递应用状态机、验证限次、独立入口与人工重试的回归。"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from main_api.application.notification_delivery import ExternalNotificationService, ExternalDeliveryError
from main_api.application.notification_delivery_authority import NotificationDeliveryAuthority
from platform_core.notifications.delivery import BindingConfirmation, DeliveryScope, DeliveryStatus as Status, RetryRequest, SendResult, payload_hash
from platform_core.notifications.external_config import ExternalDeliveryConfig


def setup_service():
    now = datetime.now(timezone.utc)
    scope = DeliveryScope(domain_id="12")
    connection = dict(id=uuid4(), enabled=True, credential_version=1, row_version=1, credential_ref="opaque")
    binding = dict(id=uuid4(), connection_id=connection["id"], actor_id="B", target_type="open_id",
        external_id="ou_B", status="VERIFIED", row_version=1)
    entry = dict(id=uuid4(), recipient_actor_id="B", audience="PERSONAL", expires_at=now+timedelta(days=1),
        event_type="aiops.proposal.review_required", route_key="aiops.run.detail",
        route_params={"domain_id":"12","run_id":str(uuid4()),"proposal_id":str(uuid4())}, domain_id=12)
    job = dict(id=uuid4(), domain_id=12, connection_id=connection["id"], connection_version=1,
        binding_id=binding["id"], binding_version=1, recipient_actor_id="B", target_type="open_id",
        external_id="ou_B", purpose="BUSINESS", entry_id=entry["id"], content={"safe":"fixed"},
        content_hash=payload_hash({"safe":"fixed"}), event_type=entry["event_type"], provider_uuid=str(uuid4()),
        row_version=1, status=Status.UNKNOWN, expires_at=now+timedelta(hours=2), first_sent_at=now,
        generation=0, attempt_count=1, retry_count=0, network_started_at=None)
    records = {"connection":connection,"binding":binding,"entry":entry,"job":job}
    repo = SimpleNamespace(get=AsyncMock(side_effect=lambda kind,**kw: records.get(kind)),
        active_policies=AsyncMock(return_value=[dict(connection_id=connection["id"],event_types=[job["event_type"]],
            binding_ids=[str(binding["id"])],actor_ids=[])]), mark_network=AsyncMock(return_value=True),
        reserve_rate=AsyncMock(return_value=(None,now)), finish=AsyncMock(), claim=AsyncMock(return_value=job),
        change=AsyncMock(return_value=True), add_once=AsyncMock(return_value=True), audit=AsyncMock(),
        record_verification_failure=AsyncMock(), entry_by_id=AsyncMock(return_value=entry))
    active = [0]
    @asynccontextmanager
    async def factory():
        active[0] += 1
        try:
            yield SimpleNamespace(external_delivery=repo, commit=AsyncMock())
        finally:
            active[0] -= 1
    async def send(**kw):
        assert active[0] == 0, "网络调用不能位于数据库事务内"
        return SendResult(Status.ACCEPTED,message_id="om_test")
    service=ExternalNotificationService(uow_factory=factory,
        config=ExternalDeliveryConfig(enabled=True, public_origin="https://portal.test"),
        vault=SimpleNamespace(read=AsyncMock(return_value={"app_secret":"fixture-secret"})),
        sender=SimpleNamespace(send=AsyncMock(side_effect=send)),
        authority=SimpleNamespace(check=AsyncMock(return_value=None),actor_active=AsyncMock(return_value=True),
            domain_active=AsyncMock(return_value=True), delivery_available=AsyncMock(return_value=True)))
    return service, repo, records, scope, now


@pytest.mark.asyncio
@pytest.mark.parametrize("case,expected", [
    ("success",Status.ACCEPTED),("paused",Status.BLOCKED),("connection",Status.BLOCKED),
    ("unverified",Status.BLOCKED),("unbound",Status.CANCELLED),("closed",Status.CANCELLED),
    ("revoked",Status.BLOCKED),("domain",Status.BLOCKED),("hash",Status.BLOCKED),("credential",Status.BLOCKED),
])
async def test_send_rechecks_current_policy_binding_authority_and_has_no_transaction(case,expected):
    service,repo,records,scope,now=setup_service()
    if case=="paused": repo.active_policies.return_value=[]
    if case=="connection": records["connection"]["enabled"]=False
    if case=="unverified": records["binding"]["status"]="UNVERIFIED"
    if case=="unbound": records["binding"]["row_version"]=2
    if case=="closed": service.authority.check.return_value="NOTIFICATION_TASK_CLOSED"
    if case=="revoked": service.authority.check.return_value="NOTIFICATION_ACCESS_REVOKED"
    if case=="domain": service.authority.delivery_available.return_value=False
    if case=="hash": records["job"]["content"]["safe"]="changed"
    if case=="credential": service.vault.read.side_effect=ValueError("sensitive-secret")
    result,_=await service._deliver(scope=scope,job=records["job"])
    assert result.status==expected
    assert service.sender.send.await_count==(1 if case=="success" else 0)
    if case!="success": repo.mark_network.assert_not_awaited()


@pytest.mark.asyncio
async def test_uncertain_attempt_preserved_while_same_uuid_is_queued():
    service,repo,records,scope,now=setup_service()
    service.sender.send.side_effect=None
    service.sender.send.return_value=SendResult(Status.UNKNOWN,"FEISHU_RESPONSE_UNCERTAIN")
    assert await service.dispatch_once()
    kwargs=repo.finish.await_args.kwargs
    assert kwargs["attempt_status"]==Status.UNKNOWN
    assert kwargs["result"].status==Status.RETRY_WAIT
    assert kwargs["retry_count"]==1
    assert service.sender.send.await_args.kwargs["provider_uuid"]==records["job"]["provider_uuid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("expires,expected", [(2, Status.RETRY_WAIT), (0.25, Status.EXPIRED)])
async def test_quiet_hours_rechecked_before_retry_without_network_or_retry_budget(monkeypatch, expires, expected):
    service,repo,records,scope,now=setup_service()
    now = datetime(2026, 11, 5, 7, 30, tzinfo=timezone.utc)
    monkeypatch.setattr("main_api.application.notification_delivery.now_utc", lambda: now)
    service.config = ExternalDeliveryConfig(enabled=True, public_origin="https://portal.test",
        quiet_start_utc_hour=22, quiet_end_utc_hour=8)
    job = records["job"]
    job.update(event_type="aiops.situation.detected", original_severity="INFO",
        expires_at=now+timedelta(hours=expires), first_sent_at=None)
    records["entry"]["expires_at"]=now+timedelta(days=1)
    repo.active_policies.return_value[0]["event_types"]=[job["event_type"]]
    assert await service.dispatch_once()
    result = repo.finish.await_args.kwargs
    assert result["result"].status == expected
    if expected == Status.RETRY_WAIT:
        assert result["available_at"] == now.replace(hour=8, minute=0)
        assert result["retry_count"] == 0
    service.sender.send.assert_not_awaited()
    repo.mark_network.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_rate_wait_does_not_exhaust_retries_or_enter_network():
    service,repo,records,scope,now=setup_service()
    repo.reserve_rate.return_value=(None,now+timedelta(seconds=30))
    await service.dispatch_once()
    assert repo.finish.await_args.kwargs["retry_count"]==0
    service.sender.send.assert_not_awaited()
    repo.mark_network.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [True, False])
async def test_connection_check_cannot_claim_rotated_credentials_are_valid(changed):
    service,repo,records,scope,now=setup_service()
    service.sender.check=AsyncMock(return_value=SendResult(Status.ACCEPTED))
    repo.change.return_value=changed
    if changed:
        result=await service.check_connection(scope=scope,actor_id="admin",object_id=records["connection"]["id"])
        assert result["health"]=="TOKEN_VALID"
    else:
        with pytest.raises(ExternalDeliveryError,match="VERSION_CONFLICT"):
            await service.check_connection(scope=scope,actor_id="admin",object_id=records["connection"]["id"])
        assert repo.audit.await_args.kwargs["outcome"]=="STALE_VERSION_NOT_CHECKED"
    service.sender.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_retry_window_and_new_generation_preserve_frozen_target():
    service,repo,records,scope,now=setup_service()
    job=records["job"]
    await service.retry(scope=scope,actor_id="admin",job_id=job["id"],body=RetryRequest(expected_row_version=1))
    assert repo.change.await_args.kwargs["values"]["status"]==Status.RETRY_WAIT
    repo.add_once.assert_not_awaited()
    job["first_sent_at"]=now-timedelta(seconds=3400)
    with pytest.raises(ExternalDeliveryError,match="DUPLICATE_RISK"):
        await service.retry(scope=scope,actor_id="admin",job_id=job["id"],body=RetryRequest(expected_row_version=1))
    result=await service.retry(scope=scope,actor_id="admin",job_id=job["id"],
        body=RetryRequest(expected_row_version=1,acknowledge_duplicate_risk=True))
    values=repo.add_once.await_args.kwargs["values"]
    assert values["id"]==result["job_id"]!=job["id"]
    assert values["provider_uuid"]!=job["provider_uuid"]
    assert values["content"]==job["content"] and values["external_id"]==job["external_id"]
    assert values["parent_job_id"]==job["id"]
    repo.add_once.return_value=False
    with pytest.raises(ExternalDeliveryError,match="ALREADY_CREATED"):
        await service.retry(scope=scope,actor_id="admin",job_id=job["id"],
            body=RetryRequest(expected_row_version=1,acknowledge_duplicate_risk=True))


@pytest.mark.asyncio
async def test_wrong_code_counts_without_invalidating_queued_verification_and_only_owner_confirms():
    service,repo,records,scope,now=setup_service()
    binding=records["binding"]
    salt="01"*16
    binding.update(status="UNVERIFIED",challenge_salt=salt,challenge_attempts=0,
        challenge_expires_at=now+timedelta(minutes=5),
        challenge_hash=hashlib.scrypt(b"123456",salt=bytes.fromhex(salt),n=16384,r=8,p=1).hex())
    with pytest.raises(ExternalDeliveryError,match="BINDING_NOT_FOUND"):
        await service.confirm(scope=scope,actor_id="C",object_id=binding["id"],
            body=BindingConfirmation(code="123456",expected_row_version=1))
    with pytest.raises(ExternalDeliveryError,match="VERIFICATION_INVALID"):
        await service.confirm(scope=scope,actor_id="B",object_id=binding["id"],
            body=BindingConfirmation(code="000000",expected_row_version=1))
    repo.record_verification_failure.assert_awaited_once()
    repo.change.assert_not_awaited()
    binding["challenge_attempts"]=service.config.verification_max_attempts
    with pytest.raises(ExternalDeliveryError,match="EXPIRED"):
        await service.confirm(scope=scope,actor_id="B",object_id=binding["id"],
            body=BindingConfirmation(code="123456",expected_row_version=1))
    binding["challenge_attempts"]=1
    result=await service.confirm(scope=scope,actor_id="B",object_id=binding["id"],
        body=BindingConfirmation(code="123456",expected_row_version=1))
    assert result["status"]=="VERIFIED"
    assert repo.change.await_args.kwargs["values"]["challenge_ref"] is None


@pytest.mark.asyncio
async def test_independent_entry_never_reads_inbox_and_forwarded_personal_entry_is_denied():
    service,repo,records,scope,now=setup_service()
    with pytest.raises(ExternalDeliveryError,match="NOT_FOUND"):
        await service.resolve_entry(actor_id="C",entry_id=records["entry"]["id"])
    result=await service.resolve_entry(actor_id="B",entry_id=records["entry"]["id"])
    assert result["domain_id"]=="12"
    assert result["route_params"]==records["entry"]["route_params"]
    service.authority.check.return_value="NOTIFICATION_ACCESS_REVOKED"
    with pytest.raises(ExternalDeliveryError,match="REVOKED"):
        await service.resolve_entry(actor_id="B",entry_id=records["entry"]["id"])


@pytest.mark.asyncio
async def test_low_sensitivity_group_still_requires_current_app_and_domain():
    client=SimpleNamespace(resolve_notification_resource=AsyncMock())
    authority=NotificationDeliveryAuthority(uow_factory=None,client=client,caller_service="test")
    authority.delivery_available=AsyncMock(return_value=False)
    decision=await authority.check(scope=DeliveryScope(domain_id="12"),actor_id=None,
        entry={"audience":"GROUP"},actionable=False)
    assert decision=="NOTIFICATION_DOMAIN_UNAVAILABLE"
    client.resolve_notification_resource.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("actor,assignee,initiator,run_actor,expected",[
    ("B","B","A","A",None),("A","B","A","A","NOTIFICATION_APPLICANT_NOT_APPROVER"),
    ("B","C","A","A","NOTIFICATION_NOT_ASSIGNEE"),("B",None,"B","A","NOTIFICATION_APPLICANT_NOT_APPROVER"),
])
async def test_authority_uses_current_approver_and_excludes_both_applicants(actor,assignee,initiator,run_actor,expected):
    service,repo,records,scope,now=setup_service()
    entry=records["entry"]
    from platform_core.contracts.aiops.notification_projection import NotificationResourceRequest,NotificationResourceSnapshot
    request=NotificationResourceRequest(event_type=entry["event_type"],domain_id=scope.domain_id,
        route_key=entry["route_key"],route_params=entry["route_params"])
    client=SimpleNamespace(resolve_notification_resource=AsyncMock(return_value=NotificationResourceSnapshot(
        request=request,actionable=True,source_version=1,initiator_actor_id=initiator,
        run_actor_id=run_actor,assignee_actor_id=assignee).model_dump(mode="json")))
    authority=NotificationDeliveryAuthority(uow_factory=None,client=client,caller_service="test")
    authority.domain_active=AsyncMock(return_value=True)
    authority.delivery_available=AsyncMock(return_value=True)
    authority.permissions=AsyncMock(return_value={"aiops:use","aiops:proposal:approve"})
    assert await authority.check(scope=scope,actor_id=actor,entry=entry,actionable=True)==expected
