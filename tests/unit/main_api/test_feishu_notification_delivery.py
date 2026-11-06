"""外部投递共同合同、HTTP 故障与无 Inbox 投影回归。"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import ValidationError

from platform_core.notifications.delivery import (
    BindingCreate, DeliveryScope, DeliveryStatus as Status, ExternalPolicyCreate,
    can_retry_uncertain, payload_hash, render_card, stable_id,
)
from platform_core.notifications.external_config import ExternalDeliveryConfig
from platform_core.notifications.external_projection import after_quiet, enqueue_external_event, noise_schedule
from platform_core.notifications.feishu import FeishuSender


def test_identity_is_structural_and_payload_frozen():
    assert stable_id("a:b", "c") != stable_id("a", "b:c")
    assert len(str(stable_id("event", "target"))) <= 50
    assert payload_hash({"x": 1, "y": 2}) == payload_hash({"y": 2, "x": 1})


def test_dedupe_is_not_permanent():
    now = datetime.now(timezone.utc)
    assert can_retry_uncertain(now - timedelta(seconds=3299), now)
    assert not can_retry_uncertain(now - timedelta(seconds=3300), now)
    assert not can_retry_uncertain(now + timedelta(seconds=1), now)
    assert not can_retry_uncertain(None, now)


def test_outbound_card_uses_no_business_text():
    card = render_card(event_type="aiops.execution.succeeded", url="https://portal.test/notifications/open/x")
    assert "效果待验证" in json.dumps(card, ensure_ascii=False)
    with pytest.raises(ValueError):
        render_card(event_type="aiops.proposal.review_required", url="https://portal.test", group=True)
    group = render_card(event_type="aiops.situation.detected", url="https://portal.test", group=True)
    assert "@all" not in json.dumps(group)


@pytest.mark.parametrize("values", [
    {"token_url": "https://evil.test/token"},
    {"message_origin": "http://open.feishu.cn"},
    {"enabled": True},
    {"public_origin": "https://portal.test/redirect"},
    {"fixture_mode": True, "token_url": "http://evil.test/token"},
    {"quiet_start_utc_hour": 20},
    {"quiet_start_utc_hour": 20, "quiet_end_utc_hour": 20},
    {"coalesce_seconds": 301},
])
def test_configuration_rejects_uncontrolled_endpoints(values):
    with pytest.raises(ValidationError):
        ExternalDeliveryConfig(**values)


def test_production_rejects_fixtures():
    config = ExternalDeliveryConfig(fixture_mode=True, token_url="http://localhost/token", message_origin="http://localhost")
    with pytest.raises(ValueError):
        config.validate_runtime(production=True)


@pytest.mark.asyncio
async def test_oauth_singleflight_and_new_protocol():
    requests = []
    async def transport(request):
        requests.append(request)
        if request.url.host == "accounts.feishu.cn":
            data = parse_qs(request.content.decode())
            assert data["grant_type"] == ["client_credentials"]
            assert data["client_id"] == ["cli_test"]
            assert data["client_secret"] == ["fixture-secret"]
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"code": 0, "access_token": "x" * 5000, "expires_in": 3600, "token_type": "Bearer"})
        assert request.url.params["receive_id_type"] == "open_id"
        payload = json.loads(request.content)
        assert payload["msg_type"] == "interactive" and isinstance(payload["content"], str)
        return httpx.Response(200, json={"code": 0, "data": {"message_id": "om_test"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        sender = FeishuSender(config=ExternalDeliveryConfig(), client=client)
        results = await asyncio.gather(*(sender.send(connection={"id": "c", "app_id": "cli_test", "credential_version": 1},
            secret="fixture-secret", target_type="open_id", external_id="ou_test",
            content={"elements": []}, provider_uuid=str(stable_id(i))) for i in range(8)))
        assert all(result.status == Status.ACCEPTED for result in results)
        assert sum(request.url.host == "accounts.feishu.cn" for request in requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body,expected", [
    (200, {"code": 0, "data": {}}, Status.UNKNOWN),
    (200, {"code": 230002, "msg": "secret-must-not-be-returned"}, Status.BLOCKED),
    (200, {"code": 99991400}, Status.RETRY_WAIT),
    (429, {}, Status.RETRY_WAIT),
    (503, {}, Status.UNKNOWN),
    (200, [], Status.UNKNOWN),
    (200, {"code": 0, "data": {"message_id": "invalid-receipt"}}, Status.UNKNOWN),
    (201, {"code": 0, "data": {"message_id": "om_uncertain"}}, Status.UNKNOWN),
    (401, [], Status.RETRY_WAIT),
    (403, [], Status.BLOCKED),
])
async def test_provider_failures_are_honest_and_safe(status, body, expected):
    def transport(request):
        if request.url.host == "accounts.feishu.cn":
            return httpx.Response(200, json={"access_token": "fake", "expires_in": 3600, "token_type": "Bearer"})
        return httpx.Response(status, json=body, headers={"x-ogw-ratelimit-reset": "7"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await FeishuSender(config=ExternalDeliveryConfig(), client=client).send(
            connection={"id": "c", "app_id": "cli_test", "credential_version": 1}, secret="fixture-secret",
            target_type="open_id", external_id="ou_test", content={}, provider_uuid="one")
    assert result.status == expected
    assert "secret-must-not" not in repr(result)
    if status == 429:
        assert result.retry_after_seconds == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("exception,expected", [(httpx.ReadTimeout, Status.UNKNOWN), (httpx.ConnectError, Status.RETRY_WAIT)])
async def test_lost_response_is_not_definite_failure(exception, expected):
    def transport(request):
        if request.url.host == "accounts.feishu.cn":
            return httpx.Response(200, json={"access_token": "fake", "expires_in": 3600, "token_type": "Bearer"})
        raise exception("hidden detail", request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await FeishuSender(config=ExternalDeliveryConfig(), client=client).send(
            connection={"id": "c", "app_id": "cli_test", "credential_version": 1}, secret="fixture-secret",
            target_type="open_id", external_id="ou_test", content={}, provider_uuid="one")
    assert result.status == expected and "hidden" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("body,expected", [
    ({"code": 99991400}, Status.RETRY_WAIT),
    ({"access_token": "fake", "expires_in": float("inf"), "token_type": "Bearer"}, Status.BLOCKED),
    ({"error": "invalid_client"}, Status.BLOCKED),
])
async def test_token_business_error_and_invalid_expiry_do_not_send_message(body, expected):
    requests = []
    def transport(request):
        requests.append(request)
        # 手工 JSON 允许构造非法 Infinity 响应，验证不能被缓存。
        return httpx.Response(200, content=json.dumps(body), headers={"Content-Type": "application/json"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await FeishuSender(config=ExternalDeliveryConfig(), client=client).send(
            connection={"id": "c", "app_id": "cli_test", "credential_version": 1}, secret="fixture-secret",
            target_type="open_id", external_id="ou_test", content={}, provider_uuid="one")
    assert result.status == expected
    assert len(requests) == 1 and requests[0].url.host == "accounts.feishu.cn"


class ProjectionRepository:
    def __init__(self, *, missing=False, verified=True):
        self.connection = {"id": uuid4(), "enabled": True, "credential_version": 1}
        self.binding = {"id": uuid4(), "connection_id": self.connection["id"], "actor_id": "approver",
            "target_type": "open_id", "external_id": "ou_test", "row_version": 1,
            "status": "VERIFIED" if verified else "UNVERIFIED"}
        self.missing, self.records = missing, {}
    async def active_policies(self, **kwargs):
        return [{"connection_id": self.connection["id"], "event_types": ["aiops.proposal.review_required"],
                 "binding_ids": [] if self.missing else [self.binding["id"]], "actor_ids": ["approver"]}]
    async def get(self, kind, object_id, **kwargs):
        return self.connection if kind == "connection" else (None if self.missing else self.binding)
    async def bindings_for_connection(self, **kwargs):
        return [] if self.missing else [self.binding]
    async def add_once(self, kind, values, **kwargs):
        key = (kind, values["id"])
        if key in self.records:
            return False
        self.records[key] = dict(values)
        return True
    async def change(self, kind, object_id, values, **kwargs):
        self.records[(kind, object_id)].update(values)

    async def coalesce_candidates(self, coalesce_key, exclude_id, **kwargs):
        return [row | {"row_version": 1} for (kind, key), row in self.records.items()
            if kind == "job" and key != exclude_id and row["coalesce_key"] == coalesce_key
            and row["status"] in {Status.PENDING, Status.RETRY_WAIT} and not row.get("first_sent_at")]

    async def audit(self, **kwargs):
        pass


def test_quiet_utc_half_open_and_cross_day_boundaries():
    config = ExternalDeliveryConfig(quiet_start_utc_hour=22, quiet_end_utc_hour=8)
    day = datetime(2026, 11, 5, tzinfo=timezone.utc)
    assert after_quiet(config, day.replace(hour=21)) == day.replace(hour=21)
    assert after_quiet(config, day.replace(hour=22)) == day.replace(hour=8) + timedelta(days=1)
    assert after_quiet(config, day.replace(hour=7, minute=59)) == day.replace(hour=8)
    assert after_quiet(config, day.replace(hour=8)) == day.replace(hour=8)
    daytime = ExternalDeliveryConfig(quiet_start_utc_hour=9, quiet_end_utc_hour=17)
    assert after_quiet(daytime, day.replace(hour=12)) == day.replace(hour=17)


@pytest.mark.parametrize("event_type,severity", [
    ("aiops.proposal.review_required", "INFO"), ("aiops.diagnostic.input_required", "WARNING"),
    ("aiops.situation.escalated", "WARNING"), ("aiops.situation.detected", "HIGH"),
    ("aiops.situation.detected", "CRITICAL"), ("aiops.execution.succeeded", "INFO"),
    ("aiops.situation.detected", None),
])
def test_urgent_actionable_and_result_messages_bypass_noise_control(event_type, severity):
    config = ExternalDeliveryConfig(quiet_start_utc_hour=22, quiet_end_utc_hour=8, coalesce_seconds=60)
    now = datetime(2026, 11, 5, 23, 0, 1, tzinfo=timezone.utc)
    assert noise_schedule(config, now=now, event_type=event_type, severity=severity) == (now, None)


@pytest.mark.asyncio
async def test_low_priority_coalescing_retains_evidence_and_replay_never_cancels_latest(monkeypatch):
    repo = ProjectionRepository()
    repo.active_policies = _alert_policies(repo)
    now = datetime(2026, 11, 5, 12, 0, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("platform_core.notifications.external_projection.now_utc", lambda: now)
    kwargs = dict(config=ExternalDeliveryConfig(enabled=True, public_origin="https://portal.test", coalesce_seconds=60),
        repo=repo, scope=DeliveryScope(domain_id="12"), source_event_id=uuid4(),
        event_type="aiops.situation.detected", severity="WARNING", legitimate_actors={"approver"},
        route={"route_key":"aiops.situation.detail","route_params":{"domain_id":"12","situation_id":str(uuid4())}},
        occurred_at=now)
    await enqueue_external_event(**kwargs)
    old_id = next(key for kind, key in repo.records if kind == "job")
    newer = kwargs | {"source_event_id": uuid4()}
    await enqueue_external_event(**newer)
    assert repo.records[("job", old_id)]["last_error_code"] == "NOTIFICATION_COALESCED"
    assert repo.records[("job", old_id)]["status"] == Status.CANCELLED
    await enqueue_external_event(**kwargs)
    jobs = [row for (kind, _), row in repo.records.items() if kind == "job"]
    assert len(jobs) == 2
    pending = [row for row in jobs if row["status"] == Status.PENDING]
    assert len(pending) == 1 and pending[0]["source_event_id"] == newer["source_event_id"]
    assert pending[0]["available_at"] == now.replace(minute=1, second=0)
    assert len([1 for kind, _ in repo.records if kind == "entry"]) == 2
    # 另一资源、另一等级或另一固定时间窗不能被错误合并。
    await enqueue_external_event(**(newer | {"source_event_id":uuid4(), "severity":"INFO"}))
    await enqueue_external_event(**(newer | {"source_event_id":uuid4(),
        "route":{"route_key":"aiops.situation.detail","route_params":{"domain_id":"12","situation_id":str(uuid4())}}}))
    now += timedelta(seconds=60)
    await enqueue_external_event(**(newer | {"source_event_id":uuid4()}))
    assert len([row for (kind, _), row in repo.records.items() if kind == "job" and row["status"] == Status.PENDING]) == 4


def _alert_policies(repo):
    async def policies(**kwargs):
        return [{"connection_id":repo.connection["id"], "event_types":["aiops.situation.detected"],
            "binding_ids":[repo.binding["id"]], "actor_ids":[]}]
    return policies


@pytest.mark.asyncio
async def test_quiet_never_extends_expiry_or_discards_the_task(monkeypatch):
    repo = ProjectionRepository()
    repo.active_policies = _alert_policies(repo)
    now = datetime(2026, 11, 5, 23, 0, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("platform_core.notifications.external_projection.now_utc", lambda: now)
    config = ExternalDeliveryConfig(enabled=True, public_origin="https://portal.test",
        quiet_start_utc_hour=22, quiet_end_utc_hour=8)
    await enqueue_external_event(config=config, repo=repo, scope=DeliveryScope(domain_id="12"),
        source_event_id=uuid4(), event_type="aiops.situation.detected", severity="INFO",
        legitimate_actors={"approver"}, occurred_at=now,
        route={"route_key":"aiops.situation.detail","route_params":{"domain_id":"12","situation_id":str(uuid4())}})
    job = next(row for (kind, _), row in repo.records.items() if kind == "job")
    assert job["status"] == Status.EXPIRED and job["last_error_code"] == "NOTIFICATION_QUIET_EXCEEDS_TTL"
    assert job["expires_at"] == now + timedelta(seconds=config.job_ttl_seconds)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing,verified,expected", [(False, True, Status.PENDING), (False, False, Status.BLOCKED), (True, False, Status.BLOCKED)])
async def test_external_projection_needs_no_inbox_and_reports_missing_identity(missing, verified, expected):
    repo = ProjectionRepository(missing=missing, verified=verified)
    event_id, run_id, proposal_id = uuid4(), uuid4(), uuid4()
    scope = DeliveryScope(domain_id="12")
    kwargs = dict(config=ExternalDeliveryConfig(enabled=True, public_origin="https://portal.test"),
        repo=repo, scope=scope, source_event_id=event_id, event_type="aiops.proposal.review_required",
        route={"route_key": "aiops.run.detail", "route_params": {"domain_id": "12", "run_id": str(run_id), "proposal_id": str(proposal_id)}},
        legitimate_actors={"approver"}, occurred_at=datetime.now(timezone.utc))
    await enqueue_external_event(**kwargs)
    await enqueue_external_event(**kwargs)
    jobs = [row for (kind, _), row in repo.records.items() if kind == "job"]
    assert len(jobs) == 1 and jobs[0]["status"] == expected
    assert "notification_id" not in jobs[0]
    assert not any(key == "summary" for key in jobs[0]["content"])
    repo.records.clear()
    kwargs["legitimate_actors"] = {"requester"}
    await enqueue_external_event(**kwargs)
    assert not repo.records


def test_scope_and_target_contracts():
    with pytest.raises(ValidationError):
        BindingCreate(connection_id=uuid4(), target_type="open_id", external_id="ou_test")
    with pytest.raises(ValidationError):
        ExternalPolicyCreate(connection_id=uuid4(), event_types=["aiops.execution.succeeded"])
