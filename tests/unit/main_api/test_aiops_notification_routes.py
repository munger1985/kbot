"""AIOps 通知白名单定位与收件页隔离，不依赖真实 Oracle。"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import oracle

from main_api.application.aiops_notification_routes import AIOPS_RESOURCE_EVENTS, aiops_notification_route
from main_api.application.notification_center import NotificationCenterService
from main_api.entities import NotificationInboxEntity
from main_api.repositories.notifications import NotificationRepository
from platform_core.identity import uuid7
from platform_core.notifications import NotificationEnvelope


def fixture(event="aiops.case.draft_ready"):
    resource_id, asset_id, version_id = str(uuid7()), str(uuid7()), str(uuid7())
    report = event == "aiops.report.ready"
    safe = {"job_id": str(uuid7()) if report else resource_id,
        "job_type": "REPORT_GENERATION" if report else "CASE_EXTRACTION", "source_version": 8}
    if report:
        safe["report_version"] = "3"
    elif event.endswith("draft_ready"):
        safe.update(asset_id=asset_id, asset_version_id=version_id)
    envelope = NotificationEnvelope(domain_id=73, event_type=event,
        resource_type="aiops_report" if report else "aiops_output_job", resource_id=resource_id,
        summary="安全摘要", correlation_id="trace", recipient_actor_ids=["alice"], safe_data=safe)
    row = NotificationInboxEntity(inbox_id=uuid7(), outbox_id=uuid7(), domain_id=73,
        recipient_actor_id="alice", event_type=event, resource_type=envelope.resource_type,
        resource_id=resource_id, title="任务通知", summary="安全摘要", category="RESULT", severity="INFO",
        expires_at=datetime.now(timezone.utc) + timedelta(days=1), row_version=1)
    return row, envelope.model_dump(mode="json")


@pytest.mark.parametrize("event", ["aiops.output_job.started", "aiops.output_job.failed", "aiops.case.draft_ready", "aiops.report.ready"])
def test_exact_locators_only_are_exposed(event):
    row, payload = fixture(event)
    payload["safe_data"]["unrelated_internal_note"] = "不应公开"
    result = aiops_notification_route(row, payload, domain_id=73)
    params = result["route_params"]
    assert params["domain_id"] == "73"
    assert params["job_id"] == payload["safe_data"]["job_id"]
    assert "不应公开" not in str(result)
    if event.endswith("draft_ready"):
        assert params["asset_version_id"] == payload["safe_data"]["asset_version_id"]
    elif event.endswith("report.ready"):
        assert params["report_id"] == row.resource_id and params["report_version"] == "3"
    else:
        assert "asset_id" not in params and "asset_version_id" not in params


@pytest.mark.parametrize("change", ["domain", "event", "resource", "kind", "job", "type", "version", "missing", "unsafe"])
def test_invalid_or_mismatched_case_envelope_fails_closed(change):
    row, payload = fixture()
    if change == "domain":
        payload["domain_id"] = 74
    elif change == "event":
        payload["event_type"] = "aiops.output_job.failed"
    elif change == "resource":
        payload["resource_id"] = str(uuid7())
    elif change == "kind":
        payload["resource_type"] = "aiops_report"
    elif change == "job":
        payload["safe_data"]["job_id"] = str(uuid7())
    elif change == "type":
        payload["safe_data"]["job_type"] = "REPORT_GENERATION"
    elif change == "version":
        payload["safe_data"]["asset_version_id"] = "javascript:alert(1)"
    elif change == "missing":
        payload["safe_data"].pop("asset_id")
    else:
        payload["safe_data"]["password"] = "不应公开"
    assert aiops_notification_route(row, payload, domain_id=73) == {}


def test_public_serialization_batches_only_current_page_and_never_exposes_full_safe_data():
    async def run():
        row, payload = fixture()
        repository = SimpleNamespace(inbox_route_payloads=AsyncMock(return_value={row.inbox_id: payload}))
        result = await NotificationCenterService._serialize_inbox(repository, [row], domain_id=73, actor_id="alice")
        assert result[0]["route_params"]["asset_version_id"] == payload["safe_data"]["asset_version_id"]
        assert "safe_data" not in result[0]
        repository.inbox_route_payloads.assert_awaited_once_with(domain_id=73, actor_id="alice", inbox_ids=[row.inbox_id])
        repository.inbox_route_payloads.reset_mock()
        assert await NotificationCenterService._serialize_inbox(repository, [], domain_id=73, actor_id="alice") == []
        repository.inbox_route_payloads.assert_not_awaited()
    asyncio.run(run())


def test_repository_enrichment_scopes_inbox_and_outbox_without_schema_copy():
    async def run():
        row, payload = fixture()
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(all=lambda: [(row.inbox_id, payload)])))
        repository = NotificationRepository(session)
        assert await repository.inbox_route_payloads(domain_id=73, actor_id="alice", inbox_ids=[]) == {}
        session.execute.assert_not_awaited()
        assert await repository.inbox_route_payloads(domain_id=73, actor_id="alice", inbox_ids=[row.inbox_id]) == {row.inbox_id: payload}
        compiled = session.execute.await_args.args[0].compile(dialect=oracle.dialect())
        sql = str(compiled).upper().replace('"', '')
        assert 'KBOT_NOTIFICATION_INBOX.RECIPIENT_ACTOR_ID' in sql
        assert 'KBOT_NOTIFICATION_INBOX.DOMAIN_ID' in sql
        assert 'KBOT_NOTIFICATION_OUTBOX.DOMAIN_ID' in sql
        assert 'KBOT_NOTIFICATION_OUTBOX.PRODUCER_SERVICE' in sql
        assert "alice" in compiled.params.values() and "aiops" in compiled.params.values()
    asyncio.run(run())


@pytest.mark.parametrize("event", AIOPS_RESOURCE_EVENTS)
def test_run_situation_and_proposal_routes_preserve_exact_domain_and_resource(event):
    resource_type, route_key, id_key = AIOPS_RESOURCE_EVENTS[event]
    row, _ = fixture()
    row.event_type, row.resource_type = event, resource_type
    run_id = str(uuid7())
    envelope = NotificationEnvelope(domain_id=73, event_type=event,
        resource_type=resource_type, resource_id=row.resource_id,
        summary="安全摘要", correlation_id="trace", recipient_actor_ids=["alice"],
        safe_data={"run_id": run_id} if resource_type == "aiops_proposal" else {})
    payload = envelope.model_dump(mode="json")
    result = aiops_notification_route(row, payload, domain_id=73)
    assert result["route_key"] == route_key
    assert result["route_params"] == {
        "domain_id": "73", id_key: row.resource_id,
        **({"run_id": run_id} if resource_type == "aiops_proposal" else {}),
    }
    assert aiops_notification_route(row, payload, domain_id=74) == {}
    payload["resource_id"] = "javascript:alert(1)"
    row.resource_id = payload["resource_id"]
    assert aiops_notification_route(row, payload, domain_id=73) == {}


def test_proposal_without_valid_run_cannot_navigate_to_another_resource():
    row, _ = fixture()
    row.event_type, row.resource_type = "aiops.proposal.review_required", "aiops_proposal"
    for safe in ({}, {"run_id": "javascript:alert(1)"}):
        payload = NotificationEnvelope(domain_id=73, event_type=row.event_type,
            resource_type=row.resource_type, resource_id=row.resource_id,
            summary="安全摘要", correlation_id="trace", safe_data=safe).model_dump(mode="json")
        assert aiops_notification_route(row, payload, domain_id=73) == {}
