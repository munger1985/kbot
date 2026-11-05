"""平台真实通知适配器回归：确切任务身份、安全摘要、事务时间与受控定位。"""

import asyncio
import inspect
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.repositories.platform_notification import PlatformNotificationRepository
from platform_core.contracts.aiops.output_jobs import OutputJobResult
from platform_core.identity import uuid7


@pytest.mark.parametrize("kind", ["failed", "input", "started", "situation", "proposal", "case"])
def test_diagnostic_notifications_keep_the_authoritative_domain_and_resource(kind):
    async def run():
        repo, job, payload, count = adapter()
        source = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7(), domain_id=job.domain_id,
            tenant_id=job.tenant_id, actor_id=job.actor_id, trace_id="trace", situation_id=None)
        subscriptions = SimpleNamespace(recipient_actor_ids=AsyncMock(return_value=[job.actor_id]))
        if job.tenant_id:
            repo._notification_subscriptions = subscriptions
        else:
            repo._uow.notification_subscriptions = subscriptions
        resource_id = source.ops_run_id
        id_key = "run_id"
        if kind in {"failed", "input"}:
            await repo.emit_run_event(run=source,
                event_type="aiops.run.failed" if kind == "failed" else "aiops.diagnostic.input_required",
                summary="安全摘要", actor_id=job.actor_id,
                **({"hitl_id": uuid7(), "source_status": "PENDING"} if kind != "failed" else {}))
        elif kind == "started":
            await repo.emit_run_started(run=source, target_name="运维目标")
        elif kind == "situation":
            target = SimpleNamespace(target_id=source.target_id, domain_id=job.domain_id,
                tenant_id=job.tenant_id, display_name="运维目标")
            situation = SimpleNamespace(situation_id=uuid7(), severity="CRITICAL")
            resource_id, id_key = situation.situation_id, "situation_id"
            await repo.emit_situation_event(target=target, situation=situation,
                event_type="aiops.situation.detected", stage="SITUATION_DETECTED",
                summary="安全摘要", trace_id="trace")
        elif kind == "proposal":
            proposal = SimpleNamespace(proposal_id=uuid7())
            await repo.emit_proposal_event(run=source, proposal=proposal,
                event_type="aiops.proposal.review_required", summary="安全摘要", actor_id=job.actor_id)
        else:
            asset = SimpleNamespace(domain_id=job.domain_id, tenant_id=job.tenant_id, created_by=job.actor_id)
            await repo.emit_case_review_required(asset=asset,
                version=SimpleNamespace(asset_version_id=uuid7()), run=source)
        data = payload()
        assert count() == 1
        if job.tenant_id:
            assert data["route_params"]["domain_id"] == str(job.domain_id)
            assert data["route_params"][id_key] == str(resource_id)
        else:
            assert data["domain_id"] == job.domain_id
            if kind != "proposal":
                assert data["resource_id"] == str(resource_id)
            else:
                assert data["safe_data"]["run_id"] == str(source.ops_run_id)
    asyncio.run(run())


NOW = datetime(2026, 11, 3, 3, tzinfo=timezone.utc)
def test_input_notifications_require_concrete_identity_and_dedupe_per_hitl():
    async def run():
        repo, job, payload, count = adapter()
        source = SimpleNamespace(ops_run_id=uuid7(), domain_id=job.domain_id,
            tenant_id=job.tenant_id, actor_id=job.actor_id, trace_id="trace")
        with pytest.raises(ValueError):
            await repo.emit_run_event(run=source, event_type="aiops.diagnostic.input_required",
                summary="等待补证", actor_id=job.actor_id)
        assert count() == 0
        keys = []
        for hitl_id in (uuid7(), uuid7()):
            await repo.emit_run_event(run=source, event_type="aiops.diagnostic.input_required",
                summary="等待补证", actor_id=job.actor_id, hitl_id=hitl_id, source_status="PENDING")
            data = payload()
            facts = data.get("safe_payload") or data.get("safe_data")
            assert facts["hitl_id"] == str(hitl_id)
            keys.append(data["dedupe_key"] if job.tenant_id else repo._uow.notification_outbox.add.await_args.kwargs["event_key"])
        assert keys[0] != keys[1]
    asyncio.run(run())


def test_no_subscriber_does_not_erase_business_events_or_original_severity():
    async def run():
        repo, job, payload, count = adapter()
        subscriptions = SimpleNamespace(recipient_actor_ids=AsyncMock(return_value=[]))
        if job.tenant_id:
            repo._notification_subscriptions = subscriptions
        else:
            repo._uow.notification_subscriptions = subscriptions
        target = SimpleNamespace(target_id=uuid7(), domain_id=job.domain_id,
            tenant_id=job.tenant_id, display_name="运维目标")
        situation = SimpleNamespace(situation_id=uuid7(), severity="CRITICAL")
        await repo.emit_situation_event(target=target, situation=situation,
            event_type="aiops.situation.escalated", stage="SITUATION_DETECTED", summary="严重程度已升级", trace_id="trace")
        assert count() == 1
        data = payload()
        assert (data.get("safe_payload") or data.get("safe_data"))["severity"] == "CRITICAL"
        assert (data["recipient_user_ids"] if job.tenant_id else data["recipient_actor_ids"]) == []
    asyncio.run(run())



def adapter():
    tenant = "session" in inspect.signature(PlatformNotificationRepository).parameters
    job = SimpleNamespace(job_id=uuid7(), actor_id=uuid7() if tenant else "human",
        domain_id=uuid7() if tenant else 9, tenant_id=uuid7() if tenant else None,
        job_type="CASE_EXTRACTION", trace_id="trace", attempt_count=2, row_version=8,
        error_message="禁止进入通知的诊断正文和凭据")
    if tenant:
        session = SimpleNamespace(execute=AsyncMock())
        repo = PlatformNotificationRepository(session,
            notification_subscriptions=object(), situations=object())
        def payload():
            return session.execute.await_args.args[0].compile().params["payload"]
        def count():
            return session.execute.await_count
    else:
        uow = SimpleNamespace(notification_outbox=SimpleNamespace(add=AsyncMock()))
        repo = PlatformNotificationRepository(uow)
        def payload():
            return uow.notification_outbox.add.await_args.kwargs["envelope"].model_dump(mode="json")
        def count():
            return uow.notification_outbox.add.await_count
    return repo, job, payload, count


@pytest.mark.parametrize("event_type", [
    "aiops.output_job.started", "aiops.output_job.failed", "aiops.case.draft_ready",
])
def test_output_notification_is_safe_and_uses_actual_task_not_run(event_type):
    async def run():
        repo, job, payload, count = adapter()
        result = OutputJobResult(asset_id=uuid7(), asset_version_id=uuid7()) if event_type.endswith("draft_ready") else None
        await repo.emit_output_job_event(job=job, event_type=event_type, now=NOW, result=result)
        data = payload()
        assert count() == 1
        assert data["operation_id"] == data["resource_id"] == str(job.job_id)
        assert data["resource_type"] == "aiops_output_job"
        assert datetime.fromisoformat(data["occurred_at"].replace("Z", "+00:00")) == NOW
        assert "禁止进入通知" not in str(data)
        if "tenant_id" in data:
            assert data["recipient_user_ids"] == [str(job.actor_id)]
            assert data["route_key"] == "aiops.output_job.detail"
            assert data["safe_payload"]["source_version"] == 8
            assert data["route_params"]["domain_id"] == str(job.domain_id)
            assert data["route_params"]["job_type"] == "CASE_EXTRACTION"
            if result is not None:
                assert data["route_params"]["asset_version_id"] == str(result.asset_version_id)
        else:
            assert data["recipient_actor_ids"] == [job.actor_id]
            assert data["safe_data"]["job_id"] == str(job.job_id)
            assert data["safe_data"]["job_type"] == job.job_type
            assert data["safe_data"]["source_version"] == 8
            if result is not None:
                assert data["safe_data"]["asset_id"] == str(result.asset_id)
                assert data["safe_data"]["asset_version_id"] == str(result.asset_version_id)
        if result is not None:
            assert "人工审核发布后" in str(data)
    asyncio.run(run())


@pytest.mark.parametrize("event_type,result", [
    ("aiops.case.draft_ready", None),
    ("aiops.case.draft_ready", OutputJobResult(report_id=uuid7())),
    ("aiops.output_job.failed", OutputJobResult(asset_id=uuid7(), asset_version_id=uuid7())),
    ("unknown", None),
])
def test_invalid_output_event_never_reaches_notification_storage(event_type, result):
    async def run():
        repo, job, _, count = adapter()
        with pytest.raises(ValueError):
            await repo.emit_output_job_event(job=job, event_type=event_type, now=NOW, result=result)
        assert count() == 0
    asyncio.run(run())

@pytest.mark.parametrize("status", ["READY", "PARTIAL"])
@pytest.mark.parametrize("trigger", ["CHAT", "ALERT", "SCHEDULE"])
def test_report_task_notifies_explicit_human_with_exact_job_and_safe_summary(status, trigger):
    async def run():
        repo, job, payload, count = adapter()
        job.job_type, job.status = "REPORT_GENERATION", "SUCCEEDED"
        source = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7(), domain_id=job.domain_id,
            tenant_id=job.tenant_id, trigger_type=trigger, actor_id=None, trace_id="original-run",
            situation_id=None)
        report = SimpleNamespace(report_id=uuid7(), ops_run_id=source.ops_run_id, report_version=3,
            status=status, summary="禁止进入通知的完整诊断正文")
        subscriptions = SimpleNamespace(recipient_actor_ids=AsyncMock(return_value=[]))
        if job.tenant_id:
            repo._notification_subscriptions = subscriptions
        else:
            repo._uow.notification_subscriptions = subscriptions
        await repo.emit_report_ready(run=source, report=report, actor_id=job.actor_id, output_job=job, now=NOW)
        data = payload()
        assert count() == 1
        assert data["resource_id"] == str(report.report_id)
        assert data["operation_id"] == str(job.job_id)
        assert data["initiator_id" if job.tenant_id else "initiator_actor_id"] == str(job.actor_id)
        assert data["correlation_id"] == job.trace_id
        assert "禁止进入通知" not in str(data)
        assert ("缺口" in str(data)) == (status == "PARTIAL")
        assert datetime.fromisoformat(data["occurred_at"].replace("Z", "+00:00")) == NOW
        if job.tenant_id:
            assert data["recipient_user_ids"] == [str(job.actor_id)]
            assert data["route_params"]["domain_id"] == str(job.domain_id)
            assert data["route_params"]["report_version"] == "3"
            assert data["route_params"]["job_id"] == str(job.job_id)
        else:
            assert data["recipient_actor_ids"] == [job.actor_id]
            assert data["safe_data"]["job_id"] == str(job.job_id)
            assert data["safe_data"]["report_version"] == "3"
    asyncio.run(run())


@pytest.mark.parametrize("change", ["type", "domain", "actor", "run", "state", "clock"])
def test_report_task_notice_rejects_mismatched_task_without_writing(change):
    async def run():
        repo, job, _, count = adapter()
        job.job_type, job.status = "REPORT_GENERATION", "SUCCEEDED"
        source = SimpleNamespace(ops_run_id=uuid7(), domain_id=job.domain_id, tenant_id=job.tenant_id)
        report = SimpleNamespace(ops_run_id=source.ops_run_id, status="READY")
        actor, now = job.actor_id, NOW
        if change == "type": job.job_type = "CASE_EXTRACTION"
        if change == "domain": source.domain_id = uuid7() if job.tenant_id else 10
        if change == "actor": actor = uuid7() if job.tenant_id else "another"
        if change == "run": report.ops_run_id = uuid7()
        if change == "state": job.status = "FAILED"
        if change == "clock": now = None
        with pytest.raises(ValueError):
            await repo.emit_report_ready(run=source, report=report, actor_id=actor, output_job=job, now=now)
        assert count() == 0
    asyncio.run(run())
