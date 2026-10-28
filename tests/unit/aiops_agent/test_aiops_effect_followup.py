"""自动效果验证的固定窗口、授权、幂等与失败收口。"""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.changes.effect import _key
from aiops_agent.application.changes.effect_followup import (
    BusinessEffectFollowupService, schedule_effect_followup,
)
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.errors import state_conflict
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_business_effect import NOW, freeze_fixture, report


async def setup_followup():
    changes, uow, run, proposal, command = freeze_fixture()
    run.domain_id = 1
    await changes.freeze_effect_plan(
        proposal_id=proposal.proposal_id, domain_id=run.domain_id,
        actor_id=run.actor_id, command=command, trace_id="test",
    )
    receipt_id, verification_id = uuid7(), uuid7()
    receipt_body = {
        "schema_version": "USER_PROVIDED_ACTION_RESULT.v1",
        "proposal_id": str(proposal.proposal_id),
        "occurred_at": (NOW + timedelta(seconds=1)).isoformat(), "status": "EXECUTED",
    }
    verification_body = {
        "schema_version": "ACTION_VERIFICATION.v1", "proposal_id": str(proposal.proposal_id),
        "source_run_id": str(run.ops_run_id), "result_artifact_id": str(receipt_id),
        "status": "VERIFIED", "summary": "动作只读复核已完成",
    }
    sources = []
    for identity, body, trust in (
        (receipt_id, receipt_body, "USER_PROVIDED"),
        (verification_id, verification_body, "SOURCE_VERIFIED"),
    ):
        item = SimpleNamespace(
            artifact_id=identity, artifact_key=str(identity), ops_run_id=run.ops_run_id,
            payload_json=body, content_hash=sha256_json(body),
            schema_version=body["schema_version"], trust_level=trust, security_level=3,
        )
        await uow.runs.add_artifact(item)
        sources.append(item)
    receipt, verification = sources
    await schedule_effect_followup(
        uow=uow, run=run, proposal=proposal, verification_artifact=verification,
        receipt=receipt, now=NOW + timedelta(seconds=2), trace_id="test",
    )
    payload = uow.outbox.add.await_args.args[0].payload_json
    workloads = SimpleNamespace(create_workload_report=AsyncMock())
    service = BusinessEffectFollowupService(
        uow_factory=lambda: uow, change_service=changes, workload_service=workloads,
    )
    return service, changes, workloads, uow, run, proposal, payload


def test_fixed_window_respects_settle_delay_and_collection_cadence():
    async def execute():
        _, _, _, uow, _, proposal, payload = await setup_followup()
        assert datetime.fromisoformat(payload["period_start"]) == NOW + timedelta(seconds=61)
        assert datetime.fromisoformat(payload["period_end"]) == NOW + timedelta(seconds=1081)
        assert uow.outbox.add.await_args.args[0].available_at == NOW + timedelta(seconds=1141)
        assert uow.outbox.add.await_args.args[0].event_type == "OPS_BUSINESS_EFFECT_REQUESTED"
        assert _key(proposal.proposal_id, "automatic") == uow.outbox.add.await_args.args[0].idempotency_key
    asyncio.run(execute())


def test_early_or_changed_task_cannot_generate_report():
    async def execute():
        service, _, workloads, _, _, _, payload = await setup_followup()
        with pytest.raises(Exception, match="窗口尚未结束"):
            await service.execute(payload)
        changed = {**payload, "period_end": (NOW + timedelta(days=1)).isoformat()}
        with pytest.raises(Exception, match="冻结的任务不一致"):
            await service.execute(changed)
        workloads.create_workload_report.assert_not_awaited()
    asyncio.run(execute())


def test_real_evaluation_creates_report_and_automatic_delivery_is_idempotent():
    async def execute():
        service, _, workloads, uow, run, _, payload = await setup_followup()
        body = report(NOW + timedelta(minutes=2), mean=50)
        after = SimpleNamespace(
            artifact_id=uuid7(), artifact_key="after", ops_run_id=run.ops_run_id,
            payload_json=body, content_hash=sha256_json(body), schema_version=body["schema_version"],
            trust_level="SOURCE_VERIFIED", security_level=3,
        )
        await uow.runs.add_artifact(after)
        workloads.create_workload_report.return_value = SimpleNamespace(artifact_id=after.artifact_id)
        uow.runs.database_now.return_value = NOW + timedelta(hours=1)
        await service.execute(payload)
        assert uow.inspections.publish_report.await_args.args[0].result == "IMPROVED"
        view = await service._changes._effect_view(uow, run, await uow.changes.get_proposal())
        assert view.followup["status"] == "COMPLETED"
        assert view.result["attribution_strength"] == "LIMITED"
        assert view.result["execution_source"] == "USER_PROVIDED"
        assert view.report_id is not None
        await service.execute(payload)
        workloads.create_workload_report.assert_awaited_once()
        uow.inspections.publish_report.assert_awaited_once()
        arguments = workloads.create_workload_report.await_args.kwargs
        assert arguments["period_end"].isoformat() == payload["period_end"]
    asyncio.run(execute())


def test_revoked_agent_prevents_reading_but_keeps_safe_failure_audit():
    async def execute():
        service, _, workloads, uow, run, proposal, payload = await setup_followup()
        uow.agents.version_has_target.return_value = False
        with pytest.raises(Exception, match="已无此 Target"):
            await service.execute(payload)
        workloads.create_workload_report.assert_not_awaited()
        await service.fail(payload, state_conflict("内部正文不得复制"))
        artifact = await uow.runs.get_artifact_by_key(
            ops_run_id=run.ops_run_id, artifact_key=_key(proposal.proposal_id, "automatic-result"),
        )
        assert artifact.payload_json["status"] == "FAILED"
        assert "内部正文" not in str(artifact.payload_json)
        await service.fail(payload, RuntimeError("不得重复写入失败记录"))
        assert sum(call.args[0].schema_version == "BUSINESS_EFFECT_FOLLOWUP_RESULT.v1" for call in uow.runs.add_artifact.await_args_list) == 1
    asyncio.run(execute())


def test_unconfigured_consumer_must_not_silently_log_as_success():
    sink = AIOpsDomainOutboxSink(runtime_service=None, fallback=SimpleNamespace(publish=AsyncMock()))
    with pytest.raises(RuntimeError, match="服务未配置"):
        asyncio.run(sink.publish("OPS_BUSINESS_EFFECT_REQUESTED", {}))
    sink._fallback.publish.assert_not_awaited()


def test_outbox_dispatch_and_terminal_failure_use_real_effect_service():
    async def execute():
        effect = SimpleNamespace(execute=AsyncMock(), fail=AsyncMock())
        sink = AIOpsDomainOutboxSink(
            runtime_service=None, fallback=SimpleNamespace(publish=AsyncMock()),
            business_effect_service=effect,
        )
        payload = {"proposal_id": str(uuid7())}
        error = RuntimeError("内部原文")
        await sink.publish("OPS_BUSINESS_EFFECT_REQUESTED", payload)
        await sink.on_terminal_failure("OPS_BUSINESS_EFFECT_REQUESTED", payload, error)
        effect.execute.assert_awaited_once_with(payload)
        effect.fail.assert_awaited_once_with(payload, error)
        sink._fallback.publish.assert_not_awaited()
    asyncio.run(execute())


def test_disabled_collection_requires_explicit_manual_verification():
    async def execute():
        changes, uow, run, proposal, command = freeze_fixture()
        uow.targets.get_scoped.return_value.workload_snapshot_policy_json = {"enabled": False}
        kwargs = dict(proposal_id=proposal.proposal_id, domain_id=run.domain_id, actor_id=run.actor_id, trace_id="test")
        with pytest.raises(Exception, match="不会自动开启采集"):
            await changes.freeze_effect_plan(command=command, **kwargs)
        view = await changes.freeze_effect_plan(command=command.model_copy(update={"auto_evaluate": False}), **kwargs)
        assert view.frozen_plan["business_effect"]["auto_evaluate"] is False
        assert view.followup == {}
    asyncio.run(execute())
