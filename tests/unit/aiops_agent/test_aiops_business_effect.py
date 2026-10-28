"""业务效果必须来自执行前冻结的条件及可复算窗口，而不是动作回执。"""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.changes.service import AIOpsChangeService
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.effect_measurement import evaluate_effect, measurement, validate_window_report
from aiops_agent.application.reporting import render_pdf, report_presentation, resolve_system_template
from platform_core.contracts.aiops.effect import EffectEvaluationCommand, EffectPlanCommand, EffectSignalRule
from platform_core.identity import uuid7


NOW = datetime(2026, 10, 28, 8, tzinfo=UTC)
TARGET = str(uuid7())
IDENTITY = {"statement_identity_type": "POSTGRESQL_QUERY_ID", "statement_identity_value": "1285382359565720997", "database_name": "mes", "user_identifier": "app", "top_level": True}


def report(start, mean=100, calls=1000):
    payload = {"schema_version": "AIOPS_WORKLOAD_REPORT.v2", "target_id": TARGET, "status": "READY", "database_type": "POSTGRESQL", "period_start": start.isoformat(), "period_end": (start + timedelta(minutes=15)).isoformat(), "continuous_seconds": 900, "snapshot_count": 16, "instance_identity": {"identity_type": "POSTGRESQL_SYSTEM_IDENTIFIER", "identity_value": "9876543210123456789"}, "instance_started_at": "2026-10-01T00:00:00Z", "database_versions": ["16"], "catalog_versions": ["catalog-v1"], "capability_probe_versions": ["probe-v1"], "load_dimension_semantics": {"transactions_per_second": {"value_kind": "COUNTER_RATE", "unit": "transactions/s", "business_volume": True}}, "load_profile": {"transactions_per_second": calls / 900}, "top_statements": [{**IDENTITY, "execution_count": calls, "total_duration_microseconds": calls * mean, "rows_processed": calls * 2}]}
    payload["signal_coverage_version"] = "signal-coverage.v1"
    payload["top_statements"][0]["covered_seconds"] = 900
    payload["load_dimension_semantics"]["transactions_per_second"]["covered_seconds"] = 900
    return payload


def signals():
    return [
        {"key": "latency", "role": "PRIMARY", "metric": "statement.mean_duration_microseconds", "statement_identity": IDENTITY, "comparison": "DECREASE_BY", "threshold": "0.2"},
        {"key": "rows", "role": "GUARDRAIL", "metric": "statement.rows_per_call", "statement_identity": IDENTITY, "comparison": "AT_LEAST", "threshold": "2"},
        {"key": "load", "role": "LOAD", "metric": "load.transactions_per_second", "comparison": "STABLE_WITHIN", "threshold": "0.1"},
    ]


def evaluate(before=None, after=None, **kwargs):
    return evaluate_effect(frozen={"signals": signals(), "settle_delay_seconds": 60, "after_window_seconds": 900, "max_baseline_age_seconds": 3600}, baseline=before or report(NOW - timedelta(minutes=20)), after=after or report(NOW + timedelta(minutes=2), mean=50), occurred_at=NOW, direct_status=kwargs.get("direct_status", "VERIFIED"), controlled_execution=kwargs.get("controlled_execution", True))


def test_real_window_improvement_is_not_causal_proof():
    result = evaluate()
    assert result["measured_effect"] == "IMPROVED"
    assert result["attribution_strength"] == "LIMITED"
    assert result["guardrails_passed"] is True
    assert result["symptom_recovered"] is None
    assert result["measurements"]["latency"]["after"] == "50"


@pytest.mark.parametrize("change,expected", [
    ({"instance_identity": {"identity_type": "POSTGRESQL_SYSTEM_IDENTIFIER", "identity_value": "different"}}, "EFFECT_IDENTITY_OR_SOURCE_CHANGED"),
    ({"load_profile": {"transactions_per_second": 0.1}}, "EFFECT_WORKLOAD_NOT_COMPARABLE"),
    ({"database_versions": ["17"]}, "EFFECT_IDENTITY_OR_SOURCE_CHANGED"),
    ({"instance_started_at": "2026-10-28T00:00:00Z"}, "EFFECT_IDENTITY_OR_SOURCE_CHANGED"),
])
def test_confounders_do_not_become_success(change, expected):
    after = report(NOW + timedelta(minutes=2), mean=50)
    after.update(change)
    result = evaluate(after=after)
    assert result["measured_effect"] == "INCONCLUSIVE"
    assert expected in result["gap_codes"]


def test_cumulative_max_is_never_differenced():
    rule = EffectSignalRule(**{**signals()[0], "metric": "statement.max_duration_microseconds"})
    with pytest.raises(ValueError):
        measurement(report(NOW), rule)


def test_missing_statement_is_not_zero_or_improved():
    after = report(NOW + timedelta(minutes=2))
    after["top_statements"] = []
    result = evaluate(after=after)
    assert result["measured_effect"] == "INCONCLUSIVE"
    assert result["measurements"]["latency"]["passed"] is None


def test_guardrail_regression_is_degraded():
    after = report(NOW + timedelta(minutes=2), mean=50)
    after["top_statements"][0]["rows_processed"] = 1
    assert evaluate(after=after)["measured_effect"] == "DEGRADED"


def test_unrelated_signal_gaps_do_not_invalidate_complete_statement_and_load():
    after = report(NOW + timedelta(minutes=2), mean=50)
    after.update(statement_gaps={"EVICTED_OR_INACTIVE": 1}, metric_gaps={"MISSING_DIMENSION": 3})
    assert evaluate(after=after)["measured_effect"] == "IMPROVED"


@pytest.mark.parametrize("signal", ["statement", "load"])
def test_selected_signal_partial_coverage_blocks_positive_effect(signal):
    after = report(NOW + timedelta(minutes=2), mean=50)
    subject = after["top_statements"][0] if signal == "statement" else after["load_dimension_semantics"]["transactions_per_second"]
    subject["covered_seconds"] = 899
    assert evaluate(after=after)["measured_effect"] == "INCONCLUSIVE"


def test_manual_action_is_not_agent_execution():
    result = evaluate(controlled_execution=False)
    assert result["execution_source"] == "USER_PROVIDED"
    assert result["attribution_strength"] != "SUPPORTED"


@pytest.mark.parametrize("key,value", [("snapshot_count", 1), ("continuous_seconds", 60), ("signal_coverage_version", "unknown"), ("discontinuities", [{"reason": "RESET"}])])
def test_partial_statistics_are_not_a_comparable_window(key, value):
    payload = report(NOW)
    payload[key] = value
    with pytest.raises(ValueError):
        validate_window_report(payload)


def test_exact_mean_uses_sum_and_count_not_average_of_averages():
    payload = report(NOW)
    payload["top_statements"][0].update(execution_count=3, total_duration_microseconds=9007199254740993, mean_duration_microseconds=0)
    value, _, _ = measurement(payload, EffectSignalRule(**signals()[0]))
    assert str(value) == "3002399751580331"


def test_freeze_requires_load_primary_and_guardrail():
    with pytest.raises(ValueError):
        EffectPlanCommand(expected_row_version=1, baseline_artifact_id=uuid7(), signals=signals()[:2], stop_conditions=["护栏退化停止"])


class Uow:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def freeze_fixture():
    run = SimpleNamespace(ops_run_id=uuid7(), domain_id=uuid7(), target_id=TARGET, agent_id=uuid7(), actor_id="owner", plan_snapshot_json={"target": {"row_version": 1}})
    proposal = SimpleNamespace(proposal_id=uuid7(), ops_run_id=run.ops_run_id, ops_task_id=uuid7(), expires_at=NOW + timedelta(hours=1), row_version=1, status="ADVISORY_READY", solution_group_key="sql", action_template_id="manual.sql", action_template_version="1")
    payload = report(NOW - timedelta(minutes=20))
    baseline = SimpleNamespace(artifact_id=uuid7(), ops_run_id=run.ops_run_id, payload_json=payload, content_hash=sha256_json(payload), schema_version="AIOPS_WORKLOAD_REPORT.v2", trust_level="SOURCE_VERIFIED", security_level=3)
    artifacts = {str(baseline.artifact_id): baseline}
    by_key = {}
    async def add_artifact(entity):
        artifacts[str(entity.artifact_id)] = entity
        by_key[entity.artifact_key] = entity
        return entity
    uow = Uow()
    uow.runs = SimpleNamespace(get_run=AsyncMock(return_value=run), get_run_scoped=AsyncMock(return_value=run), get_artifact=AsyncMock(side_effect=lambda artifact_id: artifacts.get(str(artifact_id))), get_artifact_by_key=AsyncMock(side_effect=lambda ops_run_id, artifact_key: by_key.get(artifact_key)), get_latest_effect_result=AsyncMock(return_value=None), add_artifact=AsyncMock(side_effect=add_artifact), database_now=AsyncMock(return_value=NOW), append_event=AsyncMock())
    uow.changes = SimpleNamespace(get_proposal_scoped=AsyncMock(return_value=proposal), get_proposal=AsyncMock(return_value=proposal))
    uow.agents = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(current_version_id=uuid7())), version_has_target=AsyncMock(return_value=True))
    uow.targets = SimpleNamespace(get_scoped=AsyncMock(return_value=SimpleNamespace(status="ENABLED", row_version=1, workload_snapshot_policy_json={"enabled": True, "interval_minutes": 1})))
    uow.commit = AsyncMock()
    reports = {}
    async def publish_report(entity):
        reports[entity.report_key] = entity
        return entity
    uow.inspections = SimpleNamespace(get_current_effect_report=AsyncMock(side_effect=lambda ops_run_id, report_key: reports.get(report_key)), publish_report=AsyncMock(side_effect=publish_report), add_report_sources=AsyncMock())
    uow.outbox = SimpleNamespace(add=AsyncMock())
    uow.platform_notifications = SimpleNamespace(emit_report_ready=AsyncMock())
    uow.runs.get_latest_effect_result.side_effect = lambda ops_run_id, proposal_id: next((value for value in reversed(tuple(by_key.values())) if value.schema_version == "COMPARISON_RESULT.v1"), None)
    service = AIOpsChangeService(uow_factory=lambda: uow)
    service._snapshot = AsyncMock(return_value=SimpleNamespace(verification_plan=("db.sql.object_indexes",)))
    command = EffectPlanCommand(expected_row_version=1, baseline_artifact_id=baseline.artifact_id, signals=signals(), stop_conditions=["护栏退化时停止并回退"])
    return service, uow, run, proposal, command


def test_freeze_persists_real_baseline_and_is_immutable():
    service, uow, run, proposal, command = freeze_fixture()
    async def execute():
        kwargs = dict(proposal_id=proposal.proposal_id, domain_id=run.domain_id, actor_id="owner", trace_id="test")
        result = await service.freeze_effect_plan(command=command, **kwargs)
        assert result.frozen_plan["business_effect"]["baseline_hash"]
        assert result.plan_artifact is not None
        await service.freeze_effect_plan(command=command, **kwargs)
        uow.runs.add_artifact.assert_awaited_once()
        with pytest.raises(Exception, match="效果计划已冻结"):
            await service.freeze_effect_plan(command=command.model_copy(update={"settle_delay_seconds": 120}), **kwargs)
    asyncio.run(execute())


@pytest.mark.parametrize("status", ["APPROVED", "EXECUTING", "SUCCEEDED"])
def test_cannot_select_success_conditions_after_execution(status):
    service, uow, run, proposal, command = freeze_fixture()
    proposal.status = status
    with pytest.raises(Exception, match="执行或审批前"):
        asyncio.run(service.freeze_effect_plan(proposal_id=proposal.proposal_id, domain_id=run.domain_id, actor_id="owner", command=command, trace_id="test"))
    uow.runs.add_artifact.assert_not_awaited()


def test_cross_user_cannot_read_effect_evidence():
    service, uow, run, proposal, command = freeze_fixture()
    with pytest.raises(Exception, match="不存在"):
        asyncio.run(service.get_effect_verification(proposal_id=proposal.proposal_id, domain_id=run.domain_id, actor_id="other"))


def test_effect_recomputes_native_evidence_publishes_report_and_keeps_manual_trust():
    service, uow, run, proposal, command = freeze_fixture()
    async def execute():
        kwargs = dict(proposal_id=proposal.proposal_id, domain_id=run.domain_id, actor_id="owner", trace_id="test")
        await service.freeze_effect_plan(command=command, **kwargs)
        receipt_id, verification_id, after_id = uuid7(), uuid7(), uuid7()
        receipt_payload = {"schema_version": "USER_PROVIDED_ACTION_RESULT.v1", "proposal_id": str(proposal.proposal_id), "occurred_at": (NOW + timedelta(seconds=1)).isoformat(), "status": "EXECUTED"}
        verification_payload = {"schema_version": "ACTION_VERIFICATION.v1", "proposal_id": str(proposal.proposal_id), "source_run_id": str(run.ops_run_id), "result_artifact_id": str(receipt_id), "status": "VERIFIED", "summary": "已只读复核后置条件"}
        after = report(NOW + timedelta(minutes=2), mean=50)
        for identity, payload, trust in ((receipt_id, receipt_payload, "USER_PROVIDED"), (verification_id, verification_payload, "SOURCE_VERIFIED"), (after_id, after, "SOURCE_VERIFIED")):
            await uow.runs.add_artifact(SimpleNamespace(artifact_id=identity, ops_run_id=run.ops_run_id, artifact_key=str(identity), artifact_type=payload["schema_version"].split(".")[0], schema_version=payload["schema_version"], content_hash=sha256_json(payload), payload_json=payload, trust_level=trust, security_level=3))
        uow.runs.database_now.return_value = NOW + timedelta(hours=1)
        evaluate_command = EffectEvaluationCommand(expected_row_version=1, after_artifact_id=after_id, verification_artifact_id=verification_id)
        result = await service.evaluate_effect_plan(command=evaluate_command, **kwargs)
        assert result.result["measured_effect"] == "IMPROVED"
        assert result.result["attribution_strength"] == "LIMITED"
        assert result.report_id is not None
        assert result.result["source_result_artifact_id"] == str(receipt_id)
        report_entity = uow.inspections.publish_report.await_args.args[0]
        assert report_entity.result == "IMPROVED"
        assert report_entity.security_level == 3
        assert len(uow.inspections.add_report_sources.await_args.args[0]) == 6
        report_artifact = await uow.runs.get_artifact(artifact_id=report_entity.content_artifact_id)
        assert report_artifact.payload_json["provenance"]["template"]["definition"]
        presentation = report_presentation(payload=report_artifact.payload_json, template=resolve_system_template("system:comparison.business"))
        sections = {item["kind"]: item for item in presentation["sections"]}
        assert any("基线 100" in value and "处置后 50" in value for value in sections["FINDINGS"]["items"])
        assert len(sections["EVIDENCE_APPENDIX"]["items"]) == 6
        assert render_pdf(presentation).startswith(b"%PDF-")
        await service.evaluate_effect_plan(command=evaluate_command, **kwargs)
        uow.inspections.publish_report.assert_awaited_once()
        refreshed = await service._effect_view(uow, run, proposal)
        assert refreshed.report_id == result.report_id
        uow.platform_notifications.emit_report_ready.assert_awaited_once()
        uow.outbox.add.assert_awaited_once()
        after["top_statements"][0]["execution_count"] = 1
        with pytest.raises(Exception, match="来源链已经改变|完整性"):
            await service.evaluate_effect_plan(command=evaluate_command, **kwargs)
    asyncio.run(execute())


def test_primary_regression_is_not_reported_unchanged():
    assert evaluate(after=report(NOW + timedelta(minutes=2), mean=150))["measured_effect"] == "DEGRADED"


def test_adverse_observation_is_not_erased_by_missing_evidence():
    after = report(NOW + timedelta(minutes=2), mean=50)
    after["top_statements"] = []
    assert evaluate(after=after, direct_status="ADVERSE")["result"] == "DEGRADED"


def test_guardrail_failure_is_visible_even_with_load_gap():
    after = report(NOW + timedelta(minutes=2), mean=50)
    after["top_statements"][0]["rows_processed"] = 1
    after["load_profile"] = {}
    value = evaluate(after=after)
    assert value["result"] == "DEGRADED"
    assert "EFFECT_SIGNAL_UNAVAILABLE" in value["gap_codes"]


def test_machine_agent_whitelist_cannot_be_bypassed_by_matching_owner():
    service, uow, run, proposal, command = freeze_fixture()
    with pytest.raises(Exception, match="不存在"):
        asyncio.run(service.get_effect_verification(proposal_id=proposal.proposal_id, domain_id=run.domain_id, actor_id="owner", agent_ids=(uuid7(),)))


def test_source_scope_removed_from_current_agent_cannot_reuse_old_artifact(monkeypatch):
    from aiops_agent.application.changes import effect
    _, uow, run, _, command = freeze_fixture()
    source_id = uuid7()
    run.plan_snapshot_json["monitoring"] = {"bindings": [{"source": {"source_id": str(source_id)}}]}
    monkeypatch.setattr(effect, "source_scope_is_current", AsyncMock(return_value=True))
    uow.agents.version_source_ids = AsyncMock(return_value=())
    with pytest.raises(Exception, match="已无效果证据所使用的来源权限"):
        asyncio.run(effect.effect_source(uow=uow, artifact_id=command.baseline_artifact_id, run=run))
    uow.agents.version_source_ids.return_value = (source_id,)
    assert asyncio.run(effect.effect_source(uow=uow, artifact_id=command.baseline_artifact_id, run=run)) is not None


def test_cross_agent_window_requires_owner_and_frozen_target_version():
    from aiops_agent.application.changes.effect import effect_source
    _, uow, run, _, command = freeze_fixture()
    source_run = deepcopy(run)
    source_run.agent_id = uuid7()
    source_run.workflow_kind = "WORKLOAD_REPORT"
    uow.runs.get_run_scoped.return_value = source_run
    assert asyncio.run(effect_source(uow=uow, artifact_id=command.baseline_artifact_id, run=run)) is not None
    source_run.actor_id = "another-owner"
    with pytest.raises(Exception, match="不存在"):
        asyncio.run(effect_source(uow=uow, artifact_id=command.baseline_artifact_id, run=run))
    source_run.actor_id = run.actor_id
    source_run.plan_snapshot_json = {"target": {"row_version": 2}}
    with pytest.raises(Exception, match="Target 配置已变化"):
        asyncio.run(effect_source(uow=uow, artifact_id=command.baseline_artifact_id, run=run))
