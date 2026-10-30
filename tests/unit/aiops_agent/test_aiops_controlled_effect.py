"""匹配对照的正向生产、混杂反例及报告/案例原始来源重放。"""

import asyncio
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.case_source_validation import validate_case_effect
from aiops_agent.application.changes.effect_replay import replay_supported_effect
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.control_measurement import validate_control_baselines
from aiops_agent.application.effect_measurement import evaluate_effect
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from platform_core.contracts.aiops.effect import EffectControlPlan, EffectEvaluationCommand, EffectPlanCommand
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_business_effect import NOW, TARGET, freeze_fixture, report, signals


def comparison_fixture():
    control_target = uuid7()
    baseline_id, control_baseline_id, control_after_id = (str(uuid7()) for _ in range(3))
    def control(start, mean=100, calls=1000):
        value = report(start, mean=mean, calls=calls)
        value["target_id"] = str(control_target)
        value["instance_identity"]["identity_value"] = "separate-control-system"
        return value
    pairs, reports = [], {}
    for minutes in (50, 35):
        ti, ci = str(uuid7()), str(uuid7())
        reports[ti] = report(NOW - timedelta(minutes=minutes))
        reports[ci] = control(NOW - timedelta(minutes=minutes))
        pairs.append({"treatment_artifact_id": ti, "control_artifact_id": ci})
    reports[control_baseline_id] = control(NOW - timedelta(minutes=20))
    reports[control_after_id] = control(NOW + timedelta(minutes=2))
    frozen = {"baseline_artifact_id": baseline_id, "signals": signals(),
              "settle_delay_seconds": 60, "after_window_seconds": 900, "max_baseline_age_seconds": 3600,
              "controlled_comparison": {"target_id": str(control_target), "baseline_artifact_id": control_baseline_id,
                                        "prior_pairs": pairs, "signals": signals()}}
    baseline = report(NOW - timedelta(minutes=20))
    after = report(NOW + timedelta(minutes=2), mean=50)
    return frozen, baseline, after, reports, control_after_id


def evaluate(frozen, baseline, after, reports, control_after_id, **kwargs):
    return evaluate_effect(
        frozen=frozen, baseline=baseline, after=after, occurred_at=NOW,
        direct_status=kwargs.get("direct_status", "VERIFIED"),
        controlled_execution=kwargs.get("controlled_execution", True),
        control_reports=reports, control_after_ref=control_after_id,
    )


def test_matched_control_produces_supported_effect_but_not_root_cause_proof():
    result = evaluate(*comparison_fixture())
    assert result["measured_effect"] == "IMPROVED"
    assert result["attribution_strength"] == "SUPPORTED"
    assert result["control_analysis"]["signals"]["latency"]["adjusted_relative_change"] == "-0.5"
    assert not result["gap_codes"]
    assert "不等于随机试验或已确认根因" in result["causal_limitations"][0]


@pytest.mark.parametrize("change", ["shared_improvement", "load_drop", "restarted", "source_changed", "misaligned", "same_physical", "missing", "too_few_calls", "guardrail"])
def test_control_counterexamples_cannot_gain_supported_attribution(change):
    frozen, baseline, after, reports, aid = comparison_fixture()
    item = reports[aid]
    if change == "shared_improvement":
        item["top_statements"][0]["total_duration_microseconds"] //= 2
    elif change == "load_drop":
        item["load_totals"]["transactions"] //= 2
    elif change == "restarted":
        item["instance_started_at"] = NOW.isoformat()
    elif change == "source_changed":
        item["catalog_versions"] = ["new-catalog"]
    elif change == "misaligned":
        item["period_start"] = (NOW + timedelta(minutes=3)).isoformat()
        item["period_end"] = (NOW + timedelta(minutes=18)).isoformat()
    elif change == "same_physical":
        item["instance_identity"] = deepcopy(after["instance_identity"])
    elif change == "missing":
        reports.pop(aid)
    elif change == "too_few_calls":
        item["top_statements"][0]["execution_count"] = 2
    else:
        item["top_statements"][0]["rows_processed"] = 1
    assert evaluate(frozen, baseline, after, reports, aid)["attribution_strength"] != "SUPPORTED"


@pytest.mark.parametrize("change", ["trend", "overlap", "old", "insufficient", "wrong_subject", "negative", "different_unit"])
def test_execution_cannot_freeze_a_bad_preperiod(change):
    frozen, baseline, _, reports, _ = comparison_fixture()
    pair = frozen["controlled_comparison"]["prior_pairs"][0]
    item = reports[pair["treatment_artifact_id"]]
    if change == "trend":
        item["top_statements"][0]["total_duration_microseconds"] *= 2
    elif change == "overlap":
        item["period_start"] = baseline["period_start"]
        item["period_end"] = baseline["period_end"]
    elif change == "old":
        frozen["max_baseline_age_seconds"] = 60
    elif change == "insufficient":
        item["top_statements"][0]["execution_count"] = 1
    elif change == "wrong_subject":
        item["top_statements"][0]["database_name"] = "other"
    elif change == "negative":
        item["top_statements"][0]["total_duration_microseconds"] = -1000
    else:
        item["load_dimension_semantics"]["transactions_per_second"]["unit"] = "bytes/s"
    with pytest.raises(ValueError):
        validate_control_baselines(frozen=frozen, baseline=baseline, reports=reports)


@pytest.mark.parametrize("kwargs", [{"controlled_execution": False}, {"direct_status": "INCONCLUSIVE"}, {"direct_status": "NOT_ACHIEVED"}])
def test_matched_control_does_not_upgrade_manual_or_unverified_actions(kwargs):
    assert evaluate(*comparison_fixture(), **kwargs)["attribution_strength"] != "SUPPORTED"


def test_control_contract_cannot_change_conditions_or_accept_observed_values():
    frozen, _, _, _, _ = comparison_fixture()
    raw = {**frozen, "expected_row_version": 1, "stop_conditions": ["护栏退化停止"]}
    control = raw["controlled_comparison"]
    EffectPlanCommand.model_validate(raw)
    control["signals"][0]["threshold"] = "0.01"
    with pytest.raises(ValueError):
        EffectPlanCommand.model_validate(raw)
    control["signals"] = signals()
    control["observed_improvement"] = "0.5"
    with pytest.raises(ValueError):
        EffectPlanCommand.model_validate(raw)


async def service_fixture(*, with_control=True):
    service, uow, run, proposal, command = freeze_fixture()
    from sqlalchemy import Numeric
    from aiops_agent.entities import OpsRunEntity
    # 使用各平台真实持久化合同构造授权域，不把 UUID 塞入 Oracle 的数字 Domain。
    if isinstance(OpsRunEntity.__table__.c.domain_id.type, Numeric):
        run.domain_id = 1
    run.tenant_id = uuid7()
    frozen, _, after, reports, control_after_id = comparison_fixture()
    control = EffectControlPlan.model_validate(frozen["controlled_comparison"])
    if with_control:
        command = command.model_copy(update={"controlled_comparison": control})
    control_run = SimpleNamespace(
        ops_run_id=uuid7(), domain_id=run.domain_id, target_id=control.target_id,
        agent_id=run.agent_id, actor_id=run.actor_id, workflow_kind="WORKLOAD_REPORT",
        plan_snapshot_json={"target": {"row_version": 1}},
    )
    uow.runs.get_run_scoped.side_effect = lambda ops_run_id, domain_id: control_run if str(ops_run_id) == str(control_run.ops_run_id) else run
    for identity, body in reports.items():
        owner = control_run if body["target_id"] != TARGET else run
        await uow.runs.add_artifact(SimpleNamespace(
            artifact_id=identity, ops_run_id=owner.ops_run_id, artifact_key=identity,
            schema_version=body["schema_version"], payload_json=body, content_hash=sha256_json(body),
            trust_level="SOURCE_VERIFIED", security_level=4 if owner is control_run else 3,
        ))
    kwargs = dict(proposal_id=proposal.proposal_id, domain_id=run.domain_id, actor_id=run.actor_id, trace_id="test-control")
    await service.freeze_effect_plan(command=command, **kwargs)
    receipt_id, verification_id, after_id = uuid7(), uuid7(), uuid7()
    execution_id = uuid7()
    receipt = {"schema_version": "EXECUTION_RESULT.v1", "proposal_id": str(proposal.proposal_id),
               "execution_id": str(execution_id), "executor_request_id": str(uuid7()), "executor_instance_id": "executor",
               "status": "SUCCEEDED", "status_version": 4, "occurred_at": (NOW + timedelta(seconds=1)).isoformat(),
               "result_hash": "a" * 64, "proposal_hash": "b" * 64, "command_hash": "c" * 64, "grant_jti_hash": "d" * 64}
    verification = {"schema_version": "ACTION_VERIFICATION.v1", "proposal_id": str(proposal.proposal_id),
                    "source_run_id": str(run.ops_run_id), "result_artifact_id": str(receipt_id),
                    "status": "VERIFIED", "summary": "只读复核通过"}
    for identity, body in ((receipt_id, receipt), (verification_id, verification), (after_id, after)):
        await uow.runs.add_artifact(SimpleNamespace(
            artifact_id=identity, ops_run_id=run.ops_run_id, artifact_key=str(identity),
            schema_version=body["schema_version"], payload_json=body, content_hash=sha256_json(body),
            trust_level="SOURCE_VERIFIED", security_level=3,
        ))
    uow.changes.get_execution = AsyncMock(return_value=SimpleNamespace(
        proposal_id=proposal.proposal_id, result_artifact_id=receipt_id,
        status="SUCCEEDED", proposal_hash="b" * 64, command_hash="c" * 64,
    ))
    uow.runs.database_now.return_value = NOW + timedelta(hours=1)
    command = EffectEvaluationCommand(expected_row_version=1, after_artifact_id=after_id,
                                     verification_artifact_id=verification_id, control_after_artifact_id=control_after_id if with_control else None)
    return service, uow, run, proposal, command, kwargs


def test_service_freezes_control_hashes_then_publishes_and_replays_supported_result():
    async def execute():
        service, uow, run, proposal, command, kwargs = await service_fixture()
        result = await service.evaluate_effect_plan(command=command, **kwargs)
        assert result.result["attribution_strength"] == "SUPPORTED"
        assert len(result.result["measurement_artifact_refs"]) == 11
        assert result.frozen_plan["business_effect"]["control_target_row_version"] == 1
        report_entity = uow.inspections.publish_report.await_args.args[0]
        assert report_entity.security_level == 4
        assert "匹配对照支持" in report_entity.summary
        artifact = await uow.runs.get_artifact(artifact_id=result.comparison_artifact.artifact_id)
        assert await replay_supported_effect(uow=uow, run=run, artifact=artifact) is not None
        # 已有结果的来源回读也不能覆盖用户请求中的目标筛选，绕过冻结约束。
        with pytest.raises(Exception, match="冻结后替换"):
            await service.get_effect_verification(**{key: value for key, value in kwargs.items() if key != "trace_id"},
                                                  control_target_id=uuid7())
        profile = DiagnosisCaseProfile(
            case_kind="VERIFIED_OPTIMIZATION", problem_signature={}, environment={}, root_cause={},
            actions=({"summary": "已执行动作", "evidence_refs": ["receipt"]},),
            verification={"controlled_verification_ref": f"artifact:{artifact.artifact_id}", "comparability": "COMPARABLE",
                          "guardrails_passed": True, "measured_effect": "IMPROVED"},
            source={"run_ids": [str(run.ops_run_id)]},
        )
        assert await validate_case_effect(uow=uow, domain_id=run.domain_id, profile=profile)
        profile.case_kind = "VERIFIED_RESOLUTION"
        profile.verification["symptom_recovered"] = True
        assert not await validate_case_effect(uow=uow, domain_id=run.domain_id, profile=profile)
        uow.changes.get_execution.return_value.status = "FAILED"
        assert await replay_supported_effect(uow=uow, run=run, artifact=artifact) is None
    asyncio.run(execute())


@pytest.mark.parametrize("failure", ["revoked", "hash", "control_target_version", "wrong_owner", "result_tamper", "receipt_link"])
def test_result_replay_rechecks_source_authority_not_profile_flags(failure):
    async def execute():
        service, uow, run, _, command, kwargs = await service_fixture()
        view = await service.evaluate_effect_plan(command=command, **kwargs)
        artifact = await uow.runs.get_artifact(artifact_id=view.comparison_artifact.artifact_id)
        if failure == "revoked":
            uow.agents.version_has_target.return_value = False
        elif failure == "hash":
            ref = view.result["control_analysis"]["after_artifact_id"]
            source = await uow.runs.get_artifact(artifact_id=ref)
            source.payload_json["snapshot_count"] = 0
        elif failure == "control_target_version":
            uow.targets.get_scoped.return_value.row_version = 2
        elif failure == "wrong_owner":
            control_run = await uow.runs.get_run_scoped(ops_run_id=(await uow.runs.get_artifact(artifact_id=command.control_after_artifact_id)).ops_run_id, domain_id=run.domain_id)
            control_run.actor_id = "someone-else"
        elif failure == "result_tamper":
            artifact.payload_json["control_analysis"]["signals"]["latency"]["adjusted_relative_change"] = "-0.9"
            artifact.content_hash = sha256_json(artifact.payload_json)
        else:
            receipt = await uow.runs.get_artifact(artifact_id=artifact.payload_json["source_result_artifact_id"])
            receipt.payload_json["proposal_id"] = str(uuid7())
            receipt.content_hash = sha256_json(receipt.payload_json)
        assert await replay_supported_effect(uow=uow, run=run, artifact=artifact) is None
    asyncio.run(execute())


def test_supported_report_produces_a_verified_candidate_without_confirming_root_cause():
    from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
    from aiops_agent.application.effect_case import profile_from_effect_report
    from tests.unit.aiops_agent.test_aiops_investigation_record import draft, evidence
    from tests.unit.aiops_agent.test_aiops_turn_answer import _proposal_artifact
    async def execute():
        service, uow, run, proposal, command, kwargs = await service_fixture()
        context = SimpleNamespace(run_id=run.ops_run_id, task_id=proposal.ops_task_id,
                                  target_id=run.target_id, plan_snapshot={"answer_context": {"question": "调查业务阻塞"}})
        fact = evidence()
        record = freeze_investigation(draft=draft(), evidence=(fact,), context=context)
        root = {"schema_version": "AIOPS_TURN_RESULT.v1", "status": "COMPLETED",
                "investigation_record": record.model_dump(mode="json"), "evidence": [fact.model_dump(mode="json")]}
        snapshot = _proposal_artifact(mode="AGENT_EXECUTE")["payload"]
        snapshot["proposal"].update(proposal_id=str(proposal.proposal_id), run_id=str(run.ops_run_id),
                                    target_id=str(run.target_id), proposal_hash="b" * 64, command_hash="c" * 64)
        run.final_artifact_id, proposal.snapshot_artifact_id = uuid7(), uuid7()
        for identity, body in ((run.final_artifact_id, root), (proposal.snapshot_artifact_id, snapshot)):
            await uow.runs.add_artifact(SimpleNamespace(
                artifact_id=identity, ops_run_id=run.ops_run_id, artifact_key=str(identity),
                schema_version=body["schema_version"], payload_json=body,
                content_hash=sha256_json(body), trust_level="SOURCE_VERIFIED", security_level=4,
            ))
        uow.targets.get_scoped.return_value.db_type = "POSTGRESQL"
        uow.targets.get_scoped.return_value.version_code = "16"
        uow.targets.get_scoped.return_value.environment = "TEST"
        uow.targets.get_scoped.return_value.capabilities_json = {}
        await service.evaluate_effect_plan(command=command, **kwargs)
        published = uow.inspections.publish_report.await_args.args[0]
        content = await uow.runs.get_artifact(artifact_id=published.content_artifact_id)
        sources = uow.inspections.add_report_sources.await_args.args[0]
        published.is_current = True
        profile = await profile_from_effect_report(uow=uow, domain_id=run.domain_id,
                                                  report=published, content=content.payload_json, sources=sources)
        assert profile.case_kind == "VERIFIED_OPTIMIZATION"
        assert profile.root_cause["effective_level"] == "INCONCLUSIVE"
        assert profile.verification["method"] == "MATCHED_CONTROL_RATIO.v1"
        assert await validate_case_effect(uow=uow, domain_id=run.domain_id, profile=profile)
        assert set(profile.verification["before_refs"]).isdisjoint(profile.verification["after_refs"])
        assert len(profile.verification["after_refs"]) == 2
        # 人工改写动作或根因会通过既有修订入口降为参考，不继承已验证等级。
        from aiops_agent.application.operations_knowledge import _prepare_case_revision
        from aiops_agent.contracts.knowledge import DiagnosisCaseDraft
        edited = profile.model_copy(update={"root_cause": {**profile.root_cause, "summary": "人工改写的其他原因"}})
        revised = _prepare_case_revision(command=DiagnosisCaseDraft(expected_asset_row_version=1,
            base_version_id=uuid7(), display_name="修改后的案例", profile=edited), base_profile=profile)
        assert revised.case_kind == "DIAGNOSTIC_REFERENCE"
    asyncio.run(execute())


def test_control_candidates_are_bounded_target_scoped_and_never_expand_authority():
    async def execute():
        service, uow, run, _, _, kwargs = await service_fixture()
        control_id = next(item.args[0].payload_json["business_effect"]["controlled_comparison"]["target_id"]
                          for item in uow.runs.add_artifact.await_args_list if item.args[0].schema_version == "COMPARISON_PLAN.v1")
        artifacts = [item.args[0] for item in uow.runs.add_artifact.await_args_list
                     if item.args[0].schema_version == "AIOPS_WORKLOAD_REPORT.v2"]
        uow.runs.list_effect_artifacts = AsyncMock(side_effect=lambda **scope: [
            item for item in artifacts if str(item.payload_json["target_id"]) == str(scope["target_id"])
        ])
        request = {key: value for key, value in kwargs.items() if key != "trace_id"}
        view = await service.get_effect_verification(**request, control_target_id=control_id)
        assert {item["target_id"] for item in view.evidence_candidates} == {str(run.target_id), control_id}
        assert uow.runs.list_effect_artifacts.await_count == 2
        assert all(item.kwargs["limit"] == 20 for item in uow.runs.list_effect_artifacts.await_args_list)
        with pytest.raises(Exception, match="冻结后替换"):
            await service.get_effect_verification(**request, control_target_id=uuid7())
        uow.agents.version_has_target.return_value = False
        with pytest.raises(Exception, match="权限"):
            await service.get_effect_verification(**request, control_target_id=control_id)
    asyncio.run(execute())


def test_automatic_followup_collects_both_frozen_targets_in_the_same_requested_window():
    from aiops_agent.application.changes.effect_followup import BusinessEffectFollowupService, schedule_effect_followup
    async def execute():
        changes, uow, run, proposal, command, _ = await service_fixture()
        verification = await uow.runs.get_artifact(artifact_id=command.verification_artifact_id)
        receipt = await uow.runs.get_artifact(artifact_id=verification.payload_json["result_artifact_id"])
        await schedule_effect_followup(uow=uow, run=run, proposal=proposal,
            verification_artifact=verification, receipt=receipt, now=NOW + timedelta(seconds=2), trace_id="test")
        payload = uow.outbox.add.await_args.args[0].payload_json
        workloads = SimpleNamespace(create_workload_report=AsyncMock(side_effect=[
            SimpleNamespace(artifact_id=command.after_artifact_id),
            SimpleNamespace(artifact_id=command.control_after_artifact_id),
        ]))
        service = BusinessEffectFollowupService(uow_factory=lambda: uow, change_service=changes, workload_service=workloads)
        await service.execute(payload)
        calls = workloads.create_workload_report.await_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["target_id"] != calls[1].kwargs["target_id"]
        assert calls[0].kwargs["period_start"] == calls[1].kwargs["period_start"]
        assert calls[0].kwargs["period_end"] == calls[1].kwargs["period_end"]
        assert calls[0].kwargs["idempotency_key"] != calls[1].kwargs["idempotency_key"]
        assert uow.inspections.publish_report.await_args.args[0].result == "IMPROVED"
        # 同一冻结任务重放不再采集、不生成第二份正式报告。
        await service.execute(payload)
        assert workloads.create_workload_report.await_count == 2
        uow.inspections.publish_report.assert_awaited_once()
    asyncio.run(execute())


@pytest.mark.parametrize("database_type,identity_type,kind,value,treatment_id,control_id", [
    ("POSTGRESQL", "POSTGRESQL_SYSTEM_IDENTIFIER", "POSTGRESQL_QUERY_ID", "123", "111", "222"),
    ("MYSQL", "MYSQL_SERVER_UUID", "MYSQL_DIGEST", "a" * 64, "primary-uuid", "control-uuid"),
    ("ORACLE", "ORACLE_DATABASE_INSTANCE_CONTAINER", "ORACLE_SQL_ID", "36apy6dncjggq", "11:1:3", "22:1:3"),
])
def test_engine_specific_identity_is_preserved_in_the_same_control_comparison(database_type, identity_type, kind, value, treatment_id, control_id):
    frozen, baseline, after, reports, aid = comparison_fixture()
    for payload in (baseline, after, *reports.values()):
        payload["database_type"] = database_type
        payload["instance_identity"] = {"identity_type": identity_type,
            "identity_value": treatment_id if payload["target_id"] == TARGET else control_id}
        payload["top_statements"][0].update(statement_identity_type=kind, statement_identity_value=value)
    for rule in (*frozen["signals"], *frozen["controlled_comparison"]["signals"]):
        if rule.get("statement_identity"):
            rule["statement_identity"] = {**rule["statement_identity"], "statement_identity_type": kind, "statement_identity_value": value}
    assert evaluate(frozen, baseline, after, reports, aid)["attribution_strength"] == "SUPPORTED"
    if database_type == "ORACLE":
        # 不同实例/容器不是独立 DBID；不能用同库 RAC/PDB 别名提高归因等级。
        for payload in reports.values():
            if payload["target_id"] != TARGET:
                payload["instance_identity"]["identity_value"] = "11:2:7"
        assert evaluate(frozen, baseline, after, reports, aid)["attribution_strength"] != "SUPPORTED"
