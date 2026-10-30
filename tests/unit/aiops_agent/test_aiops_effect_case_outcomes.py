"""所有真实效果结果均可沉淀，但来源复核与成功等级不能被分类绕过。"""

import asyncio
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.case_source_validation import validate_case_effect
from aiops_agent.application.changes.effect_replay import replay_supported_effect
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.change import AdvisoryActionResult
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_business_effect import NOW
from tests.unit.aiops_agent.test_aiops_controlled_effect import service_fixture
from tests.unit.aiops_agent.test_aiops_investigation_record import draft, evidence
from tests.unit.aiops_agent.test_aiops_report_case_extraction import case_fixture, extract
from tests.unit.aiops_agent.test_aiops_turn_answer import _proposal_artifact


async def outcome_fixture(scenario):
    """先生产原始测量及正式报告，再走实际案例提炼服务，不伪造结果标志。"""
    f = await case_fixture()
    effects, uow, run, proposal, command, kwargs = await service_fixture(
        with_control=scenario not in {"observational", "manual_improved", "manual_unchanged"},
    )
    after = await uow.runs.get_artifact(artifact_id=command.after_artifact_id)
    verification = await uow.runs.get_artifact(artifact_id=command.verification_artifact_id)
    receipt = await uow.runs.get_artifact(artifact_id=verification.payload_json["result_artifact_id"])
    row = after.payload_json["top_statements"][0]
    if scenario in {"unchanged", "manual_unchanged"}:
        row["total_duration_microseconds"] = row["execution_count"] * 95
    elif scenario == "degraded":
        row["total_duration_microseconds"] = row["execution_count"] * 150
    elif scenario in {"guardrail", "guardrail_with_gap"}:
        row["rows_processed"] = 1
    if scenario == "load_gap":
        after.payload_json["load_totals"]["transactions"] = 500
        after.payload_json["load_profile"]["transactions_per_second"] = 500 / 900
    if scenario in {"verification_gap", "guardrail_with_gap"}:
        verification.payload_json["gap_codes"] = ["ACTION_POSTCONDITION_UNOBSERVED"]
    if scenario == "not_achieved":
        verification.payload_json["status"] = "NOT_ACHIEVED"
    if scenario == "adverse":
        verification.payload_json["status"] = "ADVERSE"
    if scenario.startswith("manual_"):
        # 各平台正式合同自行校验 actor 类型，不为夹具放宽产品合同。
        actor = str(uuid7())
        body = AdvisoryActionResult(
            proposal_id=str(proposal.proposal_id), status="EXECUTED",
            occurred_at=NOW + timedelta(seconds=1), submitted_at=NOW + timedelta(seconds=2),
            submitted_by=actor, note="DBA 登记人工处理，不是系统执行回执", result_hash="a" * 64,
        ).model_dump(mode="json")
        receipt.schema_version = body["schema_version"]
        receipt.trust_level = "USER_PROVIDED"
        receipt.payload_json = body
        run.actor_id = actor
        kwargs["actor_id"] = actor
    for item in (after, verification, receipt):
        item.content_hash = sha256_json(item.payload_json)
    context = SimpleNamespace(run_id=run.ops_run_id, task_id=proposal.ops_task_id,
        target_id=run.target_id, plan_snapshot={"answer_context": {"question": "调查业务阻塞"}})
    fact = evidence()
    record = freeze_investigation(draft=draft(), evidence=(fact,), context=context)
    root = {"schema_version": "AIOPS_TURN_RESULT.v1", "status": "COMPLETED",
        "investigation_record": record.model_dump(mode="json"), "evidence": [fact.model_dump(mode="json")]}
    snapshot = _proposal_artifact(mode="ADVISORY" if scenario.startswith("manual_") else "AGENT_EXECUTE")["payload"]
    snapshot["proposal"].update(proposal_id=str(proposal.proposal_id), run_id=str(run.ops_run_id),
        target_id=str(run.target_id), proposal_hash="b" * 64, command_hash="c" * 64)
    run.final_artifact_id, proposal.snapshot_artifact_id = uuid7(), uuid7()
    for identity, body in ((run.final_artifact_id, root), (proposal.snapshot_artifact_id, snapshot)):
        await uow.runs.add_artifact(SimpleNamespace(
            artifact_id=identity, ops_run_id=run.ops_run_id, artifact_key=str(identity),
            schema_version=body["schema_version"], payload_json=body, content_hash=sha256_json(body),
            trust_level="SOURCE_VERIFIED", security_level=4, provenance_json={},
        ))
    target = uow.targets.get_scoped.return_value
    target.db_type, target.version_code, target.environment, target.capabilities_json = "POSTGRESQL", "16", "TEST", {}
    f.view = await effects.evaluate_effect_plan(command=command, **kwargs)
    f.report = uow.inspections.publish_report.await_args.args[0]
    f.report.is_current = True
    f.uow.inspections.get_report_scoped.side_effect = lambda **kw: f.report
    f.uow.inspections.list_report_sources.side_effect = lambda **kw: uow.inspections.add_report_sources.await_args.args[0]
    f.uow.runs, f.uow.targets, f.uow.agents, f.uow.changes = uow.runs, uow.targets, uow.agents, uow.changes
    f.uow.agents.version_source_ids = AsyncMock(return_value=[])
    f.domain, f.effect_run, f.after, f.receipt = run.domain_id, run, after, receipt
    return f


@pytest.mark.parametrize("scenario,kind,result,status", [
    ("supported", "VERIFIED_OPTIMIZATION", "IMPROVED", "READY"),
    ("unchanged", "NEGATIVE_EXPERIENCE", "UNCHANGED", "PARTIAL"),
    ("degraded", "NEGATIVE_EXPERIENCE", "DEGRADED", "PARTIAL"),
    ("guardrail", "NEGATIVE_EXPERIENCE", "DEGRADED", "READY"),
    ("guardrail_with_gap", "NEGATIVE_EXPERIENCE", "DEGRADED", "PARTIAL"),
    ("load_gap", "DIAGNOSTIC_REFERENCE", "INCONCLUSIVE", "PARTIAL"),
    ("verification_gap", "DIAGNOSTIC_REFERENCE", "INCONCLUSIVE", "PARTIAL"),
    ("observational", "DIAGNOSTIC_REFERENCE", "IMPROVED", "READY"),
    ("manual_improved", "DIAGNOSTIC_REFERENCE", "IMPROVED", "READY"),
    ("manual_unchanged", "NEGATIVE_EXPERIENCE", "UNCHANGED", "READY"),
    ("not_achieved", "NEGATIVE_EXPERIENCE", "NOT_ACHIEVED", "READY"),
    ("adverse", "NEGATIVE_EXPERIENCE", "ADVERSE", "READY"),
])
def test_all_effect_reports_have_a_truthful_candidate_not_just_success(scenario, kind, result, status):
    async def run():
        f = await outcome_fixture(scenario)
        assert f.report.status == status
        eligibility = await f.service.case_eligibility(domain_id=f.domain, report_id=f.report.report_id)
        assert eligibility["eligible"], eligibility
        assert eligibility["case_kind"] == kind
        extracted = await extract(f)
        profile = extracted["version"]["profile"]
        assert profile["case_kind"] == kind
        assert profile["verification"]["result"] == result
        for name in ("direct_effect", "measured_effect", "symptom_recovered", "guardrails_passed", "attribution_strength", "execution_source", "gap_codes"):
            assert profile["verification"][name] == f.view.result[name]
        assert profile["verification"]["effect_result_ref"] == "artifact:" + str(f.view.comparison_artifact.artifact_id)
        assert profile["root_cause"]["effective_level"] == "INCONCLUSIVE"
        if profile["verification"]["attribution_strength"] != "SUPPORTED":
            assert "匹配对照支持本次措施的效果" not in "\n".join(profile["verification"]["causal_limitations"])
        assert await validate_case_effect(uow=f.uow, domain_id=f.domain, profile=profile)
        artifact = await f.uow.runs.get_artifact(artifact_id=f.view.comparison_artifact.artifact_id)
        assert (await replay_supported_effect(uow=f.uow, run=f.effect_run, artifact=artifact) is not None) == (kind.startswith("VERIFIED_"))
        assert extracted["version"]["status"] == "PROCESSING"
        assert extracted["asset"]["current_version_id"] is None
        model_input = f.service._structured_model.generate_structured.await_args.kwargs["input_payload"]
        assert str(f.view.comparison_artifact.artifact_id) not in str(model_input)
        if scenario.startswith("manual_"):
            assert profile["actions"][0]["source_kind"] == "USER_PROVIDED"
            body = f.service._kc.ingest_user_file.await_args.kwargs["body"].decode()
            assert "人工登记" in body and "独立窗口测量" in body
            assert "基线 100 → 处置后" in body
        if f.view.result["gap_codes"]:
            body = f.service._kc.ingest_user_file.await_args.kwargs["body"].decode()
            assert "验证缺口" in body
        if kind == "NEGATIVE_EXPERIENCE":
            body = f.service._kc.ingest_user_file.await_args.kwargs["body"].decode()
            assert "不作为推荐措施" in body
    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["unchanged", "degraded", "observational", "manual_improved", "load_gap"])
@pytest.mark.parametrize("failure", ["revoked", "source_hash", "execution", "result_tamper"])
def test_nonverified_effect_is_replayed_before_reuse(scenario, failure):
    async def run():
        f = await outcome_fixture(scenario)
        profile = (await extract(f))["version"]["profile"]
        if failure == "revoked":
            f.uow.agents.version_has_target.return_value = False
        elif failure == "source_hash":
            f.after.payload_json["snapshot_count"] = 1
        elif failure == "execution":
            f.receipt.payload_json["status"] = "FAILED"
            f.receipt.content_hash = sha256_json(f.receipt.payload_json)
        else:
            artifact = await f.uow.runs.get_artifact(artifact_id=f.view.comparison_artifact.artifact_id)
            artifact.payload_json["guardrails_passed"] = False
            artifact.content_hash = sha256_json(artifact.payload_json)
        assert not await validate_case_effect(uow=f.uow, domain_id=f.domain, profile=profile)
    asyncio.run(run())


def test_changed_execution_during_scan_blocks_negative_case_before_intake():
    async def run():
        f = await outcome_fixture("unchanged")
        async def scan(**kw):
            f.uow.changes.get_execution.return_value.status = "FAILED"
            return SimpleNamespace(output={"spans": []})
        f.service._structured_model.generate_structured.side_effect = scan
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        f.service._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())


def test_case_flags_cannot_rewrite_a_negative_measurement_as_success():
    async def run():
        f = await outcome_fixture("degraded")
        profile = (await extract(f))["version"]["profile"]
        forged = deepcopy(profile)
        forged["verification"].update(measured_effect="IMPROVED", result="UNCHANGED")
        assert not await validate_case_effect(uow=f.uow, domain_id=f.domain, profile=forged)
    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["unchanged", "observational", "manual_improved"])
def test_effect_link_cannot_be_removed_to_bypass_replay_on_nonverified_case(scenario):
    async def run():
        f = await outcome_fixture(scenario)
        profile = (await extract(f))["version"]["profile"]
        profile["verification"].pop("effect_result_ref")
        assert not await validate_case_effect(uow=f.uow, domain_id=f.domain, profile=profile)
    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["unchanged", "manual_unchanged", "observational", "load_gap"])
def test_effect_source_change_after_intake_queues_withdrawal_without_business_publication(scenario):
    async def run():
        f = await outcome_fixture(scenario)
        response = f.service._kc.ingest_user_file.return_value
        async def ingest(**kw):
            f.receipt.payload_json["status"] = "FAILED"
            f.receipt.content_hash = sha256_json(f.receipt.payload_json)
            return response
        f.service._kc.ingest_user_file.side_effect = ingest
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        f.service._kc.review_user_intake.assert_not_awaited()
        version = next(iter(f.state.versions.values()))
        assert version.status == "REVIEW_REQUIRED"
        assert f.uow.outbox.add.await_count == 1
    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["unchanged", "degraded", "manual_unchanged", "observational"])
def test_published_nonverified_case_stops_new_search_when_effect_source_is_lost(scenario):
    from aiops_agent.contracts.knowledge import KnowledgeSearchRequest
    import inspect
    async def run():
        f = await outcome_fixture(scenario)
        await extract(f)
        asset, version = next(iter(f.state.assets.values())), next(iter(f.state.versions.values()))
        index = f.state.relations[version.asset_version_id]["indexes"][0]
        asset.current_version_id = version.asset_version_id
        asset.status = version.status = "PUBLISHED"
        f.uow.operations_knowledge.list_published_candidates = AsyncMock(return_value=[(asset, version, index)])
        f.service._kc.discover = AsyncMock()
        f.receipt.payload_json["status"] = "FAILED"
        f.receipt.content_hash = sha256_json(f.receipt.payload_json)
        kwargs = dict(domain_id=f.domain, agent_id=str(f.effect_run.agent_id),
            request=KnowledgeSearchRequest(query="查询优化", knowledge_base_ids=(uuid7(),)), context=f.context)
        if "tenant_id" in inspect.signature(f.service.search).parameters:
            kwargs["tenant_id"] = uuid7()
        result = await f.service.search(**kwargs)
        assert not result["results"]
        f.service._kc.discover.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["unchanged", "observational", "load_gap"])
@pytest.mark.parametrize("invalidate", [False, True])
def test_real_publish_service_revalidates_nonpositive_execution_authority(scenario, invalidate):
    import inspect
    async def run():
        f = await outcome_fixture(scenario)
        await extract(f)
        asset, version = next(iter(f.state.assets.values())), next(iter(f.state.versions.values()))
        relations = f.state.relations[version.asset_version_id]
        relations.update(scopes=[], reviews=[])
        index = relations["indexes"][0]
        index.index_status, version.status = "READY", "REVIEW_REQUIRED"
        f.service._kc.get_bundle_status = AsyncMock(return_value={
            "current_revision_id": str(index.bundle_revision_id), "availability_status": "READY",
        })
        if invalidate:
            f.uow.changes.get_execution.return_value.status = "FAILED"
        kwargs = dict(domain_id=f.domain, asset_version_id=version.asset_version_id,
            decision="PUBLISH", expected_row_version=version.row_version, comment="人工审核", context=f.context)
        if "tenant_id" in inspect.signature(f.service.review).parameters:
            kwargs["tenant_id"] = uuid7()
        if invalidate:
            with pytest.raises(AIOpsApplicationError, match="效果来源"):
                await f.service.review(**kwargs)
            assert asset.current_version_id is None
            assert version.status == "REVIEW_REQUIRED"
        else:
            published = await f.service.review(**kwargs)
            assert published["version"]["status"] == "PUBLISHED"
            assert published["asset"]["current_version_id"] == str(version.asset_version_id)
    asyncio.run(run())


def test_background_revalidation_retires_negative_case_when_execution_is_disputed():
    async def run():
        f = await outcome_fixture("unchanged")
        await extract(f)
        asset, version = next(iter(f.state.assets.values())), next(iter(f.state.versions.values()))
        index = f.state.relations[version.asset_version_id]["indexes"][0]
        asset.current_version_id, asset.status, version.status = version.asset_version_id, "PUBLISHED", "PUBLISHED"
        f.uow.operations_knowledge.next_case_index_for_validation = AsyncMock(return_value=(index, f.domain))
        f.uow.platform_notifications = SimpleNamespace(emit_case_review_required=AsyncMock())
        f.uow.changes.get_execution.return_value.status = "FAILED"
        assert await f.service.revalidate_next_case()
        assert asset.current_version_id is None
        assert asset.status == version.status == "REVIEW_REQUIRED"
        assert f.uow.outbox.add.await_count == 1
        f.uow.platform_notifications.emit_case_review_required.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["measurement", "scope", "status", "unregistered_source"])
def test_report_projection_cannot_replace_original_effect_measurements(failure):
    async def run():
        f = await outcome_fixture("unchanged")
        content = await f.uow.runs.get_artifact(artifact_id=f.report.content_artifact_id)
        if failure == "measurement":
            content.payload_json["facts"][0]["after"] = "1"
        elif failure == "scope":
            content.payload_json["scope"]["measured_effect"] = "IMPROVED"
        elif failure == "status":
            content.payload_json["status"] = f.report.status = "READY"
        else:
            registered = list(f.uow.inspections.list_report_sources.side_effect())
            registered = [row for row in registered if str(row.source_artifact_id) != str(f.after.artifact_id)]
            f.uow.inspections.list_report_sources.side_effect = None
            f.uow.inspections.list_report_sources.return_value = registered
        content.content_hash = f.report.content_hash = sha256_json(content.payload_json)
        eligible = await f.service.case_eligibility(domain_id=f.domain, report_id=f.report.report_id)
        assert not eligible["eligible"]
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.service._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())


def test_plan_removed_between_effect_replay_and_profile_projection_is_rejected():
    async def run():
        f = await outcome_fixture("unchanged")
        profile = (await extract(f))["version"]["profile"]
        plan_id = f.view.result["comparison_plan_artifact_id"]
        original = f.uow.runs.get_artifact.side_effect
        reads = 0
        def read(artifact_id):
            nonlocal reads
            if str(artifact_id) == plan_id:
                reads += 1
                if reads > 1:
                    return None
            return original(artifact_id=artifact_id)
        f.uow.runs.get_artifact.side_effect = read
        assert not await validate_case_effect(uow=f.uow, domain_id=f.domain, profile=profile)
        assert reads == 2
    asyncio.run(run())
