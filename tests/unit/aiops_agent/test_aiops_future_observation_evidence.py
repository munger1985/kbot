"""后台观察进入后续诊断、报告和案例的正向链与撤权/篡改回归。"""

import asyncio
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.investigation.future_observation import preserve_future_observation, schedule_future_observation
from aiops_agent.application.investigation.future_observation_evidence import (
    load_future_observation_evidence, validate_reused_future_observation,
)
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.playbooks import canonical_hash
from aiops_agent.tools.observation_series import assemble_observation_series
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_future_observation import NOW, fixture, result


async def scenario(*, engine="POSTGRESQL", status="COMPLETED"):
    uow, source, task, artifact = fixture(engine=engine)
    source.plan_snapshot_json["target"] = {"row_version": 2, "db_type": engine}
    identity = await schedule_future_observation(uow=uow, run=source, task=task, artifact=artifact, now=NOW)
    background = uow.runs.rows[identity]
    info = background.plan_snapshot_json["future_observation"]
    stamp = NOW + timedelta(seconds=61)
    if status == "COMPLETED":
        for index, key in enumerate(info["point_keys"], start=1):
            point = TurnPlanningService._artifact(
                ops_run_id=identity, artifact_key=info["point_artifact_keys"][key], artifact_type="EVIDENCE",
                schema_version="DBA_TOOL_RESULT.v1", payload=result(source.target_id, engine=engine,
                    captured_at=NOW + timedelta(seconds=index * 30)).model_dump(mode="json"),
                producer="test", trust_level="SOURCE_VERIFIED",
            )
            point.security_level = 2
            await uow.runs.add_artifact(point)
        originals = await uow.runs.list_artifacts(ops_run_id=identity)
        combined = assemble_observation_series(artifacts=tuple({
            "artifact_key": item.artifact_key, "payload": item.payload_json, "content_hash": item.content_hash,
            "schema_version": item.schema_version, "trust_level": item.trust_level,
        } for item in originals), info=info, target_id=source.target_id, now=stamp)
        final = TurnPlanningService._artifact(
            ops_run_id=identity, artifact_key="test:series", artifact_type="EVIDENCE",
            schema_version="DBA_TOOL_RESULT.v1", payload=combined.model_dump(mode="json"),
            producer="test", trust_level="SOURCE_VERIFIED",
        )
        final.security_level = 2
        await uow.runs.add_artifact(final)
        background.final_artifact_id = final.artifact_id
    else:
        await preserve_future_observation(uow=uow, run=background, now=stamp, reason="FOLLOWUP_CANCELLED")
    background.status = status
    current = deepcopy(source)
    current.ops_run_id, current.status = uuid7(), "RUNNING"
    uow.runs.rows[current.ops_run_id] = current
    turn_scope = {"domain_id": source.domain_id, "conversation_id": uuid7(), "created_by": source.actor_id}
    previous = SimpleNamespace(**turn_scope, turn_id=uuid7(), turn_no=1, status="COMPLETED")
    turn = SimpleNamespace(**turn_scope, turn_id=uuid7(), turn_no=2, status="PLANNING")
    turns = {item.turn_id: item for item in (previous, turn)}
    uow.turns = SimpleNamespace(
        get_turn=AsyncMock(side_effect=lambda *, domain_id, turn_id: turns.get(turn_id)),
        list_prior_turns=AsyncMock(return_value=[previous]),
        get_run_link=AsyncMock(side_effect=lambda *, turn_id, purpose: SimpleNamespace(
            ops_run_id=source.ops_run_id if turn_id == previous.turn_id else current.ops_run_id)),
        get_run_link_by_ops_run_id=AsyncMock(return_value=SimpleNamespace(turn_id=turn.turn_id)),
        add_evidence=AsyncMock(),
    )
    uow.runs.database_now = AsyncMock(return_value=stamp + timedelta(seconds=1))
    return SimpleNamespace(uow=uow, source=source, background=background, run=current, turn=turn,
        previous=previous, baseline=artifact, now=stamp + timedelta(seconds=1))


async def imported(fixture):
    entries = await load_future_observation_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run)
    assert len(entries) == 1
    planner = object.__new__(TurnPlanningService)
    artifact, = await planner._persist_future_observation_evidence(
        uow=fixture.uow, run=fixture.run, turn=fixture.turn, entries=entries,
    )
    return artifact


def worker_context(fixture, artifacts):
    return SimpleNamespace(plan_snapshot={"answer_context": {"task_frame": {
        "objectives": ["DIAGNOSE"], "problem_statement": "继续分析已取得的后台观察",
    }}, "capability_snapshot": {"database_type": fixture.source.plan_snapshot_json["target"]["db_type"]}},
        trigger_type="CHAT", run_id=fixture.run.ops_run_id, task_id=uuid7(), attempt=1, deadline_at=None,
        input_artifacts=tuple({
            "artifact_id": str(item.artifact_id), "schema_version": item.schema_version,
            "payload": item.payload_json, "provenance": item.provenance_json,
        } for item in artifacts))


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_mapped_future_sources_keep_native_scope_and_sampling_warnings_during_report_replay(engine, monkeypatch):
    from aiops_agent.contracts.tool_execution import DbaToolResult, ToolOutcome
    from tests.unit.aiops_agent.test_aiops_native_field_lineage import observation_for, tool_for

    def mapped_result(target_id, *, engine="POSTGRESQL", captured_at=NOW, **updates):
        observed = observation_for(tool_for(engine), target=target_id).model_copy(update={"captured_at": captured_at, **updates})
        return DbaToolResult(source_type="TOOL", source_id=observed.tool_id, source_version="1.0.0",
            definition_hash="e" * 64, output_schema="DBA_TOOL_RESULT.v1", measurement_semantics="CUMULATIVE_SINCE_LOAD",
            presentation_kind="TABLE", status="SUCCEEDED", tool_outcomes=(ToolOutcome(step_id="a1", tool_id=observed.tool_id,
                tool_version=observed.tool_version, status="SUCCEEDED", observation=observed),))

    monkeypatch.setitem(fixture.__globals__, "result", mapped_result)
    monkeypatch.setitem(scenario.__globals__, "result", mapped_result)

    async def run():
        fixture = await scenario(engine=engine)
        artifact = await imported(fixture)
        assessment = await DbaEvidenceAssessmentHandler().execute(worker_context(fixture, (artifact,)))
        assert len(assessment.evidence) == 3, [(gap.code, gap.detail) for gap in assessment.gaps]
        assert all(fact.provenance.native_lineage is not None for fact in assessment.evidence)
        payload = {"evidence": [fact.model_dump(mode="json") for fact in assessment.evidence]}
        assert await validate_investigation_sources(uow=fixture.uow, run=fixture.run, payload=payload)
        payload["evidence"][0]["warnings"] = []
        assert not await validate_investigation_sources(uow=fixture.uow, run=fixture.run, payload=payload)
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
@pytest.mark.parametrize("status", ["COMPLETED", "PARTIAL", "CANCELLED", "FAILED", "EXPIRED"])
def test_terminal_series_flows_through_planner_worker_and_report_source_replay(engine, status):
    async def run():
        fixture = await scenario(engine=engine, status=status)
        artifact = await imported(fixture)
        assert await validate_reused_future_observation(uow=fixture.uow, run=fixture.run, artifact=artifact)
        assessment = await DbaEvidenceAssessmentHandler().execute(worker_context(fixture, (artifact,)))
        expected = 3 if status == "COMPLETED" else 1
        assert len(assessment.evidence) == expected
        assert all(str(fact.followup_run_id) == str(fixture.background.ops_run_id) for fact in assessment.evidence)
        assert all(any("后台未来观察" in warning for warning in fact.warnings) for fact in assessment.evidence)
        assert assessment.evidence[0].captured_at == NOW.isoformat()
        assert all(str(fact.measurement_semantics) == "CURRENT_ACTIVITY" for fact in assessment.evidence)
        assert await validate_investigation_sources(uow=fixture.uow, run=fixture.run,
            payload={"evidence": [fact.model_dump(mode="json") for fact in assessment.evidence]})
        fixture.uow.turns.list_prior_turns.assert_awaited_once_with(
            conversation_id=fixture.turn.conversation_id, before_turn_no=2, limit=17)
    asyncio.run(run())


@pytest.mark.parametrize("change", [
    "actor", "agent_version", "domain", "target_version", "binding", "source_hash",
    "raw_point_hash", "final_hash", "other_conversation", "source_request_hash", "source_run_link", "future_time",
])
def test_reuse_requires_current_authority_and_original_sources_not_self_consistent_copies(change):
    async def run():
        fixture = await scenario()
        artifact = await imported(fixture)
        if change == "actor": fixture.run.actor_id = str(uuid7())
        elif change == "agent_version": fixture.run.agent_version_id = uuid7()
        elif change == "domain": fixture.run.domain_id = str(uuid7())
        elif change == "target_version": fixture.uow.targets.get_scoped.return_value.row_version = 3
        elif change == "binding": fixture.uow.agents.get_active.return_value = None
        elif change == "source_hash": fixture.baseline.content_hash = "0" * 64
        elif change == "raw_point_hash":
            point = await fixture.uow.runs.get_artifact_by_key(ops_run_id=fixture.background.ops_run_id,
                artifact_key=fixture.background.plan_snapshot_json["future_observation"]["point_artifact_keys"]["future:2"])
            point.payload_json["tool_outcomes"][0]["observation"]["rows"] = [["forged"]]
            point.content_hash = canonical_hash(point.payload_json)
        elif change == "final_hash":
            artifact.payload_json = deepcopy(artifact.payload_json)
            artifact.payload_json["tool_outcomes"][0]["observation"]["rows"] = [["forged"]]
            artifact.content_hash = canonical_hash(artifact.payload_json)
        elif change == "other_conversation": fixture.previous.conversation_id = uuid7()
        elif change == "source_request_hash":
            request = await fixture.uow.runs.get_artifact_by_key(ops_run_id=fixture.source.ops_run_id,
                artifact_key="future-observation:request")
            request.content_hash = "0" * 64
        elif change == "source_run_link":
            fixture.uow.turns.get_run_link.return_value = SimpleNamespace(ops_run_id=uuid7())
            fixture.uow.turns.get_run_link.side_effect = None
        elif change == "future_time":
            fixture.uow.runs.database_now.return_value = NOW
        assert not await validate_reused_future_observation(uow=fixture.uow, run=fixture.run, artifact=artifact)
    asyncio.run(run())


@pytest.mark.parametrize("tamper", ["row", "time", "identity", "semantics", "provenance", "source_ref"])
def test_report_and_case_cannot_rewrite_observations_from_authorized_series(tamper):
    async def run():
        fixture = await scenario()
        artifact = await imported(fixture)
        assessment = await DbaEvidenceAssessmentHandler().execute(worker_context(fixture, (artifact,)))
        payload = {"evidence": [fact.model_dump(mode="json") for fact in assessment.evidence]}
        fact = payload["evidence"][0]
        if tamper == "row": fact["rows"] = [["forged"]]
        elif tamper == "time": fact["captured_at"] = fixture.now.isoformat()
        elif tamper == "identity": fact["followup_run_id"] = str(uuid7())
        elif tamper == "semantics": fact["measurement_semantics"] = "SNAPSHOT_DELTA"
        elif tamper == "provenance": fact["provenance"] = None
        elif tamper == "source_ref": fact["evidence_ref"] = f"artifact:{uuid7()}#{fact['step_id']}"
        assert not await validate_investigation_sources(uow=fixture.uow, run=fixture.run, payload=payload)
    asyncio.run(run())


def test_prior_sql_and_future_observations_share_fact_row_and_byte_budgets():
    async def run():
        fixture = await scenario()
        for prior in (
            {"facts": [{}] * 8},
            {"facts": [{"rows": [[]] * 512}]},
            {"facts": [{"rows": [], "text": "x" * 262144}]},
        ):
            assert not await load_future_observation_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run, prior_sql=prior)
        fixture.uow.turns.list_prior_turns.return_value = [fixture.previous] * 17
        entries = await load_future_observation_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run)
        assert len(entries) == 1
        assert entries[0]["provenance"]["future_observation_source"]["search_truncated"]
    asyncio.run(run())


def test_planner_links_future_artifact_keys_without_sql_profile_restriction():
    async def run():
        fixture = await scenario()
        class Factory:
            def __call__(self): return self
            async def __aenter__(self): return fixture.uow
            async def __aexit__(self, *_args): return None
        planner = object.__new__(TurnPlanningService)
        planner._uow_factory = Factory()
        context = SimpleNamespace(domain_id=fixture.run.domain_id, turn_id=fixture.turn.turn_id,
            ops_run_id=fixture.run.ops_run_id, prior_sql_evidence=None, source_run_evidence=None,
            resolved_uploads=(), recovery_assurance={})
        context.future_observation_evidence = await planner._load_future_observation_evidence(context)
        assert "turn-future-observation:1" in planner._initial_user_evidence_keys(context)
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
def test_formal_report_and_case_reuse_recheck_future_sources_in_actual_application_paths(engine):
    from aiops_agent.application.case_source_validation import validate_case_sources
    from aiops_agent.application.diagnosis.investigation_record import freeze_investigation, render_investigation
    from aiops_agent.application.errors import AIOpsApplicationError
    from aiops_agent.application.reporting import resolve_system_template
    from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
    from aiops_agent.contracts.turn_answer import AIOpsTurnResult, TurnAnswerBlock

    async def run():
        fixture = await scenario(engine=engine)
        imported_artifact = await imported(fixture)
        assessment = await DbaEvidenceAssessmentHandler().execute(worker_context(fixture, (imported_artifact,)))
        fact = assessment.evidence[0]
        draft = DiagnosisAnswerDraft(claims=({
            "claim_id": "c1", "kind": "FACT", "subject": "后台采样的会话",
            "statement": "后台采样时点记录到会话 42；不能外推为当前状态。", "level": "OBSERVED",
            "applicable_time": fact.captured_at, "reasoning": "原始观察返回会话标识。",
            "evidence_fields": ({"evidence_ref": fact.evidence_ref, "row_index": 0, "field": "session_id"},),
        },), goal_satisfied=True)
        context = worker_context(fixture, ())
        context.target_id = fixture.run.target_id
        record = freeze_investigation(draft=draft, evidence=assessment.evidence, context=context)
        assert record.observations[0].followup_run_id == str(fixture.background.ops_run_id)
        analysis, solution, refs = render_investigation(record)
        result = AIOpsTurnResult(status="PARTIAL", sufficiency_status="PARTIAL", evidence=assessment.evidence,
            investigation_record=record, blocks=(TurnAnswerBlock(block_type="ANALYSIS_MARKDOWN",
                schema_version="AIOPS_ANALYSIS_BLOCK.v1", payload={"markdown": analysis}, evidence_refs=refs),))
        root = TurnPlanningService._artifact(ops_run_id=fixture.run.ops_run_id, artifact_key="turn:result",
            artifact_type="AIOPS_TURN_RESULT", schema_version=result.schema_version,
            payload=result.model_dump(mode="json"), producer="test", trust_level="SOURCE_VERIFIED")
        await fixture.uow.runs.add_artifact(root)
        fixture.run.final_artifact_id, fixture.run.status = root.artifact_id, "PARTIAL"
        fixture.run.created_at, fixture.run.trigger_type = NOW, "CHAT"
        original_add = fixture.uow.runs.add_artifact
        async def add_artifact(row):
            row.artifact_id = row.artifact_id or uuid7()
            return await original_add(row)
        fixture.uow.runs.add_artifact = add_artifact
        fixture.uow.inspections = SimpleNamespace(publish_report=AsyncMock(side_effect=lambda row: row),
            add_report_sources=AsyncMock())
        fixture.uow.outbox = SimpleNamespace(add=AsyncMock(side_effect=lambda row: row))
        fixture.uow.platform_notifications = SimpleNamespace(emit_report_ready=AsyncMock())
        runtime = object.__new__(AIOpsRuntimeService)
        async def publish():
            return await runtime._publish_diagnosis_report(uow=fixture.uow, run=fixture.run,
                task=SimpleNamespace(ops_task_id=uuid7()), source_artifact=root,
                now=fixture.now, trace_id="future-report-test", actor_id=fixture.run.actor_id,
                template=resolve_system_template("system:diagnosis.standard"))
        report = await publish()
        assert report.status == "PARTIAL"
        content = await fixture.uow.runs.get_artifact(artifact_id=report.content_artifact_id)
        assert "后台采样时点记录到会话 42" in content.payload_json["facts"][0]["summary"]
        assert content.payload_json["facts"][0]["applicable_time"] == fact.captured_at
        assert content.payload_json["facts"][0]["level"] == "OBSERVED"
        assert content.payload_json["facts"][0]["evidence_refs"] == [fact.evidence_ref]
        fixture.uow.runs.get_run_scoped = AsyncMock(side_effect=lambda *, ops_run_id, domain_id: fixture.uow.runs.rows.get(ops_run_id))
        fixture.uow.operations_knowledge = SimpleNamespace(get_version_relations=AsyncMock(return_value={"sources": [
            SimpleNamespace(source_report_id=None, source_artifact_id=root.artifact_id,
                source_run_id=fixture.run.ops_run_id, content_hash=root.content_hash),
        ]}))
        fixture.uow.agents.get = AsyncMock(return_value=SimpleNamespace(current_version_id=fixture.run.agent_version_id))
        fixture.uow.agents.version_has_target = AsyncMock(return_value=True)
        fixture.uow.agents.version_source_ids = AsyncMock(return_value=())
        fixture.uow.turns.list_turns = AsyncMock(return_value=[])
        assert await validate_case_sources(uow=fixture.uow, domain_id=fixture.run.domain_id, version_id=uuid7())
        fixture.uow.agents.get_active.return_value = None
        with pytest.raises(AIOpsApplicationError):
            await publish()
        assert fixture.uow.inspections.publish_report.await_count == 1
        assert not await validate_case_sources(uow=fixture.uow, domain_id=fixture.run.domain_id, version_id=uuid7())
    asyncio.run(run())

def test_running_background_is_not_imported_as_completed_observations():
    async def run():
        fixture = await scenario()
        fixture.background.status = "RUNNING"
        assert not await load_future_observation_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run)
    asyncio.run(run())


def test_revocation_between_search_and_plan_commit_stores_only_gap_not_values():
    async def run():
        fixture = await scenario()
        entries = await load_future_observation_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run)
        fixture.uow.agents.get_active.return_value = None
        planner = object.__new__(TurnPlanningService)
        artifact, = await planner._persist_future_observation_evidence(
            uow=fixture.uow, run=fixture.run, turn=fixture.turn, entries=entries,
        )
        assert artifact.schema_version == "FUTURE_OBSERVATION_UNAVAILABLE.v1"
        assert artifact.payload_json == {"code": "FOLLOWUP_SOURCE_UNAVAILABLE"}
        fixture.uow.turns.add_evidence.assert_not_awaited()
        assessed = await DbaEvidenceAssessmentHandler().execute(worker_context(fixture, (artifact,)))
        assert assessed.evidence == () and assessed.gaps[0].code == "FOLLOWUP_SOURCE_UNAVAILABLE"
    asyncio.run(run())


def test_future_sql_cannot_be_laundered_through_prior_sql_rebinding():
    from aiops_agent.application.diagnosis.sql_evidence_answer import rebind_prior_sql_facts
    from tests.unit.aiops_agent.test_aiops_sql_evidence_answer import fact
    entry = fact("db.postgresql.statement.top", ("query_id", "database_oid", "execution_count"), (("42", 1, 100),))
    entry = type(entry).model_validate({**entry.model_dump(mode="json"), "followup_run_id": str(uuid7())})
    assert not rebind_prior_sql_facts(payload={"facts": [entry.model_dump(mode="json")]}, artifact_id=str(uuid7()),
        task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42"}},
        database_type="POSTGRESQL")

@pytest.mark.parametrize("change", ["valid", "disabled", "missing_flag", "hash", "policy_binding"])
def test_reuse_rechecks_actual_policy_binding_and_readonly_flag(change):
    async def run():
        fixture = await scenario()
        identity = uuid7()
        fixture.source.plan_snapshot_json["binding"]["policy_id"] = str(identity)
        fixture.background.plan_snapshot_json["binding"]["policy_id"] = str(identity)
        fixture.uow.agents.get_active.return_value.policy_id = identity
        fixture.background.policy_snapshot_json = {"policy_hash": "a" * 64}
        policy = SimpleNamespace(status="ACTIVE", rules_json={"readonly_database_enabled": True}, policy_hash="a" * 64)
        fixture.uow.policies.get_scoped.return_value = policy
        if change == "disabled": policy.rules_json["readonly_database_enabled"] = False
        elif change == "missing_flag": policy.rules_json = {}
        elif change == "hash": policy.policy_hash = "b" * 64
        elif change == "policy_binding": fixture.uow.agents.get_active.return_value.policy_id = uuid7()
        entries = await load_future_observation_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run)
        assert bool(entries) == (change == "valid")
    asyncio.run(run())

def test_runtime_claim_revocation_removes_old_values_and_old_judgment_before_model():
    async def run():
        fixture = await scenario()
        artifact = await imported(fixture)
        assessment = await DbaEvidenceAssessmentHandler().execute(worker_context(fixture, (artifact,)))
        cached = TurnPlanningService._artifact(
            ops_run_id=fixture.run.ops_run_id, artifact_key="old:assessment", artifact_type="EVIDENCE",
            schema_version="DBA_SUFFICIENCY.v1", payload=assessment.model_dump(mode="json"),
            producer="test", trust_level="SOURCE_VERIFIED",
        )
        await fixture.uow.runs.add_artifact(cached)
        runtime = object.__new__(AIOpsRuntimeService)
        task = SimpleNamespace(input_artifacts_json=[artifact.artifact_key, cached.artifact_key])
        valid = await runtime._input_artifacts(fixture.uow, run_id=fixture.run.ops_run_id, task=task)
        assert valid[0].schema_version == "DBA_TOOL_RESULT.v1"
        fixture.uow.agents.get_active.return_value = None
        leases = await runtime._input_artifacts(fixture.uow, run_id=fixture.run.ops_run_id, task=task)
        by_id = {str(item.artifact_id): item for item in leases}
        assert by_id[str(artifact.artifact_id)].schema_version == "FUTURE_OBSERVATION_UNAVAILABLE.v1"
        safe = DbaSufficiencyAssessment.model_validate(by_id[str(cached.artifact_id)].payload)
        assert safe.evidence == () and safe.investigation is None and safe.status == "PARTIAL"
        assert safe.gaps[-1].code == "FOLLOWUP_SOURCE_UNAVAILABLE"
        # 原始不可变 Artifact 没有被删除或就地改写。
        assert cached.payload_json["evidence"] and canonical_hash(artifact.payload_json) == artifact.content_hash
    asyncio.run(run())
