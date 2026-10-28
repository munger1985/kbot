"""历史筛行不串原生身份，报告与案例不信任只重算哈希的历史副本。"""

import asyncio
from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact, project_evidence_rows
from aiops_agent.application.diagnosis.native_measurements import native_counter_origins
from aiops_agent.application.diagnosis.sql_evidence_answer import rebind_prior_sql_facts
from aiops_agent.contracts.tool_execution import DbaToolResult
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_native_field_lineage import observation_for, tool_for, tool_artifact
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.investigation.prior_sql_evidence import load_prior_sql_evidence
from aiops_agent.application.investigation.prior_sql_replay import validate_prior_sql_payload, validate_reused_prior_sql
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.playbooks import canonical_hash
from tests.unit.aiops_agent.test_aiops_sql_subject_search import search_fixture


def native_fact(engine="POSTGRESQL", count=3):
    tool = tool_for(engine)
    identity = {"POSTGRESQL": "query_id", "MYSQL": "digest", "ORACLE": "sql_id"}[engine]
    values = [str(index + 1) if engine == "POSTGRESQL" else f"{index:064x}" if engine == "MYSQL"
              else f"{index:013d}" for index in range(count)]
    observations = [observation_for(tool, values={identity: value}) for value in values]
    original = observations[0].model_copy(update={"rows": tuple(item.rows[0] for item in observations), "row_count": count})
    artifact = tool_artifact(original, uuid7())
    result = DbaToolResult.model_validate(artifact.payload_json)
    fact = database_evidence_fact(result=result, outcome=result.tool_outcomes[0], artifact_id=artifact.artifact_id)
    frame = {"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "action_intent": "NONE", "subject_ref": {
        "subject_type": {"POSTGRESQL": "POSTGRESQL_QUERY_ID", "MYSQL": "MYSQL_DIGEST", "ORACLE": "ORACLE_SQL_ID"}[engine],
        identity: values[-1],
    }}
    return SimpleNamespace(artifact=artifact, fact=fact, frame=frame)


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_rebinding_last_sql_preserves_its_native_scope_not_the_first_sql(engine):
    fixture = native_fact(engine)
    original = fixture.fact
    selected, = rebind_prior_sql_facts(payload={"facts": [original.model_dump(mode="json")]},
        artifact_id=str(uuid7()), task_frame=fixture.frame, database_type=engine)
    assert selected.provenance.native_lineage.series[0].dimensions == original.provenance.native_lineage.series[2].dimensions
    column = original.provenance.native_lineage.fields[0].column
    assert native_counter_origins(selected.provenance, row_index=0, column=column) == native_counter_origins(original.provenance, row_index=2, column=column)
    assert selected.provenance.native_origin_ids == original.provenance.native_origin_ids
    assert selected.provenance.content_hash == original.provenance.content_hash


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_scope_outside_original_mapping_budget_stays_unknown(engine):
    fixture = native_fact(engine, count=67)
    selected, = rebind_prior_sql_facts(payload={"facts": [fixture.fact.model_dump(mode="json")]},
        artifact_id=str(uuid7()), task_frame=fixture.frame, database_type=engine)
    assert selected.row_count == 1
    assert selected.provenance.native_lineage.series_truncated
    assert selected.provenance.native_lineage.series[0].missing_dimensions
    for field in selected.provenance.native_lineage.fields:
        assert not native_counter_origins(selected.provenance, row_index=0, column=field.column)


def test_row_projection_composes_without_losing_original_scope():
    fixture = native_fact()
    once = project_evidence_rows(fixture.fact, (2, 0))
    twice = project_evidence_rows(once, (1,))
    assert twice.provenance.native_lineage.series[0].dimensions == fixture.fact.provenance.native_lineage.series[0].dimensions
    assert twice.rows == (fixture.fact.rows[0],)


@pytest.mark.parametrize("indices", [(True,), (-1,), (3,), (1, 1)])
def test_row_projection_rejects_ambiguous_or_invalid_indices(indices):
    with pytest.raises(ValueError):
        project_evidence_rows(native_fact().fact, indices)


async def replay_fixture(engine="POSTGRESQL", *, mapped=True, window=False):
    fixture = search_fixture()
    native = native_fact(engine)
    fixture.frame = native.frame
    fixture.run.plan_snapshot_json = {"target": {"row_version": 1, "db_type": engine},
        "answer_context": {"task_frame": fixture.frame}}
    fixture.old_run.plan_snapshot_json = deepcopy(fixture.run.plan_snapshot_json)
    fixture.uow.targets.get_scoped.return_value.db_type = engine
    raw = native.artifact
    raw.ops_run_id = fixture.old_run.ops_run_id
    observation = raw.payload_json["tool_outcomes"][0]["observation"]
    observation["target_id"] = str(fixture.run.target_id)
    if not mapped:
        observation["variant"] = "unmapped-original"
    raw.content_hash = canonical_hash(raw.payload_json)
    result = DbaToolResult.model_validate(raw.payload_json)
    fact = database_evidence_fact(result=result, outcome=result.tool_outcomes[0], artifact_id=raw.artifact_id)
    if window:
        from aiops_agent.application.diagnosis.evidence_projection import top_sql_window_evidence_fact
        raw.schema_version = "TOP_SQL_WINDOW_RESULT.v1"
        subject = fixture.frame["subject_ref"]
        identity = {"POSTGRESQL": "query_id", "MYSQL": "digest", "ORACLE": "sql_id"}[engine]
        raw.payload_json = {"schema_version": raw.schema_version, "target_id": str(fixture.run.target_id), "result": {
            "requested_start": "2026-10-28T01:00:00+00:00", "requested_end": "2026-10-28T01:15:00+00:00",
            "covered_intervals": [{"start": "2026-10-28T01:00:00+00:00", "end": "2026-10-28T01:15:00+00:00"}],
            "collection_enabled": True, "snapshot_ids": [str(uuid7()), str(uuid7())], "top_statements": [{
                "statement_identity_type": subject["subject_type"], "statement_identity_value": subject[identity],
                "normalized_statement": "SELECT * FROM mes.orders", "execution_count": 10,
                "total_duration_microseconds": 123456, "mean_duration_microseconds": "12345.6",
                "database_name": "16510" if engine == "POSTGRESQL" else "mes", "user_identifier": "42", "top_level": True,
            }],
        }}
        fact = top_sql_window_evidence_fact(result=raw.payload_json["result"], artifact_id=raw.artifact_id)
        raw.content_hash = canonical_hash(raw.payload_json)
    raw.security_level = 2
    fixture.artifact.payload_json = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,)).model_dump(mode="json")
    fixture.artifact.content_hash = canonical_hash(fixture.artifact.payload_json)
    fixture.artifact.security_level = 2
    fixture.turn.turn_id = uuid7()
    rows = {fixture.artifact.artifact_id: fixture.artifact, raw.artifact_id: raw}
    fixture.uow.runs.get_artifact = AsyncMock(side_effect=lambda *, artifact_id: rows.get(artifact_id))
    fixture.uow.runs.get_run = AsyncMock(side_effect=lambda *, ops_run_id, **_: fixture.run
        if ops_run_id == fixture.run.ops_run_id else fixture.old_run if ops_run_id == fixture.old_run.ops_run_id else None)
    fixture.uow.turns.get_turn = AsyncMock(side_effect=lambda *, turn_id, **_: fixture.turn
        if turn_id == fixture.turn.turn_id else fixture.previous if turn_id == fixture.previous.turn_id else None)
    fixture.uow.turns.get_run_link_by_ops_run_id = AsyncMock(return_value=SimpleNamespace(turn_id=fixture.turn.turn_id))
    payload = await load_prior_sql_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run, task_frame=fixture.frame)
    assert payload is not None
    imported = TurnPlanningService._artifact(ops_run_id=fixture.run.ops_run_id, artifact_key="turn-prior-sql-evidence:1",
        artifact_type="PRIOR_TURN_SQL_EVIDENCE", schema_version="PRIOR_TURN_SQL_EVIDENCE.v1",
        payload=payload, producer="test", trust_level="SOURCE_VERIFIED")
    imported.security_level = 2
    rows[imported.artifact_id] = imported
    facts = rebind_prior_sql_facts(payload=payload, artifact_id=str(imported.artifact_id), task_frame=fixture.frame, database_type=engine)
    fixture.imported, fixture.raw, fixture.rows, fixture.facts = imported, raw, rows, facts
    return fixture


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
@pytest.mark.parametrize("source_kind", ["native", "unmapped", "window"])
def test_prior_sources_replay_even_without_findings_graphs_or_native_mapping(engine, source_kind):
    async def run():
        fixture = await replay_fixture(engine, mapped=source_kind != "unmapped", window=source_kind == "window")
        assert await validate_reused_prior_sql(uow=fixture.uow, run=fixture.run, artifact=fixture.imported)
        assert await validate_investigation_sources(uow=fixture.uow, run=fixture.run,
            payload={"evidence": [fact.model_dump(mode="json") for fact in fixture.facts]})
        proof, = fixture.imported.payload_json["sources"]
        assert proof["source_row_indices"] == ([0] if source_kind == "window" else [2])
        assert not fixture.facts[0].calculation_graphs
    asyncio.run(run())


@pytest.mark.parametrize("change", ["raw_missing", "raw_hash", "raw_row", "source_row", "proof_row", "proof_ref", "proof_turn",
    "run_scope", "turn_scope", "target_version", "disabled", "revoked", "stripped_sources", "injected_derivation", "fact_row", "fact_warning", "fact_origin"])
def test_replay_rejects_self_consistent_imported_copies_without_original_source_or_authority(change):
    async def run():
        fixture = await replay_fixture()
        payload = fixture.imported.payload_json
        if change == "raw_missing": fixture.rows.pop(fixture.raw.artifact_id)
        elif change == "raw_hash": fixture.raw.content_hash = "f" * 64
        elif change == "raw_row":
            fixture.raw.payload_json["tool_outcomes"][0]["observation"]["rows"][2][0] = "forged"
            fixture.raw.content_hash = canonical_hash(fixture.raw.payload_json)
        elif change == "source_row":
            fixture.artifact.payload_json["evidence"][0]["rows"][2][0] = "forged"
            fixture.artifact.content_hash = canonical_hash(fixture.artifact.payload_json)
            payload["sources"][0]["source_content_hash"] = fixture.artifact.content_hash
        elif change == "proof_row": payload["sources"][0]["source_row_indices"] = [0]
        elif change == "proof_ref": payload["sources"][0]["source_evidence_ref"] = "missing"
        elif change == "proof_turn": payload["sources"][0]["source_turn_id"] = str(uuid7())
        elif change == "run_scope": fixture.old_run.actor_id = uuid7()
        elif change == "turn_scope": fixture.previous.conversation_id = uuid7()
        elif change == "target_version": fixture.uow.targets.get_scoped.return_value.row_version = 3
        elif change == "disabled": fixture.uow.targets.get_scoped.return_value.status = "DISABLED"
        elif change == "revoked": fixture.uow.agents.version_has_target.return_value = False
        elif change == "stripped_sources": payload.pop("sources")
        elif change == "injected_derivation": payload["facts"][0]["visualizations"] = [{"invented": True}]
        fixture.imported.content_hash = canonical_hash(payload)
        facts = [fact.model_dump(mode="json") for fact in fixture.facts]
        if change == "fact_row": facts[0]["rows"][0][0] = "forged"
        elif change == "fact_warning": facts[0]["warnings"] = []
        elif change == "fact_origin": facts[0]["provenance"]["native_lineage"]["series"][0]["dimensions"][0][1] = "forged"
        assert not await validate_investigation_sources(uow=fixture.uow, run=fixture.run, payload={"evidence": facts})
    asyncio.run(run())


def test_worker_lease_removes_invalid_history_and_its_old_judgments():
    async def run():
        fixture = await replay_fixture()
        imported = fixture.imported
        stale = TurnPlanningService._artifact(ops_run_id=fixture.run.ops_run_id, artifact_key="sufficiency:1",
            artifact_type="DBA_SUFFICIENCY", schema_version="DBA_SUFFICIENCY.v1", producer="test", trust_level="SOURCE_VERIFIED",
            payload=DbaSufficiencyAssessment(status="ANSWERABLE", evidence=fixture.facts).model_dump(mode="json"))
        artifacts = [imported, stale]
        fixture.uow.runs.list_tasks = AsyncMock(return_value=[])
        fixture.uow.runs.list_artifacts = AsyncMock(return_value=artifacts)
        runtime = object.__new__(AIOpsRuntimeService)
        task = SimpleNamespace(input_artifacts_json=[str(item.artifact_id) for item in artifacts])
        valid = await runtime._input_artifacts(fixture.uow, run_id=fixture.run.ops_run_id, task=task)
        assert valid[0].schema_version == "PRIOR_TURN_SQL_EVIDENCE.v1"
        fixture.uow.agents.version_has_target.return_value = False
        invalid = await runtime._input_artifacts(fixture.uow, run_id=fixture.run.ops_run_id, task=task)
        assert invalid[0].schema_version == "PRIOR_SQL_UNAVAILABLE.v1"
        assert not invalid[1].payload["evidence"]
        assert invalid[1].payload["status"] == "PARTIAL"
        assert invalid[1].payload["investigation"] is None
        assert invalid[1].payload["gaps"][0]["code"] == "PRIOR_SQL_SOURCE_UNAVAILABLE"
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_actual_report_and_case_paths_replay_history_and_reject_revoked_source(engine):
    from datetime import datetime, timezone
    from aiops_agent.application.case_source_validation import validate_case_sources
    from aiops_agent.application.diagnosis.investigation_record import freeze_investigation, render_investigation
    from aiops_agent.application.errors import AIOpsApplicationError
    from aiops_agent.application.reporting import resolve_system_template
    from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
    from aiops_agent.contracts.turn_answer import AIOpsTurnResult, TurnAnswerBlock

    async def run():
        fixture = await replay_fixture(engine)
        now = datetime(2026, 10, 28, 1, tzinfo=timezone.utc)
        fact, = fixture.facts
        field = "executions" if engine == "ORACLE" else "execution_count"
        draft = DiagnosisAnswerDraft(claims=({"claim_id": "c1", "kind": "FACT", "subject": "指定 SQL 的历史统计",
            "statement": "历史采集时点记录到指定 SQL；不等于当前窗口的增量。", "level": "OBSERVED",
            "applicable_time": fact.captured_at, "reasoning": "来自同会话原始工具行。",
            "evidence_fields": ({"evidence_ref": fact.evidence_ref, "row_index": 0, "field": field},),
        },), goal_satisfied=False)
        context = SimpleNamespace(run_id=fixture.run.ops_run_id, target_id=fixture.run.target_id, task_id=uuid7(),
            plan_snapshot=fixture.run.plan_snapshot_json)
        record = freeze_investigation(draft=draft, evidence=fixture.facts, context=context)
        analysis, _, refs = render_investigation(record)
        result = AIOpsTurnResult(status="PARTIAL", sufficiency_status="PARTIAL", evidence=fixture.facts,
            investigation_record=record, blocks=(TurnAnswerBlock(block_type="ANALYSIS_MARKDOWN",
                schema_version="AIOPS_ANALYSIS_BLOCK.v1", payload={"markdown": analysis}, evidence_refs=refs),))
        root = TurnPlanningService._artifact(ops_run_id=fixture.run.ops_run_id, artifact_key="turn:result", artifact_type="AIOPS_TURN_RESULT",
            schema_version=result.schema_version, payload=result.model_dump(mode="json"), producer="test", trust_level="SOURCE_VERIFIED")
        fixture.rows[root.artifact_id] = root
        fixture.run.final_artifact_id, fixture.run.status = root.artifact_id, "PARTIAL"
        fixture.run.created_at, fixture.run.trigger_type = now, "CHAT"
        async def add(row):
            row.artifact_id = row.artifact_id or uuid7()
            fixture.rows[row.artifact_id] = row
            return row
        fixture.uow.runs.add_artifact = add
        fixture.uow.runs.append_event = AsyncMock(return_value=SimpleNamespace(sequence_no=1))
        fixture.uow.inspections = SimpleNamespace(publish_report=AsyncMock(side_effect=lambda row: row), add_report_sources=AsyncMock())
        fixture.uow.outbox = SimpleNamespace(add=AsyncMock(side_effect=lambda row: row))
        fixture.uow.platform_notifications = SimpleNamespace(emit_report_ready=AsyncMock())
        runtime = object.__new__(AIOpsRuntimeService)
        async def publish():
            return await runtime._publish_diagnosis_report(uow=fixture.uow, run=fixture.run, task=SimpleNamespace(ops_task_id=uuid7()),
                source_artifact=root, now=now, trace_id="prior-sql-test", actor_id=fixture.run.actor_id,
                template=resolve_system_template("system:diagnosis.standard"))
        report = await publish()
        content = fixture.rows[report.content_artifact_id]
        assert content.payload_json["facts"][0]["applicable_time"] == fact.captured_at
        assert content.payload_json["facts"][0]["evidence_refs"] == [fact.evidence_ref]
        fixture.uow.runs.get_run_scoped = AsyncMock(return_value=fixture.run)
        fixture.uow.operations_knowledge = SimpleNamespace(get_version_relations=AsyncMock(return_value={"sources": [
            SimpleNamespace(source_report_id=None, source_artifact_id=root.artifact_id,
                source_run_id=fixture.run.ops_run_id, content_hash=root.content_hash)]}))
        fixture.uow.agents.get = AsyncMock(return_value=SimpleNamespace(current_version_id=fixture.run.agent_version_id))
        fixture.uow.agents.version_source_ids = AsyncMock(return_value=())
        fixture.uow.turns.list_turns = AsyncMock(return_value=[])
        assert await validate_case_sources(uow=fixture.uow, domain_id=fixture.run.domain_id, version_id=uuid7())
        fixture.raw.content_hash = "f" * 64
        assert not await validate_case_sources(uow=fixture.uow, domain_id=fixture.run.domain_id, version_id=uuid7())
        with pytest.raises(AIOpsApplicationError):
            await publish()
    asyncio.run(run())


def test_import_cannot_lower_original_source_security_level():
    async def run():
        fixture = await replay_fixture()
        fixture.raw.security_level = 3
        assert not await validate_reused_prior_sql(uow=fixture.uow, run=fixture.run, artifact=fixture.imported)
        fixture.imported.security_level = 3
        assert await validate_reused_prior_sql(uow=fixture.uow, run=fixture.run, artifact=fixture.imported)
    asyncio.run(run())


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("window", [False, True])
def test_removing_history_marker_cannot_bypass_source_replay(mapped, window):
    async def run():
        fixture = await replay_fixture(mapped=mapped, window=window)
        payload = {"evidence": [fact.model_dump(mode="json") for fact in fixture.facts]}
        payload["evidence"][0]["source_id"] = "forged-current-observation"
        assert not await validate_investigation_sources(uow=fixture.uow, run=fixture.run, payload=payload)
    asyncio.run(run())
