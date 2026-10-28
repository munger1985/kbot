"""真实保护回执进入跨轮SQL诊断、报告和案例；不访问目标库或实际模型。"""

import asyncio
import ast
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.case_source_validation import validate_case_sources
from aiops_agent.application.diagnosis.investigation_record import freeze_investigation, render_investigation
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.diagnosis.sql_plans import controlled_estimate_fact
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.investigation.sql_plan_evidence import (
    ARTIFACT_TYPE, SCHEMA, load_sql_plan_capture_evidence, validate_reused_plan_capture,
)
from aiops_agent.application.reporting import normalize_report_source, resolve_system_template
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.application.sql_plan_execution import RESULT_KEY, RESULT_SCHEMA
from aiops_agent.application.sql_plan_materials import _source_rows
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
from aiops_agent.contracts.sql_diagnosis import SqlDiagnosisDraft
from aiops_agent.contracts.sql_plan import SqlPlanDocument, SqlPlanAssessment
from aiops_agent.contracts.turn_answer import AIOpsTurnResult, TurnAnswerBlock, TurnEvidenceFact
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_capture_execution import execution_store, claimed, report
from tests.unit.aiops_agent.test_aiops_sql_capture_approval import call
from tests.unit.aiops_agent.test_aiops_sql_plan_material import SECRET
from tests.unit.aiops_agent.test_aiops_sql_comparison import answer_context
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import diagnosis


def scenario(engine="POSTGRESQL", *, status="SUCCEEDED"):
    store = execution_store(engine)
    background, task, _ = claimed(store)
    exchanged = call(store.execution.exchange("signed-test"))
    nodes = ({"node_id": 0, "operation": "Aggregate", "metrics": {"estimated_rows": "1"}},
        {"node_id": 1, "parent_id": 0, "operation": "Seq Scan", "schema_name": "mes", "relation_name": "t",
         "metrics": {"estimated_rows": "10000", "optimizer_total_cost": "123.5"}})
    physical = store.original.payload_json["tool_outcomes"][0]["observation"]["provenance"]["instance_identity"]
    updates = {"nodes": nodes, "physical_identity": physical} if status == "SUCCEEDED" else {"status": "FAILED", "nodes": (), "error_code": "PLAN_CAPTURE_FAILED"}
    call(store.execution.report(report(store, exchanged.resource_lease, **updates)))
    store.source, store.background = store.run, background
    store.original.artifact_key, store.original.ops_task_id = "source:sql", None
    store.original.artifact_type = "EVIDENCE"
    store.original.payload_uri, store.original.provenance_json = None, {}
    store.source.trace_id = "controlled-estimate-test"
    store.run.workflow_kind = "CHAT_TURN"
    store.run.plan_snapshot_json["target"] = {"row_version": store.target.row_version, "db_type": engine}
    current = deepcopy(store.run)
    current.ops_run_id, current.status = uuid7(), "RUNNING"
    store.all_runs.append(current)
    store.current, store.reader = current, store.execution.results
    subject = _source_rows(store.original, store.source, engine)[0].subject
    store.frame = {"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject,
        "problem_statement": "继续分析指定SQL并利用已经采集的估算计划", "objectives": ["DIAGNOSE"]}
    current.plan_snapshot_json.update(answer_context={"task_frame": store.frame}, capability_snapshot={"database_type": engine})
    turn_scope = {key: getattr(store.scope, key) for key in ("tenant_id", "domain_id") if hasattr(store.scope, key)}
    turn_scope.update(conversation_id=uuid7(), created_by=store.source.actor_id)
    store.previous = SimpleNamespace(**turn_scope, turn_id=uuid7(), turn_no=1, status="COMPLETED")
    store.turn = SimpleNamespace(**turn_scope, turn_id=uuid7(), turn_no=2, status="PLANNING")
    turns = {row.turn_id: row for row in (store.previous, store.turn)}
    store.turns = SimpleNamespace(get_turn=AsyncMock(side_effect=lambda **args: turns.get(args["turn_id"])),
        list_prior_turns=AsyncMock(return_value=[store.previous]),
        get_run_link=AsyncMock(side_effect=lambda **args: SimpleNamespace(ops_run_id=
            store.source.ops_run_id if args["turn_id"] == store.previous.turn_id else current.ops_run_id)),
        get_run_link_by_ops_run_id=AsyncMock(return_value=SimpleNamespace(turn_id=store.turn.turn_id)),
        add_evidence=AsyncMock(), list_turns=AsyncMock(return_value=[]))
    store.runs.list_plan_capture_results = AsyncMock(side_effect=lambda **args: [row for row in store.artifacts
        if row.ops_run_id == args["ops_run_id"] and row.schema_version == RESULT_SCHEMA
        and row.artifact_key.startswith(RESULT_KEY + ":")][:args["limit"]])
    store.runs.list_artifacts = AsyncMock(side_effect=lambda **args: [row for row in store.artifacts if row.ops_run_id == args["ops_run_id"]])
    store.runs.get_run_scoped = AsyncMock(side_effect=lambda **args: next((row for row in store.all_runs if row.ops_run_id == args["ops_run_id"]), None))
    store.runs.list_tasks = AsyncMock(return_value=[])
    store.runs.database_now = AsyncMock(side_effect=lambda: store.now)
    store.vault.open_for_capture = AsyncMock(side_effect=AssertionError("读取证据不能兑换代表参数"))
    store.secret_reader.read.reset_mock()
    store.planner = object.__new__(TurnPlanningService)
    store.planner._uow_factory = lambda: store
    store.planner._plan_capture_reader = store.reader
    store.runtime = object.__new__(AIOpsRuntimeService)
    store.runtime._plan_capture_reader = store.reader
    return store


async def load(store, **kwargs):
    return await load_sql_plan_capture_evidence(uow=store, turn=store.turn, run=store.current,
        task_frame=store.frame, reader=store.reader, **kwargs)


async def imported(store):
    entries = await load(store)
    assert len(entries) == 1
    artifact, = await store.planner._persist_sql_plan_capture_evidence(
        uow=store, run=store.current, turn=store.turn, entries=entries)
    return artifact


def worker_context(store, artifacts):
    return SimpleNamespace(plan_snapshot=store.current.plan_snapshot_json, run_id=store.current.ops_run_id,
        target_id=store.current.target_id, task_id=uuid7(), attempt=1, deadline_at=None, trigger_type="CHAT",
        input_artifacts=tuple({"artifact_id": str(row.artifact_id), "schema_version": row.schema_version,
            "payload": row.payload_json, "provenance": dict(getattr(row, "provenance_json", None) or {})} for row in artifacts))


async def diagnosed(store):
    artifact = await imported(store)
    current_stat = deepcopy(store.original)
    current_stat.artifact_id, current_stat.ops_run_id = uuid7(), store.current.ops_run_id
    await store.runs.add_artifact(current_stat)
    assessment = await DbaEvidenceAssessmentHandler().execute(worker_context(store, (current_stat, artifact)))
    return artifact, assessment


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_capture_to_next_turn_worker_bundle_and_frozen_record_preserves_estimate_boundaries(engine):
    store = scenario(engine)
    async def run():
        artifact, assessment = await diagnosed(store)
        fact = next(row for row in assessment.evidence if row.sql_plan is not None)
        assert fact.sql_plan.origin == "CONTROLLED_ESTIMATE" and fact.sql_plan.runtime_statistics == "NONE"
        assert not fact.sql_plan.original_parse_context_verified and fact.sql_plan.source_sql_sha256
        bundle = build_sql_evidence_bundle(task_frame=store.frame, database_type=engine, evidence=assessment.evidence)
        assert bundle["plan_assessments"][0]["identity_match"] == "SOURCE_MATCH"
        assert bundle["plan_assessments"][0]["physical_match"] == "SOURCE_MATCH"
        assert bundle["plan_assessments"][0]["origin"] == "CONTROLLED_ESTIMATE"
        assert bundle["plan_assessments"][0]["runtime_statistics"] == "NONE"
        assert not bundle["plan_assessments"][0]["production_execution_verified"]
        assert bundle["plan_assessments"][0]["representative_parameters"] == "UNVERIFIED"
        observation = next(row for row in bundle["observations"] if row["sql_plan"] is not None)
        assert observation["historical"] and observation["row_count"] == 2
        assert await validate_investigation_sources(uow=store, run=store.current,
            payload={"evidence": [fact.model_dump(mode="json")]}, plan_capture_reader=store.reader)
        assert not await validate_investigation_sources(uow=store, run=store.current,
            payload={"evidence": [fact.model_dump(mode="json")]})
        wire = str(artifact.payload_json) + fact.model_dump_json()
        assert SECRET not in wire and "private_schema" not in wire and "private_role" not in wire
        store.vault.open_for_capture.assert_not_awaited()
        store.secret_reader.read.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("change", ["actor", "domain", "agent", "agent_version", "target", "target_version", "target_disabled",
    "binding", "policy", "credential", "endpoint", "conversation", "source_turn", "source_hash", "material_hash",
    "result_rehash", "frozen_plan", "frozen_task", "ciphertext", "future_time", "mirror", "classification", "rehashed_target_snapshot"])
def test_reuse_refuses_changed_authority_or_rehashed_sources(change):
    store = scenario()
    async def run():
        artifact = await imported(store)
        if change in {"actor", "domain", "agent", "agent_version", "target"}:
            setattr(store.current, {"actor": "actor_id", "domain": "domain_id", "agent": "agent_id", "agent_version": "agent_version_id", "target": "target_id"}[change], uuid7())
        elif change == "target_version": store.target.row_version += 1
        elif change == "target_disabled": store.target.status = "DISABLED"
        elif change == "binding": store.binding.binding_id = uuid7()
        elif change == "policy": store.policy.rules_json["readonly_database_enabled"] = False
        elif change == "credential": store.target.diagnostic_credential_id = uuid7()
        elif change == "endpoint": store.target.endpoint_json = {**store.target.endpoint_json, "host": "other.invalid"}
        elif change == "conversation": store.previous.conversation_id = uuid7()
        elif change == "source_turn": store.previous.status = "FAILED"
        elif change == "source_hash":
            store.original.payload_json["tool_outcomes"][0]["observation"]["rows"][0][-1] = "99"
            store.original.content_hash = canonical_hash(store.original.payload_json)
        elif change == "material_hash":
            material = next(row for row in store.artifacts if row.artifact_key == "sql-plan-material:" + str(store.material.material_id))
            material.payload_json["parameter_count"] += 1
            material.content_hash = canonical_hash(material.payload_json)
        elif change == "result_rehash":
            native = next(row for row in store.artifacts if row.ops_run_id == store.background.ops_run_id and row.artifact_key == RESULT_KEY)
            native.payload_json["nodes"][1]["operation"] = "Index Scan"
            native.content_hash = canonical_hash(native.payload_json)
        elif change == "frozen_plan": store.background.plan_snapshot_json["blueprint"]["version"] = "2"
        elif change == "frozen_task": store.all_tasks[0].timeout_seconds += 1
        elif change == "ciphertext": store.credentials[-1].ciphertext = b"invalid"
        elif change == "future_time": store.now -= timedelta(seconds=1)
        elif change == "mirror":
            artifact.payload_json = deepcopy(artifact.payload_json)
            artifact.payload_json["document"]["nodes"][1]["operation"] = "Index Scan"
            artifact.content_hash = canonical_hash(artifact.payload_json)
        elif change == "classification": artifact.security_level = 1
        elif change == "rehashed_target_snapshot":
            store.target.row_version += 1
            for run in (store.current, store.source):
                run.plan_snapshot_json["investigation_execution"]["database"]["target_row_version"] = store.target.row_version
        assert not await validate_reused_plan_capture(uow=store, run=store.current, artifact=artifact, reader=store.reader)
        fact = controlled_estimate_fact({"artifact_id": str(artifact.artifact_id), "payload": artifact.payload_json})
        assert not await validate_investigation_sources(uow=store, run=store.current,
            payload={"evidence": [fact.model_dump(mode="json")]}, plan_capture_reader=store.reader)
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
@pytest.mark.parametrize("state", ["expired_material", "expired_approval", "cancelled_background", "failed_background"])
def test_safe_authoritative_result_survives_expiry_or_background_terminal_without_new_authority(engine, state):
    store = scenario(engine)
    if state in {"expired_material", "expired_approval"}: store.now += timedelta(hours=1)
    elif state == "cancelled_background": store.background.status = "CANCELLED"
    else: store.background.status = "FAILED"
    async def run():
        artifact = await imported(store)
        assert await validate_reused_plan_capture(uow=store, run=store.current, artifact=artifact, reader=store.reader)
        assert len(store.all_tasks) == 1
        store.vault.open_for_capture.assert_not_awaited()
        store.secret_reader.read.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("tamper", ["rows", "document", "time", "origin", "provenance", "trust", "semantics", "tool", "plan_missing"])
def test_formal_source_gate_replays_exact_projection_even_without_finding_or_graph(tamper):
    store = scenario()
    async def run():
        artifact = await imported(store)
        fact = controlled_estimate_fact({"artifact_id": str(artifact.artifact_id), "payload": artifact.payload_json}).model_dump(mode="json")
        if tamper == "rows": fact["rows"][1][2] = "Index Scan"
        elif tamper == "document": fact["sql_plan"]["nodes"][1]["metrics"]["estimated_rows"] = "1"
        elif tamper == "time": fact["captured_at"] = (store.now + timedelta(seconds=1)).isoformat()
        elif tamper == "origin": fact["sql_plan"]["origin"] = "USER_UPLOAD"
        elif tamper == "provenance": fact["provenance"]["content_hash"] = "a" * 64
        elif tamper == "trust": fact["trust_level"] = "USER_PROVIDED"
        elif tamper == "semantics": fact["measurement_semantics"] = "CURRENT_ACTIVITY"
        elif tamper == "tool": fact["tool_id"] = "db.sql.execution_plan"
        elif tamper == "plan_missing": fact["sql_plan"] = None
        assert not await validate_investigation_sources(uow=store, run=store.current,
            payload={"evidence": [fact]}, plan_capture_reader=store.reader)
    asyncio.run(run())


@pytest.mark.parametrize("change", ["sql_text", "sql_text_missing", "physical", "old_time"])
def test_same_identifier_does_not_prove_plan_text_physical_identity_or_time_match(change):
    store = scenario()
    async def run():
        _, assessment = await diagnosed(store)
        facts = list(assessment.evidence)
        index = next(index for index, fact in enumerate(facts) if fact.sql_plan is None)
        anchor = facts[index]
        if change in {"sql_text", "sql_text_missing"}:
            names = [column["name"] for column in anchor.columns]
            position = names.index("normalized_statement")
            rows = [list(row) for row in anchor.rows]
            rows[0][position] = "SELECT * FROM other.t" if change == "sql_text" else None
            facts[index] = anchor.model_copy(update={"rows": tuple(tuple(row) for row in rows)})
        elif change == "physical":
            index = next(index for index, fact in enumerate(facts) if fact.sql_plan is not None)
            fact = facts[index]
            facts[index] = fact.model_copy(update={"sql_plan": fact.sql_plan.model_copy(update={"physical_identity": {"database_oid": "99"}})})
        else: facts[index] = anchor.model_copy(update={"captured_at": (store.now + timedelta(hours=1)).isoformat()})
        bundle = build_sql_evidence_bundle(task_frame=store.frame, database_type="POSTGRESQL", evidence=tuple(facts))
        value = bundle["plan_assessments"][0]
        if change == "sql_text": assert value["identity_match"] == "MISMATCH"
        elif change == "sql_text_missing": assert value["identity_match"] == "INCOMPLETE"
        elif change == "physical": assert value["physical_match"] in {"MISMATCH", "INCOMPLETE"}
        else: assert value["time_relation"] == "HISTORICAL_OR_DIFFERENT_TIME"
    asyncio.run(run())


@pytest.mark.parametrize("prior", [{"facts": [{}] * 8}, {"facts": [{"rows": [[]] * 512}]},
    {"facts": [{"rows": [], "text": "x" * 262144}]}])
def test_plan_search_shares_history_fact_row_and_byte_budgets(prior):
    store = scenario()
    entries = call(load(store, prior_sql=prior))
    assert len(entries) == 1 and entries[0]["payload"] == {"code": "PLAN_CAPTURE_SEARCH_INCOMPLETE"}


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_failed_capture_cannot_turn_into_plan_evidence(engine):
    store = scenario(engine, status="FAILED")
    assert not call(load(store))


def test_revocation_between_discovery_and_commit_stores_only_explicit_gap_and_planner_links_keys():
    store = scenario()
    async def run():
        context = SimpleNamespace(domain_id=store.current.domain_id, turn_id=store.turn.turn_id,
            ops_run_id=store.current.ops_run_id, prior_sql_evidence=None, future_observation_evidence=(),
            source_run_evidence=None, resolved_uploads=(), recovery_assurance={})
        context.sql_plan_capture_evidence = await store.planner._load_sql_plan_capture_evidence(context, task_frame=store.frame)
        assert "turn-sql-plan-capture:1" in store.planner._initial_user_evidence_keys(context)
        store.target.row_version += 1
        artifact, = await store.planner._persist_sql_plan_capture_evidence(uow=store, run=store.current,
            turn=store.turn, entries=context.sql_plan_capture_evidence)
        assert artifact.schema_version == "SQL_PLAN_CAPTURE_UNAVAILABLE.v1"
        assert artifact.payload_json == {"code": "PLAN_CAPTURE_SOURCE_UNAVAILABLE"}
        assessment = await DbaEvidenceAssessmentHandler().execute(worker_context(store, (artifact,)))
        assert not assessment.evidence and any(gap.code == "PLAN_CAPTURE_SOURCE_UNAVAILABLE" for gap in assessment.gaps)
    asyncio.run(run())


def test_worker_lease_rechecks_access_and_removes_old_assessment_values_and_judgment():
    store = scenario()
    async def run():
        artifact, assessment = await diagnosed(store)
        prior = TurnPlanningService._artifact(ops_run_id=store.current.ops_run_id, artifact_key="sufficiency:1",
            artifact_type="DBA_SUFFICIENCY", schema_version="DBA_SUFFICIENCY.v1",
            payload=assessment.model_dump(mode="json"), producer="test", trust_level="SOURCE_VERIFIED")
        await store.runs.add_artifact(prior)
        task = SimpleNamespace(input_artifacts_json=[artifact.artifact_key, prior.artifact_key])
        selected = await store.runtime._input_artifacts(store, run_id=store.current.ops_run_id, task=task)
        assert any(row.schema_version == SCHEMA for row in selected)
        store.binding.binding_id = uuid7()
        selected = await store.runtime._input_artifacts(store, run_id=store.current.ops_run_id, task=task)
        assert any(row.schema_version == "SQL_PLAN_CAPTURE_UNAVAILABLE.v1" for row in selected)
        safe = next(row for row in selected if row.schema_version == "DBA_SUFFICIENCY.v1")
        assert not any(row.get("sql_plan") for row in safe.payload["evidence"])
        assert safe.payload["investigation"] is None
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_actual_formal_report_and_case_replay_preserve_plan_and_refuse_revoked_source(engine):
    store = scenario(engine)
    async def run():
        _, assessment = await diagnosed(store)
        fact = next(row for row in assessment.evidence if row.sql_plan is not None)
        draft = DiagnosisAnswerDraft(claims=({"claim_id": "c1", "kind": "FACT", "subject": "指定SQL的受控估算",
            "statement": "本次估算包含顺序扫描节点；不是实际运行计划或根因确认。", "level": "OBSERVED",
            "applicable_time": fact.captured_at, "reasoning": "隔离执行器权威回执的安全节点。",
            "evidence_fields": ({"evidence_ref": fact.evidence_ref, "row_index": 1, "field": "operation"},),
        },), goal_satisfied=False, unresolved_questions=("实际运行路径和参数代表性仍需核验。",))
        context = worker_context(store, ())
        record = freeze_investigation(draft=draft, evidence=assessment.evidence, context=context)
        assert record.sql_plan_assessments[0].origin == "CONTROLLED_ESTIMATE"
        record.validate_observations(assessment.evidence)
        analysis, _, refs = render_investigation(record)
        result = AIOpsTurnResult(status="PARTIAL", sufficiency_status="PARTIAL", evidence=assessment.evidence,
            investigation_record=record, blocks=(TurnAnswerBlock(block_type="ANALYSIS_MARKDOWN",
                schema_version="AIOPS_ANALYSIS_BLOCK.v1", payload={"markdown": analysis}, evidence_refs=refs),))
        root = TurnPlanningService._artifact(ops_run_id=store.current.ops_run_id, artifact_key="turn:result",
            artifact_type="AIOPS_TURN_RESULT", schema_version=result.schema_version, payload=result.model_dump(mode="json"),
            producer="test", trust_level="SOURCE_VERIFIED")
        await store.runs.add_artifact(root)
        store.current.final_artifact_id, store.current.status = root.artifact_id, "PARTIAL"
        store.current.created_at, store.current.trigger_type = store.now, "CHAT"
        original_add = store.runs.add_artifact
        async def add_artifact(row):
            row.artifact_id = row.artifact_id or uuid7()
            return await original_add(row)
        store.runs.add_artifact = add_artifact
        store.inspections = SimpleNamespace(publish_report=AsyncMock(side_effect=lambda row: row), add_report_sources=AsyncMock())
        store.outbox = SimpleNamespace(add=AsyncMock(side_effect=lambda row: row))
        store.platform_notifications = SimpleNamespace(emit_report_ready=AsyncMock())
        async def publish():
            return await store.runtime._publish_diagnosis_report(uow=store, run=store.current,
                task=SimpleNamespace(ops_task_id=uuid7()), source_artifact=root, now=store.now,
                trace_id="estimate-report-test", actor_id=store.current.actor_id,
                template=resolve_system_template("system:diagnosis.standard"))
        published = await publish()
        content = await store.runs.get_artifact(artifact_id=published.content_artifact_id)
        assert content.payload_json["scope"]["sql_plan_assessments"][0]["origin"] == "CONTROLLED_ESTIMATE"
        assert "不是实际运行计划" in content.payload_json["facts"][0]["summary"]
        from aiops_agent.application.reporting import report_presentation
        preview = report_presentation(payload=content.payload_json, template=resolve_system_template("system:diagnosis.standard"))
        appendix = next(row for row in preview["sections"] if row["kind"] == "EVIDENCE_APPENDIX")
        assert any("原解析环境和参数代表性未核验" in text for text in appendix["items"])
        assert normalize_report_source(schema_version=root.schema_version, payload=root.payload_json,
            source_kind="CHAT")["investigation_record"]["sql_plan_assessments"][0]["runtime_statistics"] == "NONE"
        store.operations_knowledge = SimpleNamespace(get_version_relations=AsyncMock(return_value={"sources": [
            SimpleNamespace(source_report_id=None, source_artifact_id=root.artifact_id,
                source_run_id=store.current.ops_run_id, content_hash=root.content_hash)]}))
        store.agents.get = AsyncMock(return_value=SimpleNamespace(current_version_id=store.current.agent_version_id))
        store.agents.version_has_target = AsyncMock(return_value=True)
        store.agents.version_source_ids = AsyncMock(return_value=())
        assert await validate_case_sources(uow=store, domain_id=store.current.domain_id,
            version_id=uuid7(), plan_capture_reader=store.reader)
        store.target.row_version += 1
        with pytest.raises(AIOpsApplicationError): await publish()
        assert store.inspections.publish_report.await_count == 1
        assert not await validate_case_sources(uow=store, domain_id=store.current.domain_id,
            version_id=uuid7(), plan_capture_reader=store.reader)
    asyncio.run(run())


@pytest.mark.parametrize("change", ["actual_rows", "runtime", "context_verified", "origin", "no_sql_hash"])
def test_estimate_contract_cannot_be_upgraded_to_runtime_plan_or_verified_parse_context(change):
    store = scenario()
    async def run():
        artifact = await imported(store)
        document = deepcopy(artifact.payload_json["document"])
        if change == "actual_rows": document["nodes"][1]["metrics"]["actual_rows_per_loop"] = "10000"
        elif change == "runtime": document["runtime_statistics"] = "PRESENT"
        elif change == "context_verified": document["original_parse_context_verified"] = True
        elif change == "origin": document["origin"] = "DATABASE_CURSOR"
        else: document["source_sql_sha256"] = None
        with pytest.raises(ValueError): SqlPlanDocument.model_validate(document)
    asyncio.run(run())


def test_cross_scope_prior_turn_is_rejected_before_reading_capture_artifacts():
    store = scenario()
    store.source.agent_id = uuid7()
    assert not call(load(store))
    store.runs.list_plan_capture_results.assert_not_awaited()


def test_budget_gap_survives_plan_commit_and_worker_without_any_node_values():
    store = scenario()
    async def run():
        entries = await load(store, prior_sql={"facts": [{}] * 8})
        artifact, = await store.planner._persist_sql_plan_capture_evidence(uow=store, run=store.current, turn=store.turn, entries=entries)
        assessment = await DbaEvidenceAssessmentHandler().execute(worker_context(store, (artifact,)))
        assert not assessment.evidence and any(gap.code == "PLAN_CAPTURE_SEARCH_INCOMPLETE" for gap in assessment.gaps)
    asyncio.run(run())


def test_actual_material_erasure_does_not_erase_sanitized_derived_plan_or_decrypt_old_parameters():
    store = scenario()
    call(store.service.revoke(scope=store.scope, run_id=store.source.ops_run_id, material_id=store.material.material_id))
    async def run():
        artifact = await imported(store)
        assert await validate_reused_plan_capture(uow=store, run=store.current, artifact=artifact, reader=store.reader)
        store.vault.open_for_capture.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_answer_compose_uses_captured_estimate_for_conditional_diagnosis_without_top_list(engine):
    store = scenario(engine)
    async def run():
        _, assessment = await diagnosed(store)
        plan = next(fact for fact in assessment.evidence if fact.sql_plan is not None)
        statistics = next(fact for fact in assessment.evidence if fact.sql_plan is None)
        field = "executions" if engine == "ORACLE" else "execution_count"
        draft = diagnosis(statistics, field=field, level="OBSERVED", goal=False).model_dump(mode="json")
        draft["conclusions"].extend((
            {"conclusion_id": "c2", "statement": "受控估算包含Seq Scan节点；不能当作生产实际扫描方式。",
             "level": "OBSERVED", "scope": "CONTEXT", "applicable_time": plan.captured_at,
             "evidence_fields": [{"evidence_ref": plan.evidence_ref, "row_index": 1, "field": "operation"}],
             "reasoning": "独立执行器回执已核验，节点描述的是采集环境的优化器估算。",
             "limitations": ["未核验原解析环境、参数代表性或实际执行行源。"]},
            {"conclusion_id": "c3", "statement": "过滤后扫描范围过大是可验证的成本假设，不是已确认根因。",
             "level": "HYPOTHESIS", "scope": "SUBJECT", "applicable_time": plan.captured_at,
             "evidence_fields": [{"evidence_ref": statistics.evidence_ref, "row_index": 0, "field": field},
                 {"evidence_ref": plan.evidence_ref, "row_index": 1, "field": "operation"}],
             "reasoning": "把语句工作量与已匹配估算联系起来，仍需业务参数、对象目录及实际计划验证。",
             "alternatives": ["大范围业务查询可能本来就适合顺序扫描。"],
             "limitations": ["优化器成本不是毫秒，估算扫描不证明生产使用相同路径。"]},
        ))
        draft["recommendations"].append({
            "recommendation_id": "r2", "conclusion_ids": ["c2", "c3"], "intent": "VERIFY",
            "action": "利用已有估算核对过滤列、选择性和索引目录；在代表参数下验证扫描范围，再评估组合索引。",
            "preconditions": ["确认参数业务代表性及计划采集上下文", "先完成实际执行风险评估与授权"],
            "risks": ["实际计划采集可能消耗生产资源，新增索引会增加写入成本。"],
            "validation": ["同负载可比窗口比较延迟、逻辑读及扫描行数，确认是否使用候选索引。"],
            "rollback_or_stop": "出现额外生产负载立即停止采集；索引变更需另行审批和回滚方案。",
            "requires_approval": True,
        })
        draft["unresolved_questions"] = ["原解析环境、参数代表性与实际运行路径仍未核验。"]
        model = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(
            output=SqlDiagnosisDraft.model_validate(draft), receipt=SimpleNamespace(model_dump=lambda **_: {}))))
        prompts = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="估算诊断", ref=lambda: {})))
        context = answer_context(store.frame, assessment.evidence, engine)
        result = await DbaAnswerComposeHandler(model_client=model, prompts=prompts).execute(context)
        assert model.generate_structured.await_count == 1
        bundle = model.generate_structured.await_args.kwargs["input_payload"]["evidence_bundle"]
        match = bundle["plan_assessments"][0]
        assert match["origin"] == "CONTROLLED_ESTIMATE"
        assert match["identity_match"] == match["physical_match"] == "SOURCE_MATCH"
        assert match["time_relation"] == "SOURCE_NEAR_OBSERVATION"
        assert match["runtime_statistics"] == "NONE" and not match["production_execution_verified"]
        assert result.investigation_record is not None
        assert result.investigation_record.sql_plan_assessments[0].origin == "CONTROLLED_ESTIMATE"
        assert result.status == "PARTIAL" and result.blocks[0].payload["sql_diagnosis"] is not None
        solution = next(block for block in result.blocks if str(block.block_type) == "SOLUTION_MARKDOWN")
        assert "利用已有估算" in solution.payload["markdown"] and "组合索引" in solution.payload["markdown"]
        assert all(block.payload.get("result_kind") != "TOP_SQL_LIST" for block in result.blocks)
        assert SECRET not in str(model.generate_structured.await_args.kwargs) + result.model_dump_json()
        store.vault.open_for_capture.assert_not_awaited()
        store.secret_reader.read.assert_not_awaited()
    asyncio.run(run())


def test_future_observation_consumes_the_same_history_budget_as_captured_plans():
    store = scenario()
    payload = deepcopy(store.original.payload_json)
    observation = payload["tool_outcomes"][0]["observation"]
    observation["rows"] *= 512
    observation["row_count"] = len(observation["rows"])
    entries = call(load(store, future=({"payload": payload},)))
    assert len(entries) == 1 and entries[0]["payload"] == {"code": "PLAN_CAPTURE_SEARCH_INCOMPLETE"}


@pytest.mark.parametrize("entrypoint,consumers", [
    ("api", {"AIOpsRuntimeService", "OperationsKnowledgeService"}),
    ("worker", {"AIOpsRuntimeService", "OperationsKnowledgeService", "TurnPlanningService"}),
])
def test_bootstrap_injects_authoritative_reader_into_every_production_consumer(entrypoint, consumers):
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / f"services/aiops_agent/src/aiops_agent/bootstrap/{entrypoint}.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    readers = [node for node in calls if node.func.id == "SqlPlanCaptureResultReader"]
    assert len(readers) == 1
    for consumer in consumers:
        constructors = [node for node in calls if node.func.id == consumer]
        assert constructors
        for node in constructors:
            reference = next(keyword.value for keyword in node.keywords if keyword.arg == "plan_capture_reader")
            assert isinstance(reference, ast.Name) and reference.id == "plan_capture_reader"


@pytest.mark.parametrize("requested,expected", [(0, 1), (4, 4), (10000, 17)])
def test_repository_query_is_source_run_scoped_and_bounded_before_loading_rows(requested, expected):
    from aiops_agent.repositories.runtime import OpsRunRepository
    from aiops_agent.entities import OpsArtifactEntity
    from sqlalchemy.dialects import postgresql
    run_id = uuid7()
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: ())))
    assert call(OpsRunRepository(session).list_plan_capture_results(ops_run_id=run_id, limit=requested)) == []
    statement = session.execute.await_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    query = str(compiled)
    assert "LIMIT" in query and compiled.params["param_1"] == expected
    assert run_id in compiled.params.values()
    assert "SQL_PLAN_CAPTURE_RESULT.v1" in compiled.params.values()
    assert "sql-plan-capture:result:" in compiled.params.values()
    assert OpsArtifactEntity.ops_run_id.name in query


def test_estimate_node_alone_cannot_certify_subject_access_path_or_root_cause():
    from aiops_agent.application.diagnosis.sql_diagnosis import validate_sql_diagnosis
    store = scenario()
    async def run():
        _, assessment = await diagnosed(store)
        plan = next(fact for fact in assessment.evidence if fact.sql_plan is not None)
        bundle = build_sql_evidence_bundle(task_frame=store.frame, database_type="POSTGRESQL", evidence=assessment.evidence)
        with pytest.raises(ValueError, match="背景证据不能单独"):
            validate_sql_diagnosis(diagnosis(plan, field="operation", row_index=1), bundle)
    asyncio.run(run())


def test_builtin_prompt_updates_content_without_changing_platform_seed_version():
    from platform_core.prompts.catalog import load_prompt_catalog
    entry = next(item for item in load_prompt_catalog().entries if item.prompt_key == "aiops_agent.sql_diagnosis")
    assert entry.version == "1.0.0"
    assert "origin=CONTROLLED_ESTIMATE" in entry.content and "条件化措施" in entry.content
    assert "不表示生产实际执行" in entry.content
