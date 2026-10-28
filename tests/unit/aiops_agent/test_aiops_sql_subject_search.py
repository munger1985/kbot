"""主体发现、跨轮搜索和多主体诊断的正向链及隔离回归；不替代三库实测。"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.investigation_record import sql_investigation_draft
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle, validate_sql_diagnosis
from aiops_agent.application.investigation.prior_sql_evidence import load_prior_sql_evidence
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.diagnostics.validation import validate_parameters
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import diagnosis, observation, task_frame, selection_scope
from tests.unit.aiops_agent.test_aiops_sql_subject_resolution import scenario


def search_fixture():
    scope = {key: uuid7() for key in ("tenant_id", "domain_id", "agent_id", "agent_version_id", "actor_id", "target_id")}
    conversation_id = uuid7()
    target = {"row_version": 1, "db_type": "POSTGRESQL"}
    run = SimpleNamespace(**scope, ops_run_id=uuid7(), plan_snapshot_json={"target": target})
    turn = SimpleNamespace(tenant_id=scope["tenant_id"], domain_id=scope["domain_id"], conversation_id=conversation_id,
                           created_by=scope["actor_id"], turn_no=10)
    entry = observation(names=("query_id", "database_oid", "user_oid", "top_level", "execution_count"),
                        rows=(("42", 1, 2, True, 100), ("99", 1, 2, True, 999)))
    assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(entry,))
    artifact = SimpleNamespace(artifact_id=uuid7(), ops_run_id=uuid7(), schema_version="DBA_SUFFICIENCY.v1",
                               trust_level="SOURCE_VERIFIED", payload_json=assessment.model_dump(mode="json"))
    artifact.content_hash = canonical_hash(artifact.payload_json)
    previous = SimpleNamespace(tenant_id=scope["tenant_id"], domain_id=scope["domain_id"], conversation_id=conversation_id,
        created_by=scope["actor_id"], turn_id=uuid7(), turn_no=3, status="COMPLETED", sufficiency_artifact_id=artifact.artifact_id)
    old_run = SimpleNamespace(**scope, ops_run_id=artifact.ops_run_id, plan_snapshot_json={"target": deepcopy(target)})
    latest = SimpleNamespace(**{**vars(previous), "turn_id": uuid7(), "turn_no": 9, "status": "FAILED", "sufficiency_artifact_id": None})
    uow = SimpleNamespace(
        turns=SimpleNamespace(list_prior_turns=AsyncMock(return_value=[latest, previous]),
                              get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=artifact.ops_run_id))),
        runs=SimpleNamespace(get_run=AsyncMock(return_value=old_run), get_artifact=AsyncMock(return_value=artifact)),
        targets=SimpleNamespace(get_scoped=AsyncMock(return_value=SimpleNamespace(status="ENABLED", **target))),
        agents=SimpleNamespace(version_has_target=AsyncMock(return_value=True)),
    )
    frame = {**task_frame(), "subject_ref": {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42"}}
    return SimpleNamespace(uow=uow, turn=turn, run=run, previous=previous, old_run=old_run,
                           artifact=artifact, entry=entry, frame=frame)


def load(fixture):
    return asyncio.run(load_prior_sql_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run, task_frame=fixture.frame))


def test_search_skips_failed_intervening_turn_and_keeps_exact_original_lineage():
    fixture = search_fixture()
    result = load(fixture)
    assert result["search"]["scanned_turns"] == 2
    assert result["facts"][0]["rows"] == [["42", 1, 2, True, 100]]
    assert result["facts"][0]["captured_at"] == fixture.entry.captured_at
    assert result["sources"][0]["source_row_indices"] == [0]
    assert result["sources"][0]["source_content_hash"] == fixture.artifact.content_hash
    fixture.uow.turns.list_prior_turns.assert_awaited_once_with(
        conversation_id=fixture.turn.conversation_id, before_turn_no=10, limit=17,
    )


@pytest.mark.parametrize("problem", ["hash", "trust", "agent_version", "actor", "target_version", "disabled", "revoked", "other_subject"])
def test_history_does_not_bypass_current_authority_source_integrity_or_identity(problem):
    fixture = search_fixture()
    if problem == "hash":
        fixture.artifact.content_hash = "0" * 64
    elif problem == "trust":
        fixture.artifact.trust_level = "USER_PROVIDED"
    elif problem == "agent_version":
        fixture.old_run.agent_version_id = uuid7()
    elif problem == "actor":
        fixture.old_run.actor_id = uuid7()
    elif problem == "target_version":
        fixture.old_run.plan_snapshot_json["target"]["row_version"] = 2
    elif problem == "disabled":
        fixture.uow.targets.get_scoped.return_value.status = "DISABLED"
    elif problem == "revoked":
        fixture.uow.agents.version_has_target.return_value = False
    else:
        fixture.frame["subject_ref"]["query_id"] = "123"
    assert load(fixture) is None


def test_search_is_bounded_and_does_not_scan_the_seventeenth_turn():
    fixture = search_fixture()
    fixture.uow.turns.list_prior_turns.return_value = [fixture.previous] * 17
    result = load(fixture)
    assert result["search"]["scanned_turns"] == 16 and result["search"]["truncated"]
    assert len(result["facts"]) == 1
    assert fixture.uow.runs.get_artifact.await_count == 16


def test_history_byte_budget_does_not_accept_an_oversized_matching_sql():
    fixture = search_fixture()
    raw = fixture.artifact.payload_json["evidence"][0]
    raw["columns"].append({"name": "normalized_statement", "logical_type": "STRING"})
    raw["rows"][0].append("x" * 262144)
    raw["rows"][1].append("other")
    fixture.artifact.content_hash = canonical_hash(fixture.artifact.payload_json)
    result = load(fixture)
    assert result["facts"] == [] and result["search"]["truncated"]


def test_explicit_selection_is_revalidated_at_collection_not_copied_from_snapshot():
    fixture = selection_scope()
    from aiops_agent.application.sql_selection import resolve_sql_selection
    resolved = asyncio.run(resolve_sql_selection(uow=fixture.uow, reference=fixture.reference,
        conversation=fixture.conversation, actor_id=fixture.actor))
    fixture.run.plan_snapshot_json["client_metadata"] = {"sql_selection": resolved}
    fixture.turn.turn_no = 2
    fixture.uow.conversations = SimpleNamespace(get_conversation=AsyncMock(return_value=fixture.conversation))
    frame = task_frame()
    assert asyncio.run(load_prior_sql_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run, task_frame=frame))
    fixture.artifact.content_hash = "0" * 64
    assert asyncio.run(load_prior_sql_evidence(uow=fixture.uow, turn=fixture.turn, run=fixture.run, task_frame=frame)) is None


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_multi_subject_diagnosis_and_comparison_require_each_exact_subjects_own_evidence(engine):
    frame, fact = scenario(engine)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type=engine, evidence=(fact,))
    keys = [item["subject_id"] for item in bundle["subjects"]]
    field = "executions" if engine == "ORACLE" else "execution_count"
    draft = diagnosis(fact, field=field)
    own = draft.conclusions[0].model_copy(update={"subject_ids": (keys[0],)})
    validate_sql_diagnosis(draft.model_copy(update={"conclusions": (own,)}), bundle)
    wrong = own.model_copy(update={"subject_ids": (keys[1],)})
    with pytest.raises(ValueError):
        validate_sql_diagnosis(draft.model_copy(update={"conclusions": (wrong,)}), bundle)
    compare = own.model_copy(update={"scope": "COMPARISON", "subject_ids": tuple(keys),
        "evidence_fields": (*own.evidence_fields, own.evidence_fields[0].model_copy(update={"row_index": 1}))})
    validate_sql_diagnosis(draft.model_copy(update={"conclusions": (compare,)}), bundle)
    frozen = sql_investigation_draft(draft.model_copy(update={"conclusions": (compare,)}), "SQL", bundle["subjects"])
    assert all(str(value) in frozen.claims[0].subject for value in bundle["subjects"][1]["dimensions"].values())
    with pytest.raises(ValueError):
        validate_sql_diagnosis(draft.model_copy(update={"conclusions": (compare.model_copy(update={"evidence_fields": own.evidence_fields}),)}), bundle)


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_comparison_table_can_select_original_row_without_reprinting_top_sql(engine):
    frame, fact = scenario(engine)
    assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,))
    context = SimpleNamespace(plan_snapshot={"answer_context": {"task_frame": frame}, "capability_snapshot": {"database_type": engine}},
                              input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "artifact_id": str(uuid7())},))
    table, = DbaAnswerComposeHandler._sql_candidate_blocks(assessment, context, comparison=True)
    assert table.payload["result_kind"] == "SQL_SUBJECT_COMPARISON"
    assert table.payload["row_count"] == 2 and len(table.payload["row_selections"]) == 2


def test_postgresql_can_start_identity_discovery_without_guessing_database_oid():
    subject = TurnPlanningService._single_sql_subject(
        SimpleNamespace(diagnostic_profile="SINGLE_SQL_PERFORMANCE", subject_ref={"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "-42"}),
        {"db_type": "POSTGRESQL"},
    )
    assert subject == {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "-42"}


def test_mysql_explicit_null_schema_is_an_identity_not_a_missing_value():
    subject = TurnPlanningService._single_sql_subject(
        SimpleNamespace(diagnostic_profile="SINGLE_SQL_PERFORMANCE", subject_ref={"subject_type": "MYSQL_DIGEST", "digest": "a" * 64, "schema_name": None}),
        {"db_type": "MYSQL"},
    )
    assert "schema_name" in subject and subject["schema_name"] is None


def test_subject_after_fortieth_row_is_projected_with_original_row_index():
    rows = tuple((str(index), 16510, 100, 42) for index in range(60)) + (("1285382359565720997", 16510, 100, 42),)
    fact = observation(rows=rows)
    bundle = build_sql_evidence_bundle(task_frame=task_frame(), database_type="POSTGRESQL", evidence=(fact,))
    matched = [row for row in bundle["observations"][0]["records"] if row["role"] == "SUBJECT"]
    assert len(matched) == 1 and matched[0]["row_index"] == 60


def test_conflicting_text_for_same_complete_identity_is_not_automatically_compared():
    frame, fact = scenario("POSTGRESQL")
    fact = fact.model_copy(update={"columns": (*fact.columns, {"name": "normalized_statement"}),
                                  "rows": ((*fact.rows[0], "SELECT 1"), (*fact.rows[0], "SELECT 2"))})
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=(fact,))
    assert bundle["subject_resolution"]["status"] == "AMBIGUOUS"
    assert bundle["subject_resolution"]["identity_conflict"]


@pytest.mark.parametrize("version", ["12", "13", "16", "18"])
def test_discovery_catalog_resolves_real_versioned_template_without_required_database_oid(version):
    tool = DiagnosticRegistry.load().resolve(tool_id="db.postgresql.statement.subjects", tool_version="1.0.0",
        db_type="POSTGRESQL", db_version=version, capabilities={"pg_stat_statements"}, entitlements=set())
    parameters = validate_parameters(tool.definition, {"query_id": "-42"})
    assert parameters["database_oid"] == 0 and parameters["user_oid"] == 0
    assert "(CAST(:database_oid AS bigint) = 0 OR CAST(statements.dbid AS bigint) = CAST(:database_oid AS bigint))" in tool.sql
    assert "(CAST(:user_oid AS bigint) = 0 OR CAST(statements.userid AS bigint) = CAST(:user_oid AS bigint))" in tool.sql
    high_oids = validate_parameters(tool.definition, {"query_id": "-42", "database_oid": 4294967295, "user_oid": 4294967295})
    assert high_oids["user_oid"] == 4294967295
    assert "statements.queryid = CAST(CAST(:query_id AS text) AS bigint)" in tool.sql
    if int(version) >= 14:
        assert parameters["include_nested"] is True
        assert "(:include_nested OR statements.toplevel = :top_level)" in tool.sql


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_comparison_reaches_model_and_frozen_report_instead_of_stopping_for_arbitrary_selection(engine):
    frame, fact = scenario(engine)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type=engine, evidence=(fact,))
    field = "executions" if engine == "ORACLE" else "execution_count"
    draft = diagnosis(fact, field=field)
    conclusion = draft.conclusions[0].model_copy(update={"scope": "COMPARISON",
        "subject_ids": tuple(item["subject_id"] for item in bundle["subjects"]),
        "evidence_fields": (*draft.conclusions[0].evidence_fields,
                            draft.conclusions[0].evidence_fields[0].model_copy(update={"row_index": 1}))})
    draft = draft.model_copy(update={"conclusions": (conclusion,)})
    model = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(
        output=draft, receipt=SimpleNamespace(model_dump=lambda **_: {}))))
    context = TaskExecutionContext(run_id=uuid7(), task_id=uuid7(), task_key="answer:compose", target_id=uuid7(),
        agent_id=uuid7(), actor_id=uuid7(), trigger_type="MANUAL", trace_id="test", attempt=1,
        deadline_at=None, policy_snapshot={}, plan_snapshot={"capability_snapshot": {"database_type": engine},
        "answer_context": {"task_frame": frame, "model": {}, "prompts": {}}},
        input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "artifact_id": str(uuid7()),
                          "payload": DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,)).model_dump(mode="json")},))
    result = asyncio.run(DbaAnswerComposeHandler(model_client=model,
        prompts=SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="诊断", ref=lambda: {})))).execute(context))
    assert result.investigation_record is not None
    assert "统计主体" in result.blocks[0].payload["markdown"]
    assert any(block.payload.get("result_kind") == "SQL_SUBJECT_COMPARISON" for block in result.blocks)
    assert not any(block.payload.get("result_kind") == "TOP_SQL_LIST" for block in result.blocks)
    assert len(result.investigation_record.claims[0].evidence_fields) == 2
    assert model.generate_structured.await_count == 1
