"""三库 SQL 统计主体先消歧，选行续问保持完整维度，不由模型任取第一条。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.sql_selection import decode_selection, subject_from_row
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import observation, task_frame
from tests.unit.aiops_agent.test_aiops_measurements import NOW


def scenario(engine):
    if engine == "POSTGRESQL":
        frame = task_frame()
        fact = observation(
            names=("query_id", "database_oid", "user_oid", "top_level", "execution_count"),
            rows=(("1285382359565720997", 16510, 10, True, 100), ("1285382359565720997", 16510, 20, True, 200)),
        )
    elif engine == "MYSQL":
        frame = {**task_frame(), "subject_ref": {"subject_type": "MYSQL_DIGEST", "digest": "a" * 64}}
        fact = observation(
            names=("digest", "schema_name", "execution_count"),
            rows=(("a" * 64, "orders", 100), ("a" * 64, "MES", 200)),
            tool_id="db.sql.cursor_details",
        )
    else:
        frame = {**task_frame(), "subject_ref": {"subject_type": "ORACLE_SQL_ID", "sql_id": "36apy6dncjggq"}}
        fact = observation(
            names=("sql_id", "inst_id", "child_number", "plan_hash_value", "executions", "dbid", "container_id"),
            rows=(("36apy6dncjggq", 1, 0, 100, 100, 42, 3), ("36apy6dncjggq", 1, 1, 200, 200, 42, 3)),
            tool_id="db.sql.cursor_details",
        )
    return frame, fact


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_complete_scopes_are_separate_comparison_subjects_not_merged(engine):
    frame, fact = scenario(engine)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type=engine, evidence=(fact,))
    assert bundle["subject_resolution"]["status"] == "COMPARISON"
    assert bundle["subject_resolution"]["observed_candidate_count"] == 2
    assert {row["role"] for row in bundle["observations"][0]["records"]} == {"SUBJECT"}
    assert len({item["subject_id"] for item in bundle["subjects"]}) == 2
    assert bundle["object_graph"] is None
    assert not bundle["comparison_policy"]["implicit_aggregation"]


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_selecting_a_real_row_resolves_exact_scope_without_aggregating(engine):
    frame, fact = scenario(engine)
    frame["subject_ref"] = subject_from_row(fact, 1, engine)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type=engine, evidence=(fact,))
    assert bundle["subject_resolution"]["status"] == "RESOLVED"
    assert [row["role"] for row in bundle["observations"][0]["records"]] == ["CONTEXT", "SUBJECT"]


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_candidate_table_points_to_original_artifact_and_original_row(engine):
    frame, fact = scenario(engine)
    fact = fact.model_copy(update={"truncated": True})
    assessment = DbaSufficiencyAssessment(
        status="NEEDS_CLARIFICATION", evidence=(fact,),
        clarification_question="请选择统计主体",
    )
    artifact_id = str(uuid7())
    context = SimpleNamespace(
        plan_snapshot={"answer_context": {"task_frame": frame}, "capability_snapshot": {"database_type": engine}},
        input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "artifact_id": artifact_id},),
        trigger_type="MANUAL",
    )
    blocks = DbaAnswerComposeHandler._waiting_result(assessment, context).blocks
    table = next(item for item in blocks if item.block_type == "TABLE")
    assert table.payload["result_kind"] == "SQL_SUBJECT_CANDIDATES"
    assert table.payload["follow_up_label"] == "选择并诊断"
    selections = [decode_selection(value) for value in table.payload["row_selections"]]
    assert [value["row_index"] for value in selections] == [0, 1]
    assert all(value["artifact_id"] == artifact_id for value in selections)


def test_historical_other_scope_does_not_expand_current_candidates():
    frame, fact = scenario("POSTGRESQL")
    current = fact.model_copy(update={"rows": (fact.rows[0],), "row_count": 1})
    old = fact.model_copy(update={"source_id": "prior_turn.sql_observation"})
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=(current, old))
    assert bundle["subject_resolution"]["status"] == "RESOLVED"


def test_current_session_duplicates_are_not_statement_scope_ambiguity():
    frame, fact = scenario("POSTGRESQL")
    fact = fact.model_copy(update={"rows": (fact.rows[0], fact.rows[0])})
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=(fact,))
    assert bundle["subject_resolution"]["observed_candidate_count"] == 1


def test_oracle_selection_dimensions_survive_planning_including_zero_child():
    frame, fact = scenario("ORACLE")
    subject = subject_from_row(fact, 0, "ORACLE")
    compact = SimpleNamespace(diagnostic_profile="SINGLE_SQL_PERFORMANCE", subject_ref=subject)
    resolved = TurnPlanningService._single_sql_subject(compact, {"db_type": "ORACLE"})
    assert resolved["inst_id"] == 1
    assert resolved["child_number"] == 0
    assert resolved["plan_hash_value"] == 100


@pytest.mark.parametrize("key,value", [("inst_id", 0), ("child_number", -1), ("plan_hash_value", True), ("dbid", 2**65)])
def test_invalid_oracle_dimensions_are_not_silently_discarded(key, value):
    frame, _ = scenario("ORACLE")
    compact = SimpleNamespace(diagnostic_profile="SINGLE_SQL_PERFORMANCE", subject_ref={**frame["subject_ref"], key: value})
    assert TurnPlanningService._single_sql_subject(compact, {"db_type": "ORACLE"}) is None


def test_ambiguity_stops_before_model_assessment_and_is_nonretryable():
    async def execute():
        frame, fact = scenario("POSTGRESQL")
        fact = fact.model_copy(update={"truncated": True})
        model = SimpleNamespace(generate_structured=AsyncMock())
        handler = DbaEvidenceAssessmentHandler(model_client=model, prompts=SimpleNamespace())
        context = SimpleNamespace(
            input_artifacts=({"schema_version": "PRIOR_TURN_SQL_EVIDENCE.v1", "artifact_id": str(uuid7()), "payload": {"facts": []}},),
            plan_snapshot={"answer_context": {"task_frame": frame, "model": {"configured": True}}, "capability_snapshot": {"database_type": "POSTGRESQL"}},
            trigger_type="MANUAL",
        )
        from tests.unit.aiops_agent.test_aiops_turn_answer import _tool_artifact
        artifact = _tool_artifact(semantics="CUMULATIVE_SINCE_LOAD", row_count=2, tool_id=fact.tool_id)
        raw = artifact["payload"]["tool_outcomes"][0]["observation"]
        raw.update(
            db_type="POSTGRESQL", db_version="16",
            columns=[{**column, "sensitivity": "PUBLIC"} for column in fact.columns],
            rows=fact.rows, captured_at=NOW.isoformat(),
            truncated=True,
        )
        context.input_artifacts = (artifact,)
        assessment = await handler.execute(context)
        assert assessment.status == "NEEDS_CLARIFICATION"
        assert any(gap.code == "SQL_SUBJECT_AMBIGUOUS" and not gap.retryable for gap in assessment.gaps)
        model.generate_structured.assert_not_awaited()
    asyncio.run(execute())
