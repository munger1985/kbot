"""固定采集到最终回答的三库图形合同；缺计划保留真实状态且不改模型正文。"""

import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.diagnosis.sql_plan_presentation import matching_plan_refs
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceGap
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.executor.drivers.base import DiagnosticDriverError
from aiops_agent.executor.drivers.statement_plan import PLAN_COLUMNS, mysql_statement_plan, postgresql_statement_plan, readonly_plan_sql
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.contracts.aiops.executor import DatabaseColumn
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_oracle_cursor_plan import direct, fact, observation
from tests.unit.aiops_agent.test_aiops_sql_plan_import import SUBJECTS, PHYSICAL, anchor
from tests.unit.aiops_agent.test_aiops_sql_plan_priority import frame
from tests.unit.aiops_agent.test_aiops_turn_answer import _context


def sample_cursor(**updates):
    sql = "SELECT * FROM mes.orders WHERE id = 12"
    source = dict(thread_id=None, event_id=None, connection_id=None, digest="A" * 64, schema_name="mes",
        source_sql="SELECT * FROM mes.orders WHERE id = ?", sample_sql=sql, sample_digest="A" * 64,
        sample_complete=1, sample_captured_at="2026-11-09 09:00:00", current_database="mes")
    source.update(updates)
    row = tuple(source.values())
    cursor = SimpleNamespace(description=[(key,) for key in source], execute=AsyncMock(),
        fetchmany=AsyncMock(side_effect=[[row], [row]]), fetchone=AsyncMock(return_value=(json.dumps({
            "query_block": {"table": {"table_name": "orders", "access_type": "ref", "key": "orders_idx",
                "rows_examined_per_scan": 1, "cost_info": {"read_cost": "1.2"}}}}),)))
    return source, cursor


def test_mysql_sample_is_explained_verbatim_without_replaying_or_guessing_digest_parameters():
    async def exercise():
        source, cursor = sample_cursor()
        rows = await mysql_statement_plan(cursor, bound_source_sql="SOURCE", ensure_active=lambda: None,
            parameters={"source_sql_sha256": canonical_hash(source["source_sql"])})
        assert rows and json.loads(rows[0][0])["schema_name"] == "mes"
        commands = [call.args[0] for call in cursor.execute.await_args_list]
        assert commands == ["SOURCE", "EXPLAIN FORMAT=JSON " + source["sample_sql"], "SOURCE"]
        assert source["sample_sql"] not in rows[0][2] and "WHERE" not in rows[0][2]
    asyncio.run(exercise())


def test_mysql_explain_notes_never_log_statement_literals():
    from aiops_agent.executor.drivers.mysql import _PlanCaptureCursor
    import warnings
    from loguru import logger
    async def exercise():
        cursor = object.__new__(_PlanCaptureCursor)
        cursor._result = None
        connection = SimpleNamespace(show_warnings=AsyncMock(return_value=[("Note", 1003, "PRIVATE_SQL_AND_LITERAL")] ))
        messages = []
        sink = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
        try:
            with warnings.catch_warnings(record=True) as caught:
                await cursor._show_warnings(connection)
            assert not caught
            assert "PRIVATE_SQL_AND_LITERAL" not in "".join(messages)
            assert "1003" in "".join(messages)
            connection.show_warnings.assert_awaited_once()
        finally:
            logger.remove(sink)
    asyncio.run(exercise())


def test_mysql_native_date_and_conditional_builtins_use_their_dialect_semantics():
    sql = "SELECT IFNULL(SUM(x), 0), NOW(), DATE(d), DATEDIFF(d, NOW()), TIMESTAMPDIFF(DAY, d, NOW()) FROM orders"
    assert readonly_plan_sql(sql, dialect="mysql") == sql
    with pytest.raises(DiagnosticDriverError, match="SQL_PLAN_FUNCTION_UNSUPPORTED"):
        readonly_plan_sql("SELECT DATE(evil.custom_func()) FROM orders", dialect="mysql")


@pytest.mark.parametrize("tamper,expected", [(None, "SOURCE_SERVER_MATCH"), ("server", "MISMATCH"),
    ("missing", "INCOMPLETE"), ("untrusted", "INCOMPLETE"), ("historical", "INCOMPLETE"), ("text", "INCOMPLETE")])
def test_server_scope_only_matches_current_verified_mysql_estimate(tamper, expected):
    from aiops_agent.executor.drivers.statement_plan import safe_plan_row
    from aiops_agent.domain.evidence.sql_plan_matching import assess_sql_plans
    entry = anchor("MYSQL")
    physical = {"server_uuid": PHYSICAL["MYSQL"]["server_uuid"]}
    entry = entry.model_copy(update={"provenance": entry.provenance.model_copy(update={"physical_identity": tuple(physical.items())})})
    sql = "SELECT * FROM mes.orders"
    rows = (safe_plan_row({"query_block": {"table": {"table_name": "orders", "access_type": "ALL"}}},
        database_type="MYSQL", subject=SUBJECTS["MYSQL"], source_hash=canonical_hash(sql if tamper != "text" else "SELECT 1")),)
    tool = next(item for item in DiagnosticRegistry.load().tools if item.definition.tool_id == "db.mysql.statement.plan")
    if tamper == "server":
        physical = {"server_uuid": "cb5538b8-749c-4de4-8fa2-fae0b2049eb1"}
    elif tamper == "missing":
        physical = {}
    obs = observation().model_copy(update={"db_type": "MYSQL", "tool_id": tool.definition.tool_id,
        "variant": tool.definition.variant, "columns": tuple(DatabaseColumn(name=name, logical_type="STRING", sensitivity="PUBLIC") for name in PLAN_COLUMNS),
        "rows": rows, "row_count": 1, "provenance": {"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": physical}})
    value = database_evidence_fact(result=SimpleNamespace(source_id=tool.definition.tool_id,
        measurement_semantics="NOT_APPLICABLE", presentation_kind="TABLE", sampling=None, observation_series=None),
        outcome=SimpleNamespace(observation=obs, step_id="a1", tool_id=tool.definition.tool_id), artifact_id=uuid7())
    if tamper == "untrusted":
        value = value.model_copy(update={"trust_level": "USER_PROVIDED"})
    elif tamper == "historical":
        entry = entry.model_copy(update={"captured_at": "2026-11-01T00:00:00+00:00"})
    assessed = assess_sql_plans(database_type="MYSQL", subject=SUBJECTS["MYSQL"], evidence=(entry, value))[0]
    assert assessed.physical_match == expected
    assert not assessed.production_execution_verified and assessed.runtime_statistics == "NONE"
    visible = matching_plan_refs({"plan_assessments": [assessed.model_dump(mode="json")]})
    assert bool(visible) is (tamper is None)


@pytest.mark.parametrize("update", [{"origin": "USER_UPLOAD"}, {"origin": "CONTROLLED_ESTIMATE"},
    {"database_type": "POSTGRESQL"}, {"runtime_statistics": "PRESENT"},
    {"identity_match": "INCOMPLETE"}, {"time_relation": "HISTORICAL_OR_DIFFERENT_TIME"}])
def test_server_scope_contract_cannot_upgrade_uploads_or_runtime_claims(update):
    from aiops_agent.contracts.sql_plan import SqlPlanAssessment
    from pydantic import ValidationError
    value = dict(evidence_ref="artifact:plan#a1", database_type="MYSQL", subject=SUBJECTS["MYSQL"], origin="DATABASE_ESTIMATE",
        identity_match="SOURCE_MATCH", physical_match="SOURCE_SERVER_MATCH", time_relation="SOURCE_NEAR_OBSERVATION",
        runtime_statistics="NONE", node_count=1, limitations=("启动代次未知",))
    with pytest.raises(ValidationError):
        SqlPlanAssessment.model_validate({**value, **update})


@pytest.mark.parametrize("updates,code", [
    ({"sample_complete": 0}, "SQL_PLAN_SAMPLE_INVALID"),
    ({"sample_digest": "B" * 64}, "SQL_PLAN_SAMPLE_INVALID"),
    ({"current_database": "other"}, "SQL_PLAN_DATABASE_MISMATCH"),
    ({"sample_sql": "SELECT * FROM mes.orders WHERE id = ?"}, "SQL_PLAN_PARAMETER_TYPES_UNAVAILABLE"),
    ({"sample_sql": "SELECT SLEEP(1) FROM mes.orders"}, "SQL_PLAN_FUNCTION_UNSUPPORTED"),
    ({"sample_sql": "SELECT evil.custom_func()"}, "SQL_PLAN_FUNCTION_UNSUPPORTED"),
    ({"sample_sql": "UPDATE mes.orders SET id=1"}, "SQL_PLAN_STATEMENT_UNSUPPORTED"),
    ({"sample_sql": "SELECT * FROM mes.orders FOR UPDATE"}, "SQL_PLAN_STATEMENT_UNSUPPORTED"),
    ({"sample_sql": "SELECT @v := 1"}, "SQL_PLAN_PARAMETER_TYPES_UNAVAILABLE"),
    ({"sample_sql": "SELECT /*+ SET_VAR(sort_buffer_size=10000) */ * FROM mes.orders"}, "SQL_PLAN_STATEMENT_UNSUPPORTED"),
])
def test_mysql_unverified_or_unsafe_sample_stops_before_explain(updates, code):
    async def exercise():
        source, cursor = sample_cursor(**updates)
        with pytest.raises(DiagnosticDriverError) as error:
            await mysql_statement_plan(cursor, bound_source_sql="SOURCE", ensure_active=lambda: None,
                parameters={"source_sql_sha256": canonical_hash(source["source_sql"])})
        assert error.value.code == code
        assert cursor.execute.await_count == 1
    asyncio.run(exercise())


def test_mysql_changed_sample_cannot_become_a_verified_plan():
    async def exercise():
        source, cursor = sample_cursor()
        changed = {**source, "sample_sql": "SELECT * FROM mes.orders WHERE id=999"}
        cursor.fetchmany.side_effect = [[tuple(source.values())], [tuple(changed.values())]]
        with pytest.raises(DiagnosticDriverError, match="SQL_PLAN_SOURCE_CHANGED"):
            await mysql_statement_plan(cursor, bound_source_sql="SOURCE", ensure_active=lambda: None,
                parameters={"source_sql_sha256": canonical_hash(source["source_sql"])})
    asyncio.run(exercise())


@pytest.mark.parametrize("engine,boot_known", [("ORACLE", True), ("POSTGRESQL", True), ("MYSQL", True), ("MYSQL", False)])
def test_fixed_tool_projection_matching_and_final_answer_keep_the_same_graph(engine, boot_known):
    async def exercise():
        entry = direct() if engine == "ORACLE" else anchor(engine)
        if engine == "ORACLE":
            value = fact(observation())
        else:
            tool_id = "db.postgresql.statement.plan" if engine == "POSTGRESQL" else "db.mysql.statement.plan"
            tool = next(item for item in DiagnosticRegistry.load().tools if item.definition.tool_id == tool_id)
            if engine == "POSTGRESQL":
                sql = "SELECT * FROM mes.orders WHERE id=$1"
                source = {**SUBJECTS[engine], "source_sql": sql}
                connection = SimpleNamespace(fetch=AsyncMock(return_value=[source]), execute=AsyncMock(),
                    get_server_version=lambda: SimpleNamespace(major=16), fetchval=AsyncMock(side_effect=[16510, 1,
                        [{"Plan": {"Node Type": "Index Scan", "Schema": "mes", "Relation Name": "orders",
                            "Index Name": "orders_idx", "Plan Rows": 1}}]]))
                rows = await postgresql_statement_plan(connection, source_sql=tool.sql,
                    parameters={**SUBJECTS[engine], "source_sql_sha256": canonical_hash(sql)})
                statement_field = "normalized_statement"
            else:
                source, cursor = sample_cursor(digest=SUBJECTS[engine]["digest"], sample_digest=SUBJECTS[engine]["digest"])
                sql = source["source_sql"]
                rows = await mysql_statement_plan(cursor, bound_source_sql=tool.sql,
                    parameters={"source_sql_sha256": canonical_hash(sql)}, ensure_active=lambda: None)
                statement_field = "digest_text"
            names = [column["name"] for column in entry.columns]
            values = list(entry.rows[0])
            values[names.index(statement_field)] = sql
            entry = entry.model_copy(update={"rows": (tuple(values),)})
            physical = dict(PHYSICAL[engine])
            if not boot_known:
                physical.pop("instance_started_at")
                entry = entry.model_copy(update={"provenance": entry.provenance.model_copy(update={"physical_identity": tuple(physical.items())})})
            obs = observation().model_copy(update={"db_type": engine, "db_version": "16" if engine == "POSTGRESQL" else "8.4",
                "tool_id": tool_id, "variant": tool.definition.variant, "template_sha256": tool.definition.template_sha256,
                "columns": tuple(DatabaseColumn(name=name, logical_type="STRING", sensitivity="PUBLIC") for name in PLAN_COLUMNS),
                "rows": rows, "row_count": 1, "result_sha256": canonical_hash(rows),
                "provenance": {"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": physical}})
            outcome = SimpleNamespace(observation=obs, step_id="a1", tool_id=tool_id)
            result = SimpleNamespace(source_id=tool_id, measurement_semantics="NOT_APPLICABLE", presentation_kind="TABLE", sampling=None, observation_series=None)
            value = database_evidence_fact(result=result, outcome=outcome, artifact_id=uuid7())
        task = frame(engine, {"subject_type": {"ORACLE": "ORACLE_SQL_ID", "POSTGRESQL": "POSTGRESQL_QUERY_ID", "MYSQL": "MYSQL_DIGEST"}[engine], **SUBJECTS[engine]})
        bundle = build_sql_evidence_bundle(task_frame=task, database_type=engine, evidence=(entry, value))
        assert value.evidence_ref in matching_plan_refs(bundle)
        assert bundle["plan_assessments"][0]["physical_match"] == ("SOURCE_MATCH" if boot_known else "SOURCE_SERVER_MATCH")
        assert not bundle["plan_assessments"][0]["production_execution_verified"]
        context = _context(task_frame_overrides=task)
        context.plan_snapshot["capability_snapshot"] = {"database_type": engine}
        assessment = DbaSufficiencyAssessment(status="PARTIAL", evidence=(entry, value))
        context = replace(context, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},))
        answer = await DbaAnswerComposeHandler(model_client=None, prompts=None).execute(context)
        shown = [block for block in answer.blocks if block.payload.get("result_kind") == "SQL_EXECUTION_PLAN"]
        assert len(shown) == 1
        assert shown[0].payload["plan_graph"]["nodes"] == [node.model_dump(mode="json") for node in value.sql_plan.nodes]
        assert shown[0].payload["plan_graph"]["database_type"] == engine
        assert shown[0].payload["plan_origin"] == ("DATABASE_CURSOR" if engine == "ORACLE" else "DATABASE_ESTIMATE")
        assert shown[0].payload["presentation_role"] == "PRIMARY"
        assert answer.blocks[0].block_type == "MARKDOWN"
        assert shown[0].evidence_refs == (value.evidence_ref,)
    asyncio.run(exercise())


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
@pytest.mark.parametrize("state", ["NOT_ATTEMPTED", "FAILED", "NO_MATCHING_PLAN"])
def test_missing_plan_has_a_visible_card_with_truthful_collection_state(engine, state):
    async def exercise():
        entry = anchor(engine)
        tool = {"ORACLE": "db.oracle.statement.plan", "POSTGRESQL": "db.postgresql.statement.plan", "MYSQL": "db.mysql.statement.plan"}[engine]
        gaps = (TurnEvidenceGap(source_id=tool, step_id="a2", code="PRIVILEGE_MISSING", detail="目标拒绝计划读取权限", evidence_ref="artifact:failed#a2"),) if state == "FAILED" else ()
        evidence = (entry, entry.model_copy(update={"tool_id": tool, "rows": (), "row_count": 0})) if state == "NO_MATCHING_PLAN" else (entry,)
        assessment = DbaSufficiencyAssessment(status="PARTIAL", evidence=evidence, gaps=gaps)
        context = _context(task_frame_overrides=frame(engine, SUBJECTS[engine]))
        context.plan_snapshot["capability_snapshot"] = {"database_type": engine}
        context = replace(context, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},))
        answer = await DbaAnswerComposeHandler(model_client=None, prompts=None).execute(context)
        cards = [b for b in answer.blocks if b.payload.get("result_kind") == "SQL_EXECUTION_PLAN"]
        assert len(cards) == 1 and cards[0].payload["presentation_role"] == "PRIMARY"
        assert cards[0].payload["sql_plan_acquisition"]["collection_state"] == state
        assert "plan_graph" not in cards[0].payload and cards[0].payload["row_count"] == 0
        assert cards[0].payload["sql_plan_acquisition"]["display_reason"]
        assert answer.blocks[0].block_type == "MARKDOWN"
    asyncio.run(exercise())
