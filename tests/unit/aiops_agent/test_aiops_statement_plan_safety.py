"""计划工具区分语法与函数，并在流式生成前保留真实采集状态。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.sql_plan_presentation import plan_availability_block
from aiops_agent.contracts.turn_answer import TurnEvidenceGap
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.executor.drivers.base import DiagnosticDriverError
from aiops_agent.executor.drivers.statement_plan import postgresql_statement_plan, readonly_plan_sql, safe_plan_row
from aiops_agent.domain.evidence.sql_plan import parse_sql_plan_json, SqlPlanImportError
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.database_handlers import _database_gap_detail
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from tests.unit.aiops_agent.test_aiops_sql_review import setup


@pytest.mark.parametrize("sql", [
    "SELECT metric_value FROM mes.signal WHERE equipment_id=$1 AND metric_id=$2 AND observed_at >= $3 AND observed_at < $4 ORDER BY observed_at DESC LIMIT $5",
    "SELECT id FROM mes.signal WHERE (id=$1 OR id=$2) AND NOT (metric_id=$3)",
    "SELECT equipment_id, SUM(metric_value) FROM mes.signal GROUP BY equipment_id HAVING SUM(metric_value)>$1 AND COUNT(*)>$2",
    "SELECT a.id FROM mes.signal a JOIN mes.equipment b ON a.id=b.id AND b.metric_id=$1 WHERE a.id>$2 OR b.id=$3",
    "SELECT CASE WHEN id=$1 AND metric_id=$2 THEN metric_value WHEN id=$3 OR metric_id=$4 THEN 0 ELSE -1 END FROM mes.signal",
    "WITH chosen AS (SELECT id FROM mes.signal WHERE id=$1 AND metric_id=$2) SELECT id FROM chosen WHERE id>$3 OR id=$4",
    "SELECT COUNT(*) FROM mes.signal WHERE CAST(id AS bigint)>$1 AND COALESCE(metric_id,0)>$2",
])
def test_plan_accepts_structural_conditions_without_function_name_exceptions(sql):
    assert readonly_plan_sql(sql) == sql


@pytest.mark.parametrize("sql", [
    "SELECT id FROM mes.signal WHERE id=$1 AND pg_sleep(1) IS NULL",
    "SELECT id FROM mes.signal WHERE id=$1 OR evil.custom_func(id)>0",
    "SELECT CASE WHEN id=$1 AND metric_id=$2 THEN 1 ELSE pg_sleep(1) END FROM mes.signal",
    "SELECT CASE WHEN id=$1 THEN evil.custom_func(id) ELSE 0 END FROM mes.signal",
    'SELECT "and"(id,metric_id) FROM mes.signal',
    'SELECT "or"(id,metric_id) FROM mes.signal',
    'SELECT "if"(id,metric_id,$1) FROM mes.signal',
    'SELECT id FROM mes.signal WHERE id=$1 AND "and"(id,metric_id)',
    'SELECT CASE WHEN id=$1 THEN "if"(id,metric_id,$2) ELSE 0 END FROM mes.signal',
    "SELECT id FROM mes.signal WHERE id=$1 AND EXISTS(SELECT pg_advisory_lock(1))",
])
def test_condition_or_case_does_not_hide_unsupported_function(sql):
    with pytest.raises(DiagnosticDriverError) as error:
        readonly_plan_sql(sql)
    assert error.value.code == "SQL_PLAN_FUNCTION_UNSUPPORTED"


def test_parameterized_conditions_reach_generic_explain_without_executing_business_sql():
    async def exercise():
        sql="SELECT metric_value FROM mes.signal WHERE equipment_id=$1 AND metric_id=$2 ORDER BY observed_at DESC LIMIT $3"
        source=dict(query_id="-6589067333046971749",database_oid=42,user_oid=11,top_level=True,source_sql=sql)
        connection=SimpleNamespace(fetch=AsyncMock(return_value=[source]),execute=AsyncMock(),
            fetchval=AsyncMock(side_effect=[42,3,[{"Plan":{"Node Type":"Limit","Plan Rows":1},
                "Query Identifier":5378376035200847435}]]))
        params={key:source[key] for key in ("query_id","database_oid","user_oid","top_level")}
        params["source_sql_sha256"]=canonical_hash(sql)
        tool=next(item for item in DiagnosticRegistry.load().tools if item.definition.tool_id=="db.postgresql.statement.plan")
        result=await postgresql_statement_plan(connection,source_sql=tool.sql,parameters=params)
        assert len(result)==1
        assert json.loads(result[0][0]) == {"query_id":"-6589067333046971749",
            "database_oid":"42","user_oid":"11","top_level":True}
        assert result[0][1] == params["source_sql_sha256"]
        assert connection.fetchval.await_args.args[0]=="EXPLAIN (FORMAT JSON, VERBOSE TRUE) EXECUTE aiops_statement_plan(NULL,NULL,NULL)"
        commands=[call.args[0] for call in connection.execute.await_args_list]
        assert "SET LOCAL plan_cache_mode = force_generic_plan" in commands
        assert not any(command.startswith("EXECUTE ") or "ANALYZE" in command for command in commands)
    asyncio.run(exercise())


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("as_text", [False, True])
def test_fixed_estimate_binds_verified_source_but_uploaded_native_identity_remains_strict(as_list, as_text):
    subject={"query_id":"-6589067333046971749","database_oid":42,"user_oid":11,"top_level":True}
    native={"Plan":{"Node Type":"Index Scan","Plan Rows":7,"Index Name":"signal_idx"},
        "Query Identifier":5378376035200847435}
    native=[native] if as_list else native
    row=safe_plan_row(json.dumps(native) if as_text else native, database_type="POSTGRESQL",
        subject=subject,source_hash="a"*64)
    assert json.loads(row[0])["query_id"]==subject["query_id"]
    assert json.loads(row[2])[0]["operation"]=="Index Scan"
    with pytest.raises(SqlPlanImportError):
        parse_sql_plan_json(json.dumps({"schema_version":"SQL_PLAN_IMPORT.v1",
            "database_type":"POSTGRESQL","subject":subject,"plan":native}))


@pytest.mark.parametrize("code", [
    "SQL_PLAN_FUNCTION_UNSUPPORTED","SQL_PLAN_PARSE_CONTEXT_REQUIRED","SQL_PLAN_STATEMENT_UNSUPPORTED",
    "SQL_PLAN_SUBJECT_AMBIGUOUS","SQL_PLAN_SOURCE_CHANGED","SQL_PLAN_DATABASE_MISMATCH",
    "SQL_PLAN_PARAMETER_LIMIT","SQL_PLAN_PARAMETER_TYPES_UNAVAILABLE","SQL_PLAN_OUTPUT_INVALID","SQL_PLAN_EXECUTION_CHANGED",
])
def test_plan_failure_retains_specific_receipt_and_readable_reason(code):
    detail=_database_gap_detail(code)
    assert detail!="该数据库诊断证据本次不可用"
    gap=TurnEvidenceGap(source_id="db.postgresql.statement.plan",step_id="a1",code=code,detail=detail,
        evidence_ref="artifact:failed-plan#a1")
    block=plan_availability_block(database_type="POSTGRESQL",bundle={},gaps=(gap,))
    state=block.payload["sql_plan_acquisition"]
    assert state["collection_state"]=="FAILED"
    assert state["attempts"][0]["status"]=="FAILED"
    assert state["attempts"][0]["row_count"] is None
    assert state["attempts"][0]["error_code"]==code
    assert state["failures"][0]["detail"]==detail
    assert detail in block.payload["markdown"]


def test_sql_model_receives_plan_failure_before_answer_and_raw_answer_is_preserved():
    async def exercise():
        _,task,context,model,prompts,assessment=setup()
        detail=_database_gap_detail("SQL_PLAN_FUNCTION_UNSUPPORTED")
        gap=TurnEvidenceGap(source_id="db.postgresql.statement.plan",step_id="a1",
            code="SQL_PLAN_FUNCTION_UNSUPPORTED",detail=detail,evidence_ref="artifact:failed-plan#a1")
        assessment=assessment.model_copy(update={"gaps":(gap,)})
        assessed=await DbaEvidenceAssessmentHandler(model_client=model,prompts=prompts)._review_sql(context,assessment,task)
        inputs=model.generate_structured.await_args.kwargs["input_payload"]
        assert inputs["plan_acquisition"]["collection_state"]=="FAILED"
        assert inputs["plan_acquisition"]["failures"][0]["detail"]==detail
        assert "不能直接将内部采集失败转成请 DBA 提供" in inputs["delivery_requirements"]["plan_acquisition"]
        from dataclasses import replace
        answer_context=replace(context,input_artifacts=({"schema_version":"DBA_SUFFICIENCY.v1","payload":assessed.model_dump(mode="json")},))
        result=await DbaAnswerComposeHandler(model_client=model,prompts=prompts).execute(answer_context)
        block=next(item for item in result.blocks if item.block_type=="MARKDOWN")
        assert block.payload["sql_plan_acquisition"]["failures"][0]["code"]=="SQL_PLAN_FUNCTION_UNSUPPORTED"
        assert block.payload["markdown"]==assessed.sql_checkpoint.answer_markdown
        model.generate_structured.assert_awaited_once()
    asyncio.run(exercise())
