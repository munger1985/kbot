"""三库计划结构投影保留原生算子、身份和精确指标，不从字符缩进反推图形。"""

from types import SimpleNamespace

import pytest

from aiops_agent.application.diagnosis.sql_plan_presentation import plan_table_block
from aiops_agent.contracts.sql_plan import SqlPlanDocument, SqlPlanNode


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
def test_graph_and_standard_rows_share_the_same_bounded_plan(engine):
    native = {"ORACLE": "ORACLE_PLAN_ROWS", "POSTGRESQL": "POSTGRESQL_EXPLAIN_JSON", "MYSQL": "MYSQL_EXPLAIN_JSON"}[engine]
    subject = {"ORACLE": {"sql_id": "36apy6dncjggq"}, "POSTGRESQL": {"query_id": "1285382359565720997"},
        "MYSQL": {"digest": "a" * 64}}[engine]
    document = SqlPlanDocument(database_type=engine, native_format=native, subject=subject,
        nodes=(SqlPlanNode(node_id=0, operation="SELECT STATEMENT"),
            SqlPlanNode(node_id=1, parent_id=0, operation="HASH JOIN"),
            SqlPlanNode(node_id=2, parent_id=1, operation="FIXED TABLE FULL", schema_name="SYS", relation_name="X$KSPPI",
                metrics={"estimated_rows": "0", "actual_elapsed_time_us": "12345678901234567890.123456"}),
            SqlPlanNode(node_id=3, parent_id=1, operation="Index Scan", index_name="equipment_idx")),
        warnings=("仅用于结构投影验证，未执行 SQL。",))
    fact = SimpleNamespace(sql_plan=document, evidence_ref="artifact:plan", captured_at="2026-10-29T06:00:00+00:00",
        measurement_semantics="NOT_APPLICABLE", truncated=False)
    block = plan_table_block(fact)
    graph = block.payload["plan_graph"]
    assert graph["schema_version"] == "SQL_PLAN_GRAPH.v1"
    assert graph["database_type"] == engine
    assert graph["nodes"] == [node.model_dump(mode="json") for node in document.nodes]
    assert graph["nodes"][2]["operation"] == "FIXED TABLE FULL"
    assert graph["nodes"][2]["parent_id"] == 1
    assert graph["nodes"][2]["metrics"]["estimated_rows"] == "0"
    assert graph["nodes"][2]["metrics"]["actual_elapsed_time_us"] == "12345678901234567890.123456"
    assert all("└─" not in node["operation"] for node in graph["nodes"])
    assert len(block.payload["rows"]) == len(graph["nodes"]) == block.payload["row_count"]
    assert block.payload["subject"] == subject
    assert block.evidence_refs == ("artifact:plan",)
    assert block.payload["plan_origin"] == "USER_UPLOAD"
    assert block.payload["runtime_statistics"] == "NONE"
    assert block.payload["has_row_actions"] is False
    assert "未执行 SQL" in block.payload["measurement_note"]
    assert not any(key in graph for key in ("sql_text", "parameters", "predicates", "hotspots", "elapsed_percent"))


@pytest.mark.parametrize("engine,metrics,key,label", [
    ("ORACLE", {"actual_rows_total": "9", "actual_starts": "3", "actual_elapsed_time_us": "1000"}, "actual_elapsed_time_us", "LAST 耗时（微秒）"),
    ("POSTGRESQL", {"actual_rows_per_loop": "9", "actual_loops": "3", "actual_total_time_per_loop_ms": "1.25"}, "actual_total_time_per_loop_ms", "实际耗时／循环（毫秒）"),
    ("MYSQL", {"estimated_rows_examined_per_scan": "9", "optimizer_read_cost": "2.3"}, "estimated_rows", "估算行数"),
])
def test_native_units_and_zero_remain_visible_in_the_original_plan(engine, metrics, key, label):
    native = {"ORACLE": "ORACLE_PLAN_ROWS", "POSTGRESQL": "POSTGRESQL_EXPLAIN_JSON", "MYSQL": "MYSQL_EXPLAIN_JSON"}[engine]
    document = SqlPlanDocument(database_type=engine, native_format=native,
        nodes=(SqlPlanNode(node_id=0, operation="SELECT", metrics=metrics),))
    fact = SimpleNamespace(sql_plan=document, evidence_ref="artifact:unit", captured_at="2026-10-29T06:00:00+00:00",
        measurement_semantics="NOT_APPLICABLE", truncated=False)
    columns = plan_table_block(fact).payload["columns"]
    assert {"name": key, "label": label} in columns
