"""MySQL 原生连接数组仅投影一个连接节点，保留输入顺序、估算与隐私边界。"""

import json

import pytest

from aiops_agent.domain.evidence.sql_plan import SqlPlanImportError, parse_sql_plan_json


def plan(count):
    names = ("c", "a", "b")
    return {"query_block": {"cost_info": {"query_cost": "12.50"}, "ordering_operation": {
        "nested_loop": [{"table": {
            "table_name": names[index % 3], "access_type": "range" if index == 0 else "eq_ref",
            "rows_examined_per_scan": index + 1, "cost_info": {"prefix_cost": str(index + 2)},
            "attached_condition": "private-literal", "ref": ["private-literal"],
        }} for index in range(count)]}}}


@pytest.mark.parametrize("count", [1, 2, 3])
def test_nested_loop_is_one_node_with_ordered_native_inputs(count):
    raw = plan(count)
    before = json.dumps(raw)
    document = parse_sql_plan_json(before)
    assert [node.operation for node in document.nodes] == [
        "query_block", "ordering_operation", "nested_loop", "range",
        *(["eq_ref"] * (count - 1)),
    ]
    assert [node.parent_id for node in document.nodes] == [None, 0, 1, *([2] * count)]
    inputs = document.nodes[3:]
    assert [node.relation_name for node in inputs] == list(("c", "a", "b")[:count])
    assert [node.metrics["estimated_rows_examined_per_scan"] for node in inputs] == [
        str(index + 1) for index in range(count)]
    assert [node.metrics["optimizer_prefix_cost"] for node in inputs] == [
        str(index + 2) for index in range(count)]
    assert document.nodes[0].metrics["optimizer_query_cost"] == "12.50"
    assert document.nodes[2].metrics == {}
    assert document.runtime_statistics == "NONE"
    assert "private-literal" not in document.model_dump_json()
    assert json.dumps(raw) == before
    # 原生 JSON 重放不依赖来源身份，不修改历史材料即可得到一致拓扑。
    assert parse_sql_plan_json(before) == document


def test_nested_loop_materialized_subquery_remains_attached_to_its_input():
    raw = plan(2)
    raw["query_block"]["ordering_operation"]["nested_loop"][1]["table"]["materialized_from_subquery"] = {
        "query_block": {"table": {"table_name": "inner", "access_type": "ALL"}}}
    document = parse_sql_plan_json(json.dumps(raw))
    assert [node.parent_id for node in document.nodes] == [None, 0, 1, 2, 2, 4, 5, 6]
    assert [node.operation for node in document.nodes[-3:]] == [
        "materialized_from_subquery", "query_block", "ALL"]


@pytest.mark.parametrize("inputs", [[], [None], [1], [[]], [{}], [{"attached_condition": "private-literal"}]])
def test_invalid_nested_loop_members_fail_closed(inputs):
    raw = plan(1)
    raw["query_block"]["ordering_operation"]["nested_loop"] = inputs
    with pytest.raises(SqlPlanImportError, match="^SQL_PLAN_IMPORT_INVALID$"):
        parse_sql_plan_json(json.dumps(raw))


@pytest.mark.parametrize("count,valid", [(125, True), (126, False)])
def test_nested_loop_still_enforces_node_budget(count, valid):
    # 根节点、排序与连接各占一个节点，所有输入共同计入 128 节点上限。
    raw = json.dumps(plan(count))
    if valid:
        assert len(parse_sql_plan_json(raw).nodes) == 128
    else:
        with pytest.raises(SqlPlanImportError, match="^SQL_PLAN_IMPORT_INVALID$"):
            parse_sql_plan_json(raw)


def test_nested_loop_still_enforces_depth_budget():
    native = plan(1)
    for _ in range(34):
        native = {"query_block": {"nested_loop": [{"table": {
            "table_name": "t", "materialized_from_subquery": native}}]}}
    with pytest.raises(SqlPlanImportError, match="^SQL_PLAN_IMPORT_INVALID$"):
        parse_sql_plan_json(json.dumps(native))
