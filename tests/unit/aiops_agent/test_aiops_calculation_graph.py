"""公式图在真实效果报告链中的精确计算、质量传播和拒绝路径。"""

from datetime import timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from aiops_agent.contracts.calculation import CalculationGraph, CalculationStep
from aiops_agent.domain.calculation import evaluate_graph
from aiops_agent.application.effect_measurement import calculated_measurement
from tests.unit.aiops_agent.test_aiops_measurements import NOW, scalar
from tests.unit.aiops_agent.test_aiops_business_effect import report, signals
from platform_core.contracts.aiops.effect import EffectSignalRule


def graph(*, roots=None, steps=None, outputs=("mean",)):
    return CalculationGraph(
        roots=roots or {
            "count_before": scalar(value="900719925474099312345", unit="execution"),
            "count_after": scalar(value="900719925474099312348", unit="execution", observed_at=NOW + timedelta(seconds=10)),
            "total_before": scalar(value="123456789012345678900000", unit="microsecond", native_origin_id="duration"),
            "total_after": scalar(value="123456789012345678900900", unit="microsecond", native_origin_id="duration", observed_at=NOW + timedelta(seconds=10)),
        },
        steps=steps or (
            CalculationStep(node_id="mean", operator="DIVIDE", inputs=("duration", "calls")),
            CalculationStep(node_id="calls", operator="COUNTER_DELTA", inputs=("count_before", "count_after")),
            CalculationStep(node_id="duration", operator="COUNTER_DELTA", inputs=("total_before", "total_after")),
            CalculationStep(node_id="rate", operator="RATE", inputs=("calls",)),
        ),
        outputs=outputs,
    )


def test_dependency_order_is_topological_and_big_integer_deltas_remain_exact():
    result = evaluate_graph(graph())
    assert result.values["calls"].measurement.value == 3
    assert result.values["rate"].measurement.value == Decimal("0.3")
    assert result.values["mean"].measurement.value == 300
    assert result.values["mean"].measurement.unit == "microsecond/execution"
    assert len(result.values["mean"].native_origin_ids) == 2
    assert result.decimal_precision >= 80
    assert evaluate_graph(result.graph).model_dump(mode="json") == result.model_dump(mode="json")


@pytest.mark.parametrize("change", [
    {"kind": "MAXIMUM"}, {"kind": "PERCENTILE"}, {"reset_marker": None},
    {"reset_marker": "new"}, {"source_version": "new"},
    {"value": None, "quality": "MISSING"}, {"quality": "PARTIAL"},
    {"trust_level": "USER_PROVIDED"}, {"trust_level": "MODEL_INFERENCE"},
    {"series_identity": (("query_id", "other"),)}, {"value": "0"},
    {"observed_at": NOW - timedelta(seconds=1)},
])
def test_bad_root_blocks_transitive_results_without_erasing_raw_evidence(change):
    data = graph().model_dump(mode="python")
    data["roots"]["count_after"] = {**data["roots"]["count_after"], **change}
    result = evaluate_graph(data)
    assert result.values["mean"].quality != "GOOD"
    assert result.values["mean"].measurement.value is None
    expected = change.get("value", "900719925474099312348")
    assert result.graph.roots["count_after"].value == (Decimal(expected) if expected is not None else None)


@pytest.mark.parametrize("steps,outputs", [
    ([{"node_id": "a", "operator": "RATE", "inputs": ["b"]}, {"node_id": "b", "operator": "RATE", "inputs": ["a"]}], ("a",)),
    ([{"node_id": "a", "operator": "RATE", "inputs": ["unknown"]}], ("a",)),
    ([{"node_id": "count_before", "operator": "RATE", "inputs": ["count_after"]}], ("count_before",)),
    ([{"node_id": "a", "operator": "DIVIDE", "inputs": ["count_before", "count_before"]}], ("a",)),
    ([{"node_id": "a", "operator": "RATE", "inputs": ["count_before"]}], ("count_before",)),
])
def test_cycles_missing_inputs_overrides_repeats_and_noncomputed_outputs_are_rejected(steps, outputs):
    with pytest.raises(ValidationError):
        graph(steps=tuple(CalculationStep(**item) for item in steps), outputs=outputs)


def test_depth_and_node_budget_cannot_be_bypassed():
    steps = tuple(CalculationStep(node_id=f"n{i}", operator="RATE", inputs=(f"n{i-1}" if i else "count_before",)) for i in range(33))
    with pytest.raises(ValidationError, match="深度"):
        graph(steps=steps, outputs=("n32",))


def test_unknown_units_and_mutated_graphs_cannot_bypass_input_validation():
    with pytest.raises(ValidationError, match="单位未核验"):
        scalar(unit="UNKNOWN")
    original = graph()
    mutated = original.model_copy(update={"outputs": ("absent",)})
    with pytest.raises(ValidationError, match="输出"):
        evaluate_graph(mutated)


@pytest.mark.parametrize("trust", ["USER_PROVIDED", "MODEL_INFERENCE"])
def test_result_preserves_weakest_input_trust_regardless_of_input_order(trust):
    result = evaluate_graph(graph(
        roots={
            "numerator": scalar(kind="GAUGE", value=10),
            "denominator": scalar(kind="GAUGE", value=2, trust_level=trust),
        },
        steps=(CalculationStep(node_id="mean", operator="DIVIDE", inputs=("numerator", "denominator")),),
    ))
    assert result.values["mean"].measurement.trust_level == trust
    assert result.values["mean"].measurement.value is None


@pytest.mark.parametrize("oversized", ["number", "unit"])
def test_oversized_outputs_fail_as_evidence_gaps_not_unhandled_validation_errors(oversized):
    roots = {
        "numerator": scalar(kind="GAUGE", value="1e1024" if oversized == "number" else 10, unit="x" * 128 if oversized == "unit" else "second"),
        "denominator": scalar(kind="GAUGE", value="1e-1024" if oversized == "number" else 2),
    }
    result = evaluate_graph(graph(
        roots=roots,
        steps=(CalculationStep(node_id="mean", operator="DIVIDE", inputs=("numerator", "denominator")),),
    ))
    output = result.values["mean"]
    assert output.measurement.value is None
    assert output.quality == "INCOMPARABLE"
    assert ("OUTPUT_NUMERIC_BUDGET_EXCEEDED" if oversized == "number" else "OUTPUT_UNIT_BUDGET_EXCEEDED") in output.reasons


def test_summation_cannot_double_count_shared_lineage():
    data = graph().model_dump(mode="python")
    data["steps"] = [*data["steps"],
        {"node_id": "copy", "operator": "COUNTER_DELTA", "inputs": ["count_before", "count_after"]},
        {"node_id": "sum", "operator": "SUM", "inputs": ["calls", "copy"]},
    ]
    data["outputs"] = ("sum",)
    result = evaluate_graph(data)
    assert result.values["sum"].measurement.value is None
    assert "DUPLICATED_LINEAGE" in result.values["sum"].reasons


def test_rate_requires_a_real_window_delta_and_never_differences_maxima():
    result = evaluate_graph(graph(
        steps=(CalculationStep(node_id="rate", operator="RATE", inputs=("count_before",)),),
        outputs=("rate",),
    ))
    assert result.values["rate"].measurement.value is None
    assert "NOT_A_WINDOW_DELTA" in result.values["rate"].reasons


def test_window_and_source_mismatches_block_division():
    data = graph().model_dump(mode="python")
    data["roots"]["total_before"]["observed_at"] = NOW - timedelta(seconds=1)
    result = evaluate_graph(data)
    assert result.values["mean"].measurement.value is None
    assert "WINDOW_MISMATCH" in result.values["mean"].reasons


def test_real_effect_computation_keeps_original_fields_and_can_be_replayed():
    payload = report(NOW)
    rule = EffectSignalRule(**signals()[0])
    value, unit, formula, frozen = calculated_measurement(payload, rule, source_ref="artifact:source")
    result = evaluate_graph(frozen["graph"])
    assert value == 100 and unit == "us/call"
    assert result.values["result"].measurement.value == value
    assert result.values["result"].leaf_refs == (
        "artifact:source#/top_statements/0/execution_count",
        "artifact:source#/top_statements/0/total_duration_microseconds",
    )
    payload["top_statements"][0]["mean_duration_microseconds"] = 1
    assert calculated_measurement(payload, rule, source_ref="artifact:source")[0] == 100


def test_load_is_recomputed_from_window_counts_not_displayed_rate():
    payload = report(NOW)
    rule = EffectSignalRule(**signals()[2])
    payload["load_profile"]["transactions_per_second"] = 12345
    value, _, _, frozen = calculated_measurement(payload, rule, source_ref="artifact:load")
    assert value == Decimal(1000) / Decimal(900) or str(value).startswith("1.111111111111111111111")
    assert frozen["values"]["result"]["leaf_refs"] == ["artifact:load#/load_totals/transactions"]
    del payload["load_totals"]
    with pytest.raises(KeyError):
        calculated_measurement(payload, rule, source_ref="artifact:load")


def test_top_sql_worker_freezes_calculation_graph_and_report_projection():
    import asyncio
    from types import SimpleNamespace
    from aiops_agent.application.workload import build_top_sql_window
    from aiops_agent.application.diagnosis.investigation_record import freeze_investigation, investigation_report_source
    from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
    from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
    from platform_core.identity import uuid7
    from tests.unit.aiops_agent.test_aiops_window_evidence_pipeline import evidence

    snapshots, statements = evidence("POSTGRESQL", NOW)
    for statement in statements:
        statement.database_name = "16510"
    for snapshot in snapshots:
        snapshot.instance_identity_json["database_oid"] = "16510"
        snapshot.continuity_json["sql_collection"] = {"mode": "VISIBLE_SET", "visible_set_read_complete": True}
    result = build_top_sql_window(
        snapshots=snapshots, statements=statements, period_start=snapshots[0].collected_at,
        period_end=snapshots[1].collected_at, postgresql_database_oid="16510", limit=10,
    )
    result["collection_enabled"] = True
    artifact_id = str(uuid7())
    context = SimpleNamespace(
        input_artifacts=({"schema_version": "TOP_SQL_WINDOW_RESULT.v1", "artifact_id": artifact_id, "payload": {"result": result}},),
        plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="MANUAL",
        target_id=uuid7(), run_id=uuid7(), task_id=uuid7(),
    )
    assessment = asyncio.run(DbaEvidenceAssessmentHandler().execute(context))
    fact = assessment.evidence[0]
    computed = fact.calculation_graphs[0]
    assert computed.values["mean"].measurement.value == 100
    assert computed.values["calls_per_second"].quality == "GOOD"
    assert computed.values["mean"].leaf_refs == (
        f"artifact:{artifact_id}#/result/top_statements/0/execution_count",
        f"artifact:{artifact_id}#/result/top_statements/0/total_duration_microseconds",
    )
    draft = DiagnosisAnswerDraft(
        claims=({"claim_id": "c1", "kind": "FACT", "subject": "目标 SQL", "statement": "返回受控窗口执行次数",
                 "level": "OBSERVED", "applicable_time": result["requested_end"],
                 "evidence_fields": ({"evidence_ref": fact.evidence_ref, "row_index": 0, "field": "execution_count"},),
                 "reasoning": "直接读取窗口结果"},), goal_satisfied=True,
    )
    frozen = freeze_investigation(draft=draft, evidence=assessment.evidence, context=context)
    assert frozen.observations[0].calculation_graphs == fact.calculation_graphs
    projected = investigation_report_source(frozen)["investigation_record"]["observations"][0]
    assert projected["calculation_graphs"][0] == computed.model_dump(mode="json")
