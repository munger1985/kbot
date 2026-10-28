"""窗口 SQL 图只读取已冻结的推导输入，并绑定真实结果 Artifact 的字段路径。"""

from datetime import datetime

from aiops_agent.contracts.calculation import CalculationGraph, CalculationStep
from aiops_agent.contracts.measurements import DerivedMeasurement
from aiops_agent.domain.calculation import evaluate_graph


def statement_window_graphs(result, artifact_id):
    graphs = []
    start, end = result.get("observed_start"), result.get("observed_end")
    if not start or not end:
        return ()
    start, end = (datetime.fromisoformat(str(value).replace("Z", "+00:00")) for value in (start, end))
    for index, statement in enumerate(result.get("top_statements") or ()):
        derivations = statement.get("derivations") or ()
        if len(derivations) != 1:
            continue
        derived = DerivedMeasurement.model_validate(derivations[0])
        if derived.formula != "TOTAL_DIVIDED_BY_COUNT":
            continue
        roots = {}
        for name, original, field in zip(
            ("total", "count"), derived.inputs,
            ("total_duration_microseconds", "execution_count"), strict=True,
        ):
            # 不猜测原生计数器映射；保留原来源，字段级引用只说明公式依赖。
            roots[name] = original.model_copy(update={
                "input_ref": f"artifact:{artifact_id}#/result/top_statements/{index}/{field}",
                "kind": "WINDOW_DELTA", "window_start": start, "window_end": end,
                "observed_at": max(original.observed_at, end),
                "quality": "GOOD" if derived.quality == "GOOD" else "PARTIAL",
            })
        graphs.append(evaluate_graph(CalculationGraph(
            roots=roots, steps=(
                CalculationStep(node_id="mean", operator="DIVIDE", inputs=("total", "count")),
                CalculationStep(node_id="calls_per_second", operator="RATE", inputs=("count",)),
            ), outputs=("mean", "calls_per_second"),
        )))
    return tuple(graphs)
