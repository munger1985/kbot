"""将受控窗口的原始增量编译成可重放公式图，而非读取展示用平均值。"""

from aiops_agent.application.configuration.common import canonical_json
from aiops_agent.contracts.calculation import CalculationGraph, CalculationStep
from aiops_agent.contracts.measurements import ScalarMeasurement
from aiops_agent.domain.calculation import evaluate_graph


def window_calculation(payload, rule, *, source_ref, start, end):
    scope = {
        "target_id": str(payload["target_id"]),
        "instance_identity": canonical_json(payload["instance_identity"]),
        "instance_started_at": str(payload["instance_started_at"]),
        **{key: str(value) for key, value in sorted(rule.statement_identity.items())},
    }
    version = canonical_json({key: payload[key] for key in (
        "database_versions", "catalog_versions", "capability_probe_versions",
    )})
    roots = {}
    def root(key, value, unit, path):
        roots[key] = ScalarMeasurement(
            input_ref=f"{source_ref}#/{path}", value=value, unit=unit,
            series_identity=tuple(sorted(scope.items())),
            native_origin_id=f"window:{scope['target_id']}:{path}",
            source_version=version, observed_at=end, kind="WINDOW_DELTA",
            window_start=start, window_end=end, quality="GOOD",
        )
    if rule.statement_identity:
        index, row = next((index, row) for index, row in enumerate(payload["top_statements"]) if all(
            row.get(key) == value for key, value in rule.statement_identity.items()
        ))
        prefix = f"top_statements/{index}"
        root("count", row["execution_count"], "execution", f"{prefix}/execution_count")
        if rule.metric == "statement.calls_per_second":
            steps = (CalculationStep(node_id="result", operator="RATE", inputs=("count",)),)
        else:
            key, unit = ("total_duration_microseconds", "microsecond") if rule.metric == "statement.mean_duration_microseconds" else ("rows_processed", "rows")
            root("total", row[key], unit, f"{prefix}/{key}")
            steps = (CalculationStep(node_id="result", operator="DIVIDE", inputs=("total", "count")),)
    elif rule.metric.startswith("load."):
        key = rule.metric.removeprefix("load.")
        counter = key.removesuffix("_per_second")
        unit = payload["load_dimension_semantics"][key]["unit"].removesuffix("/s")
        # 原始窗口增量已在报告中保存；禁止用展示用速率反推计数。
        root("total", payload["load_totals"][counter], unit, f"load_totals/{counter}")
        steps = (CalculationStep(node_id="result", operator="RATE", inputs=("total",)),)
    else:
        key = rule.metric.removeprefix("metric.")
        native = payload["native_window_metrics"][key]
        # JSON Pointer 保留原生代码中的斜杠和波浪线，不产生歧义路径。
        escaped = key.replace("~", "~0").replace("/", "~1")
        root("total", native["window_delta"], native["unit"], f"native_window_metrics/{escaped}/window_delta")
        steps = (CalculationStep(node_id="result", operator="RATE", inputs=("total",)),)
    return evaluate_graph(CalculationGraph(roots=roots, steps=steps, outputs=("result",)))
