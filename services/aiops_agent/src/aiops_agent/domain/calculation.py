"""以高精度执行受控代数，不执行模型表达式或推断未知计数器语义。"""

from decimal import Decimal, ROUND_HALF_EVEN, localcontext

from aiops_agent.contracts.calculation import CalculationEvaluation, CalculationGraph, CalculationValue
from aiops_agent.contracts.measurements import ScalarMeasurement
from aiops_agent.domain.measurements import derive_measurement


def evaluate_graph(graph):
    graph = CalculationGraph.model_validate(graph.model_dump(mode="python") if isinstance(graph, CalculationGraph) else graph)
    roots = tuple(graph.roots.values())
    precision = max(80, max((
        len(item.value.as_tuple().digits) + abs(item.value.as_tuple().exponent)
        for item in roots if item.value is not None
    ), default=1) + 40)
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_HALF_EVEN
        return _evaluate(graph, precision)


def _evaluate(graph, precision):
    values = {}
    nodes = {item.node_id: item for item in graph.steps}
    def resolve(key):
        if key in values:
            return values[key]
        if key in graph.roots:
            item = graph.roots[key]
            reasons = (
                *(("MISSING_VALUE",) if item.value is None else ()),
                *(("INPUT_QUALITY_OR_TRUST",) if item.quality != "GOOD" or item.trust_level != "SOURCE_VERIFIED" else ()),
            )
            result = CalculationValue(
                measurement=item, quality="MISSING" if item.value is None else "INCOMPARABLE" if reasons else "GOOD",
                reasons=reasons, leaf_refs=(item.input_ref,), native_origin_ids=(item.native_origin_id,),
            )
        else:
            step = nodes[key]
            inputs = [resolve(ref) for ref in step.inputs]
            result = _step(step, inputs)
        values[key] = result
        return result
    # 检查未用于最终输出的节点，不能把非法计算藏在不可见分支。
    for step in graph.steps:
        resolve(step.node_id)
    return CalculationEvaluation(graph=graph, values=values, decimal_precision=precision)


def _step(step, inputs):
    items = [item.measurement for item in inputs]
    first = items[0]
    reasons = list(dict.fromkeys(reason for item in inputs for reason in item.reasons))
    value, unit, kind = None, first.unit, "GAUGE"
    start, end = first.window_start, first.window_end
    if step.operator == "COUNTER_DELTA":
        result = derive_measurement("AFTER_MINUS_BEFORE", *items)
        reasons.extend(result.reasons)
        value, unit, kind = result.value, result.unit, "WINDOW_DELTA"
        start, end = items[0].observed_at, items[1].observed_at
        if start >= end:
            start = end = None
    else:
        if any(item.series_identity != first.series_identity for item in items):
            reasons.append("SCOPE_MISMATCH")
        if any((item.window_start, item.window_end, item.observed_at) != (start, end, first.observed_at) for item in items):
            reasons.append("WINDOW_MISMATCH")
        if any((item.source_version, item.reset_marker) != (first.source_version, first.reset_marker) for item in items):
            reasons.append("SOURCE_OR_RESET_MISMATCH")
        if step.operator == "RATE":
            unit, kind = first.unit + "/second", "GAUGE"
            if first.kind != "WINDOW_DELTA" or start is None or end is None or start >= end:
                reasons.append("NOT_A_WINDOW_DELTA")
            if not reasons:
                elapsed = Decimal((end - start).days * 86400 + (end - start).seconds) + Decimal((end - start).microseconds) / 1000000
                value = first.value / elapsed
        elif step.operator == "DIVIDE":
            unit = first.unit + "/" + items[1].unit
            if any(item.kind not in {"WINDOW_DELTA", "GAUGE"} for item in items):
                reasons.append("NOT_COMPARABLE_TOTALS")
            if items[1].value is not None and items[1].value <= 0:
                reasons.append("NONPOSITIVE_DENOMINATOR")
            if not reasons:
                value = first.value / items[1].value
        elif step.operator == "SUM":
            if any(item.unit != first.unit or item.kind != first.kind for item in items):
                reasons.append("UNIT_OR_KIND_MISMATCH")
            if first.kind != "WINDOW_DELTA":
                reasons.append("NONADDITIVE_INPUT")
            leaf_sets = [set(item.leaf_refs) for item in inputs]
            if any(left & right for index, left in enumerate(leaf_sets) for right in leaf_sets[index + 1:]):
                reasons.append("DUPLICATED_LINEAGE")
            if not reasons:
                value, kind = sum((item.value for item in items), Decimal(0)), first.kind
    if value is not None and (not value.is_finite() or len(value.as_tuple().digits) > 1024 or abs(value.as_tuple().exponent) > 1024):
        reasons.append("OUTPUT_NUMERIC_BUDGET_EXCEEDED")
    if len(unit) > 128:
        reasons.append("OUTPUT_UNIT_BUDGET_EXCEEDED")
        unit = "UNKNOWN"
    if reasons:
        value = None
    reasons = tuple(dict.fromkeys(reasons))
    missing = any(item.quality == "MISSING" for item in inputs)
    return CalculationValue(
        measurement=ScalarMeasurement(
            input_ref="calculation:" + step.node_id, series_identity=first.series_identity,
            native_origin_id=first.native_origin_id, source_version=first.source_version,
            reset_marker=first.reset_marker, observed_at=max(item.observed_at for item in items),
            value=value, unit=unit, kind=kind, quality="GOOD" if not reasons else "MISSING" if missing else "UNKNOWN",
            trust_level="MODEL_INFERENCE" if any(item.trust_level == "MODEL_INFERENCE" for item in items) else "USER_PROVIDED" if any(item.trust_level == "USER_PROVIDED" for item in items) else "SOURCE_VERIFIED",
            window_start=start, window_end=end,
        ),
        quality="MISSING" if missing else "INCOMPARABLE" if reasons else "GOOD",
        reasons=reasons,
        leaf_refs=tuple(sorted({ref for item in inputs for ref in item.leaf_refs})),
        native_origin_ids=tuple(sorted({origin for item in inputs for origin in item.native_origin_ids})),
    )
