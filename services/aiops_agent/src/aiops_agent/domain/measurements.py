"""派生值由程序计算；最大值、分位数和缺失数据不能当作可加计数器。"""

from decimal import Decimal, localcontext

from aiops_agent.contracts.measurements import DerivedMeasurement, ScalarMeasurement


def derive_measurement(formula, before: ScalarMeasurement, after: ScalarMeasurement):
    # 差分必须先扩大有效位，不能让进程默认 Decimal 精度舍入大计数器。
    values = [item.value for item in (before, after) if item.value is not None]
    precision = max((len(value.as_tuple().digits) + abs(value.as_tuple().exponent) for value in values), default=1)
    with localcontext() as context:
        context.prec = max(34, precision + 20)
        return _derive_measurement(formula, before, after)


def _derive_measurement(formula, before, after):
    unit = before.unit + "/s" if formula == "DELTA_PER_SECOND" else before.unit
    reasons = []
    value = None
    if any(item.value is None for item in (before, after)):
        reasons.append("MISSING_VALUE")
    if any(item.quality != "GOOD" or item.trust_level != "SOURCE_VERIFIED" for item in (before, after)):
        reasons.append("INPUT_QUALITY_OR_TRUST")
    if formula in {"AFTER_MINUS_BEFORE", "DELTA_PER_SECOND"}:
        for field in ("series_identity", "native_origin_id", "source_version", "unit", "reset_marker"):
            if getattr(before, field) != getattr(after, field):
                reasons.append(field.upper() + "_CHANGED")
        if not before.reset_marker or not after.reset_marker:
            reasons.append("RESET_BOUNDARY_UNKNOWN")
        if before.kind != "ADDITIVE_COUNTER" or after.kind != "ADDITIVE_COUNTER":
            reasons.append("NOT_ADDITIVE_COUNTER")
        elapsed = Decimal(str((after.observed_at - before.observed_at).total_seconds()))
        if elapsed <= 0:
            reasons.append("INVALID_TIME_ORDER")
        if not reasons:
            value = after.value - before.value
            if value < 0:
                reasons.append("COUNTER_RESET")
                value = None
            elif formula == "DELTA_PER_SECOND":
                value /= elapsed
    elif formula == "TOTAL_DIVIDED_BY_COUNT":
        unit = before.unit + "/" + after.unit
        # 比率必须是同一系列身份及观测时点；不能把全库分母与单 SQL 分子混合。
        if before.series_identity != after.series_identity or before.observed_at != after.observed_at:
            reasons.append("DENOMINATOR_SCOPE_MISMATCH")
        if (before.source_version, before.reset_marker, before.native_origin_id) != (after.source_version, after.reset_marker, after.native_origin_id):
            reasons.append("DENOMINATOR_SOURCE_MISMATCH")
        if not reasons:
            if after.value <= 0:
                reasons.append("NONPOSITIVE_DENOMINATOR")
            else:
                value = before.value / after.value
    else:
        raise ValueError("不支持的派生公式")
    return DerivedMeasurement(
        formula=formula, inputs=(before, after), value=value, unit=unit,
        quality="MISSING" if "MISSING_VALUE" in reasons else "INCOMPARABLE" if reasons else "GOOD",
        reasons=tuple(dict.fromkeys(reasons)),
        native_origin_ids=tuple(sorted({before.native_origin_id, after.native_origin_id})),
    )


def independent_origin_groups(facts):
    """没有原生映射时按目标保守归组，不能因为 Tool 或 Artifact 不同而增加独立支持。"""
    groups = []
    for fact in facts:
        if fact.provenance is None:
            origins = {"unknown"}
        else:
            origins = set(fact.provenance.native_origin_ids)
        overlapping = [index for index, group in enumerate(groups) if group & origins or "unknown" in group or "unknown" in origins]
        for index in reversed(overlapping):
            origins.update(groups.pop(index))
        groups.append(origins)
    return tuple(tuple(sorted(group)) for group in groups)
