"""从受控窗口报告复算效果，不把累计最大值、流量下降或人工回执当作优化。"""

from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext

from platform_core.contracts.aiops.effect import EffectSignalRule


def number(value):
    if isinstance(value, bool) or value is None or len(str(value)) > 128:
        raise ValueError("指标缺失或不是有界数值")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("指标不是数值") from None
    if not result.is_finite() or abs(result.adjusted()) > 100:
        raise ValueError("指标超出有限计算范围")
    return result


def report_window(payload):
    start, end = (datetime.fromisoformat(str(payload[key]).replace("Z", "+00:00")) for key in ("period_start", "period_end"))
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("窗口缺少时区或无效")
    return start, end


def validate_window_report(payload):
    if payload.get("schema_version") != "AIOPS_WORKLOAD_REPORT.v2" or payload.get("status") != "READY":
        raise ValueError("需要受控、可计算的窗口工作负载报告")
    start, end = report_window(payload)
    duration = Decimal(str((end - start).total_seconds()))
    if number(payload.get("continuous_seconds")) != duration or int(payload.get("snapshot_count", 0)) < 2:
        raise ValueError("窗口覆盖不完整")
    if payload.get("signal_coverage_version") != "signal-coverage.v1":
        raise ValueError("窗口缺少逐信号的有效覆盖证明")
    if any(item.get("reason") != "COUNTER_RESET" for item in payload.get("discontinuities", ())):
        raise ValueError("窗口存在身份、启动代次、来源或采集连续性缺口")
    identity = payload.get("instance_identity") or {}
    if identity.get("identity_type") not in {"POSTGRESQL_SYSTEM_IDENTIFIER", "MYSQL_SERVER_UUID", "ORACLE_DATABASE_INSTANCE_CONTAINER"} or identity.get("identity_value") in {None, "", "UNKNOWN"}:
        raise ValueError("缺少可核验的物理实例身份")
    if not payload.get("instance_started_at"):
        raise ValueError("缺少实例启动代次")
    return start, end


def _require_signal_coverage(payload, signal):
    """无关缺口不否定本指标；本指标缺少任一时段也不能冒充完整窗口。"""
    start, end = report_window(payload)
    required = number((end - start).total_seconds())
    if number(signal.get("covered_seconds")) != required:
        raise ValueError("所选信号没有覆盖完整窗口，不能用部分累计增量比较效果")


def _observed_formula(payload, rule):
    """仅使用报告已通过连续性检查的窗口增量；指标名来自实际证据而非固定画像。"""
    with localcontext() as context:
        context.prec = 80
        if rule.statement_identity:
            if rule.role == "LOAD" and rule.metric != "statement.calls_per_second":
                raise ValueError("负载可比性必须采用业务调用量，不得以延迟稳定替代")
            allowed = {"statement_identity_type", "statement_identity_value", "database_name", "user_identifier", "top_level", "plan_hash_value"}
            if not {"statement_identity_type", "statement_identity_value", "database_name", "user_identifier", "top_level"} <= rule.statement_identity.keys() or not rule.statement_identity.keys() <= allowed:
                raise ValueError("语句身份必须完整，不得仅凭 SQL ID 混合用户或数据库")
            rows = [item for item in payload.get("top_statements", ()) if all(item.get(key) == value for key, value in rule.statement_identity.items())]
            if len(rows) != 1:
                raise ValueError("目标语句缺失或身份不唯一")
            row = rows[0]
            _require_signal_coverage(payload, row)
            if rule.metric == "statement.mean_duration_microseconds":
                count = number(row.get("execution_count"))
                if count <= 0:
                    raise ValueError("没有可比较的执行次数")
                return number(row.get("total_duration_microseconds")) / count, "us/call", "total_duration_microseconds / execution_count"
            if rule.metric == "statement.rows_per_call":
                count = number(row.get("execution_count"))
                if count <= 0:
                    raise ValueError("没有可比较的执行次数")
                return number(row.get("rows_processed")) / count, "rows/call", "rows_processed / execution_count"
            if rule.metric == "statement.calls_per_second":
                return number(row.get("execution_count")) / number(payload.get("continuous_seconds")), "calls/s", "execution_count / continuous_seconds"
            raise ValueError("语句指标未提供可复算的窗口公式")
        if rule.metric.startswith("load."):
            key = rule.metric.removeprefix("load.")
            if not key.endswith("_per_second") or key not in (payload.get("load_profile") or {}):
                raise ValueError("负载速率不存在")
            semantics = (payload.get("load_dimension_semantics") or {}).get(key) or {}
            _require_signal_coverage(payload, semantics)
            if semantics.get("value_kind") != "COUNTER_RATE" or not semantics.get("unit"):
                raise ValueError("负载指标缺少受控计数器语义和单位")
            if rule.role == "LOAD" and not semantics.get("business_volume"):
                raise ValueError("该计数器不能作为业务输入量的可比条件")
            return number(payload["load_profile"][key]), semantics["unit"], "window_delta / continuous_seconds"
        if rule.metric.startswith("metric."):
            if rule.role == "LOAD":
                raise ValueError("原生资源消耗计数器不能自动当作业务输入量")
            key = rule.metric.removeprefix("metric.")
            native = (payload.get("native_window_metrics") or {}).get(key) or {}
            _require_signal_coverage(payload, native)
            unit = native.get("unit")
            if not unit:
                raise ValueError("原生指标单位未核验")
            return number(native.get("window_delta")) / number(payload.get("continuous_seconds")), f"{unit}/s", "window_delta / continuous_seconds"
        raise ValueError("指标没有经过核验的窗口计算规则")


def calculated_measurement(payload, rule, *, source_ref):
    from aiops_agent.application.effect_calculations import window_calculation
    _, unit, formula = _observed_formula(payload, rule)
    start, end = report_window(payload)
    graph = window_calculation(payload, rule, source_ref=source_ref, start=start, end=end)
    result = graph.values["result"]
    if result.quality != "GOOD" or result.measurement.value is None:
        raise ValueError("受控公式计算失败：" + ";".join(result.reasons))
    return result.measurement.value, unit, formula, graph.model_dump(mode="json")


def measurement(payload, rule):
    return calculated_measurement(payload, rule, source_ref="window-report")[:3]


def evaluate_effect(*, frozen, baseline, after, occurred_at, direct_status, controlled_execution, baseline_ref="baseline", after_ref="after"):
    """完整复算所有冻结条件；保留每项原值、公式与缺口。"""
    rules = tuple(EffectSignalRule.model_validate(item) for item in frozen["signals"])
    gaps, values = [], {}
    before_start, before_end = validate_window_report(baseline)
    after_start, after_end = validate_window_report(after)
    if before_end > occurred_at or after_start.timestamp() < occurred_at.timestamp() + frozen["settle_delay_seconds"]:
        gaps.append("EFFECT_WINDOW_OVERLAPS_ACTION")
    if (after_end - after_start).total_seconds() < frozen["after_window_seconds"]:
        gaps.append("EFFECT_AFTER_WINDOW_TOO_SHORT")
    if (occurred_at - before_end).total_seconds() > frozen["max_baseline_age_seconds"]:
        gaps.append("EFFECT_BASELINE_TOO_OLD")
    for key in ("target_id", "database_type", "instance_identity", "database_versions", "catalog_versions", "capability_probe_versions"):
        if not baseline.get(key) or baseline.get(key) != after.get(key):
            gaps.append("EFFECT_IDENTITY_OR_SOURCE_CHANGED")
    starts = [datetime.fromisoformat(str(source["instance_started_at"]).replace("Z", "+00:00")) for source in (baseline, after)]
    tolerance = 2 if baseline.get("database_type") == after.get("database_type") == "MYSQL" else 0
    if any(item.tzinfo is None for item in starts) or abs((starts[1] - starts[0]).total_seconds()) > tolerance:
        gaps.append("EFFECT_IDENTITY_OR_SOURCE_CHANGED")
    for rule in rules:
        try:
            before, unit, formula, before_graph = calculated_measurement(baseline, rule, source_ref=baseline_ref)
            current, after_unit, after_formula, after_graph = calculated_measurement(after, rule, source_ref=after_ref)
            if unit != after_unit or formula != after_formula:
                raise ValueError("指标单位或口径发生变化")
            threshold = number(rule.threshold)
            with localcontext() as context:
                context.prec = 80
                relative = (current - before) / abs(before) if before else None
                if rule.comparison in {"DECREASE_BY", "INCREASE_BY", "STABLE_WITHIN"}:
                    if before <= 0:
                        raise ValueError("相对比较需要非零正基线")
                    passed = {"DECREASE_BY": relative <= -threshold, "INCREASE_BY": relative >= threshold, "STABLE_WITHIN": abs(relative) <= threshold}[rule.comparison]
                else:
                    passed = current <= threshold if rule.comparison == "AT_MOST" else current >= threshold
            values[rule.key] = {"role": rule.role, "metric": rule.metric, "baseline": str(before), "after": str(current), "relative_change": str(relative) if relative is not None else None, "unit": unit, "formula": formula, "passed": passed,
                               "baseline_calculation": before_graph, "after_calculation": after_graph}
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            gaps.append("EFFECT_SIGNAL_UNAVAILABLE")
            values[rule.key] = {"role": rule.role, "metric": rule.metric, "passed": None, "reason": str(exc)}
    roles = {role: {key: item["passed"] for key, item in values.items() if item["role"] == role} for role in ("PRIMARY", "GUARDRAIL", "LOAD", "SYMPTOM")}
    loads_ok = bool(roles["LOAD"]) and all(item is True for item in roles["LOAD"].values())
    guards_ok = bool(roles["GUARDRAIL"]) and all(item is True for item in roles["GUARDRAIL"].values())
    if not loads_ok:
        gaps.append("EFFECT_WORKLOAD_NOT_COMPARABLE")
    primary_ok = bool(roles["PRIMARY"]) and all(item is True for item in roles["PRIMARY"].values())
    symptom = all(item is True for item in roles["SYMPTOM"].values()) if roles["SYMPTOM"] else None
    if direct_status == "ADVERSE" or any(value is False for value in roles["GUARDRAIL"].values()):
        measured = "DEGRADED"
    elif gaps:
        measured = "INCONCLUSIVE"
    elif any(
        values[rule.key].get("relative_change") is not None
        and ((rule.comparison == "DECREASE_BY" and number(values[rule.key]["relative_change"]) >= number(rule.threshold))
             or (rule.comparison == "INCREASE_BY" and number(values[rule.key]["relative_change"]) <= -number(rule.threshold)))
        for rule in rules if rule.role == "PRIMARY"
    ):
        measured = "DEGRADED"
    elif primary_ok:
        measured = "IMPROVED"
    else:
        measured = "UNCHANGED"
    # 可比窗口改善仍是观察性结果；没有对照或隔离实验，不宣称排除所有混杂因素。
    return {
        "measurements": values,
        "primary_signals": roles["PRIMARY"],
        "guardrail_signals": roles["GUARDRAIL"],
        "load_signals": roles["LOAD"],
        "gap_codes": tuple(sorted(set(gaps))),
        "measured_effect": measured,
        "symptom_recovered": symptom,
        "guardrails_passed": guards_ok if not any(value is None for value in roles["GUARDRAIL"].values()) else None,
        "attribution_strength": "LIMITED" if measured == "IMPROVED" else "UNKNOWN",
        "execution_source": "SOURCE_VERIFIED" if controlled_execution else "USER_PROVIDED",
        "causal_limitations": ("没有隔离实验或对照组，不能排除并行发布、缓存预热等混杂因素",),
        "result": "RESOLVED" if measured == "IMPROVED" and symptom and direct_status == "VERIFIED" else measured,
    }
