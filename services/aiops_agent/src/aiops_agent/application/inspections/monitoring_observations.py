"""巡检当前快照与历史统计分离；只消费本次冻结的来源事实。"""

from datetime import timedelta

from .daily_baseline import _number, _time


def current_monitoring_policy(inspection):
    """执行前冻结版本化快照窗口和时效，不由模型临时选择。"""
    from .assessment import report_policy

    policy = inspection.get("report_policy") or report_policy()
    return int(policy["current_monitoring_window_seconds"]), int(policy["maximum_current_sample_age_seconds"])


def classify_monitoring_facts(facts, *, monitoring):
    """按冻结窗口区分当前检查和历史参考，不用昨日样本证明当前正常。"""
    current = monitoring.get("current_window")
    if not current:
        return tuple(facts)
    left, right = _time(current.get("start")), _time(current.get("end"))
    result = []
    for original in facts:
        fact = dict(original)
        if fact.get("kind") == "inspection_monitoring_metric":
            if not left or not right or left >= right or _time(fact.get("window_start")) != left or _time(fact.get("window_end")) != right:
                fact["kind"] = "inspection_monitoring_reference"
            else:
                fact["assessment_mode"] = "CURRENT_SNAPSHOT"
                fact["snapshot_at"] = current["end"]
                fact["max_sample_age_seconds"] = monitoring["snapshot_max_age_seconds"]
                stamps = [_time(point.get("observed_at")) for point in fact.get("sample_points") or ()
                          if isinstance(point, dict) and _number(point.get("value")) is not None]
                valid = [stamp for stamp in stamps if stamp is not None and left <= stamp <= right]
                fact["observed_at"] = max(valid).isoformat() if valid else None
        result.append(fact)
    return tuple(result)


def snapshot_is_fresh(fact):
    """快照必须具有可追溯且在策略时效内的有效样本，缺覆盖不冒充历史完整。"""
    observed, captured = _time(fact.get("observed_at")), _time(fact.get("snapshot_at"))
    age = fact.get("max_sample_age_seconds")
    return bool(observed and captured and type(age) is int and age > 0
                and captured - timedelta(seconds=age) <= observed <= captured
                and not set(fact.get("series_quality_flags") or ()) & {
                    "TRUNCATED", "COVERAGE_UNKNOWN", "INVALID_OR_OUTSIDE_SAMPLE",
                    "SERIES_COMPUTATION_BUDGET_EXCEEDED",
                })


def monitoring_quality(facts, *, monitoring=None, gaps=()):
    """已请求但无样本的指标进入分母，历史参考不足不污染当前采样质量。"""
    values = {}
    for fact in facts:
        if fact.get("kind") != "inspection_monitoring_metric":
            continue
        code = str(fact.get("metric_code") or "")
        ratio = _number(fact.get("coverage_ratio"))
        ratio = max(0.0, min(1.0, ratio)) if ratio is not None else 0.0
        if fact.get("assessment_mode") == "CURRENT_SNAPSHOT" and not snapshot_is_fresh(fact):
            ratio = 0.0
        values[code] = min(values.get(code, 1.0), ratio)
    for binding in (monitoring or {}).get("bindings") or ():
        expected = [metric["metric_code"] for metric in binding.get("metrics") or ()]
        expected.extend(binding.get("unsupported_metrics") or ())
        for code in expected:
            values.setdefault(str(code), 0.0)
    for gap in gaps:
        code = str(gap.get("step_id") or "")
        if code and not str(gap.get("code") or "").startswith("REFERENCE_") and code in values:
            values[code] = 0.0
    ratios = list(values.values())
    return {
        "coverage_percent": round(sum(ratios) * 100 / len(ratios), 2) if ratios else None,
        "metric_count": len(ratios),
        "low_coverage_metric_count": sum(value < .8 for value in ratios),
    }
