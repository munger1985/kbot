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


def snapshot_rejection_reason(fact):
    """返回快照被拒绝的具体原因，供判定、报告限制和采样质量共用。"""
    flags = set(fact.get("series_quality_flags") or ())
    reasons = [
        text for code, text in (
            ("TRUNCATED", "采样结果已截断，不能完成当前状态核验"),
            ("COVERAGE_UNKNOWN", "来源采样口径未知，不能确认当前样本有效性"),
            ("INVALID_OR_OUTSIDE_SAMPLE", "采样包含无效值、非正常质量或窗口外时间，需要核对来源时间与冻结窗口"),
            ("SERIES_COMPUTATION_BUDGET_EXCEEDED", "采样计算超过预算，未形成可追溯的当前结果"),
        ) if code in flags
    ]
    if reasons:
        return "；".join(reasons) + "。"
    observed, captured = _time(fact.get("observed_at")), _time(fact.get("snapshot_at"))
    age = fact.get("max_sample_age_seconds")
    if not observed or not captured:
        return "未取得可追溯的有效样本时间或冻结快照时间。"
    if type(age) is not int or age <= 0:
        return "冻结的样本时效规则无效，不能完成当前状态核验。"
    if observed > captured:
        return "最新样本位于冻结快照之后，需要核对来源时钟与查询窗口。"
    if observed < captured - timedelta(seconds=age):
        return f"最新有效样本距快照已超过 {age} 秒，当前样本已过期。"
    return None


def snapshot_is_fresh(fact):
    """快照必须具有可追溯且在策略时效内的有效样本。"""
    return snapshot_rejection_reason(fact) is None


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
