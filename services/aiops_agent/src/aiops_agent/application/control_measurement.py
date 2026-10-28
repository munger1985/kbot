"""复算预登记的匹配对照；不把同向自然改善或人工执行升级为因果证明。"""

from decimal import Decimal, localcontext

from platform_core.contracts.aiops.effect import EffectControlPlan

from aiops_agent.application.effect_measurement import (
    calculated_measurement, number, validate_window_report,
)


def control_source_ids(control):
    """有界、去重的原始报告引用，调用方必须另行核验授权及哈希。"""
    return (
        (str(control.baseline_artifact_id), str(control.target_id)),
        *((str(value), target) for pair in control.prior_pairs for value, target in (
            (pair.treatment_artifact_id, None), (pair.control_artifact_id, str(control.target_id)),
        )),
    )


def _measurement(payload, rule, ref, minimum):
    if rule.statement_identity:
        rows = [row for row in payload.get("top_statements", ()) if all(
            row.get(key) == value for key, value in rule.statement_identity.items()
        )]
        if len(rows) != 1 or number(rows[0].get("execution_count")) < minimum:
            raise ValueError("对照比较的语句执行样本不足或主体不唯一")
    value, unit, formula, graph = calculated_measurement(payload, rule, source_ref=f"artifact:{ref}")
    if value < 0:
        raise ValueError("对照比较不接受负的资源量或业务量")
    return value, unit, formula, graph


def _same_series(first, second):
    for key in ("target_id", "database_type", "instance_identity", "database_versions", "catalog_versions", "capability_probe_versions"):
        if not first.get(key) or first.get(key) != second.get(key):
            raise ValueError("对照序列的物理身份、口径或来源版本发生变化")
    from datetime import datetime
    starts = [datetime.fromisoformat(str(item["instance_started_at"]).replace("Z", "+00:00")) for item in (first, second)]
    tolerance = 2 if first["database_type"] == "MYSQL" else 0
    if any(item.tzinfo is None for item in starts) or abs((starts[0] - starts[1]).total_seconds()) > tolerance:
        raise ValueError("对照序列跨越实例启动代次")


def _physical_family(payload):
    identity = payload["instance_identity"]
    expected = {"POSTGRESQL": "POSTGRESQL_SYSTEM_IDENTIFIER", "MYSQL": "MYSQL_SERVER_UUID",
                "ORACLE": "ORACLE_DATABASE_INSTANCE_CONTAINER"}.get(payload["database_type"])
    if expected is None or identity["identity_type"] != expected:
        raise ValueError("对照数据库引擎与原生物理身份类型不一致")
    value = str(identity["identity_value"])
    if expected == "ORACLE_DATABASE_INSTANCE_CONTAINER":
        # RAC 实例或不同 PDB 仍属于同一 DBID，不能拿逻辑切片作独立对照。
        parts = value.split(":")
        if len(parts) != 3 or any(not item.isdecimal() for item in parts):
            raise ValueError("Oracle 对照缺少可核验的数据库/实例/容器身份")
        value = parts[0]
    return expected, value


def _aligned(treatment, control, plan):
    ts, te = validate_window_report(treatment)
    cs, ce = validate_window_report(control)
    if max(abs((ts - cs).total_seconds()), abs((te - ce).total_seconds())) > plan.alignment_tolerance_seconds:
        raise ValueError("处理组与对照组窗口未对齐")
    if treatment["database_type"] != control["database_type"]:
        raise ValueError("处理组与对照组数据库类型不匹配")
    if str(treatment["target_id"]) == str(control["target_id"]) or _physical_family(treatment) == _physical_family(control):
        raise ValueError("逻辑别名或同一物理数据库不能充当独立对照")
    if str(control["target_id"]) != str(plan.target_id):
        raise ValueError("对照证据不属于冻结目标")
    return ts, te


def _relative(current, baseline):
    if baseline == 0:
        if current == 0:
            return Decimal(0)
        raise ValueError("零基线无法计算非零相对变化")
    return (current - baseline) / abs(baseline)


def validate_control_baselines(*, frozen, baseline, reports):
    """至少三个不重叠的前期窗口；在执行前验证负载及归一化平行趋势。"""
    plan = EffectControlPlan.model_validate(frozen["controlled_comparison"])
    from platform_core.contracts.aiops.effect import EffectSignalRule
    rules = tuple(EffectSignalRule.model_validate(item) for item in frozen["signals"])
    counterparts = {item.key: item for item in plan.signals}
    control_baseline = reports[str(plan.baseline_artifact_id)]
    baseline_start, baseline_end = _aligned(baseline, control_baseline, plan)
    pairs = [(reports[str(pair.treatment_artifact_id)], reports[str(pair.control_artifact_id)],
              str(pair.treatment_artifact_id), str(pair.control_artifact_id)) for pair in plan.prior_pairs]
    pairs.append((baseline, control_baseline, str(frozen["baseline_artifact_id"]), str(plan.baseline_artifact_id)))
    trends, previous_end = {}, None
    with localcontext() as context:
        context.prec = 80
        for treatment, control, treatment_ref, control_ref in pairs:
            start, end = _aligned(treatment, control, plan)
            if previous_end is not None and start < previous_end:
                raise ValueError("前期窗口未按时间排序或相互重叠")
            if end > baseline_end or (end - start) != (baseline_end - baseline_start):
                raise ValueError("前期窗口必须等长且不得晚于冻结基线")
            if (baseline_end - end).total_seconds() > frozen["max_baseline_age_seconds"]:
                raise ValueError("前期窗口距离冻结基线过远")
            previous_end = end
            _same_series(baseline, treatment)
            _same_series(control_baseline, control)
            for rule in rules:
                t, unit, formula, _ = _measurement(treatment, rule, treatment_ref, plan.minimum_statement_calls)
                c, cunit, cformula, _ = _measurement(control, counterparts[rule.key], control_ref, plan.minimum_statement_calls)
                if (unit, formula) != (cunit, cformula):
                    raise ValueError("处理组与对照组的指标单位或公式不一致")
                if rule.role in {"PRIMARY", "LOAD"} and (t <= 0 or c <= 0):
                    raise ValueError("主要效果及负载的对照比较需要正基线")
                trends.setdefault(rule.key, []).append((t, c))
        result = {}
        for rule in rules:
            values = trends[rule.key]
            tbase, cbase = values[-1]
            for t, c in values[:-1]:
                # 变化比例的差，而非原始量的差，允许两组规模不同。
                differential = _relative(t, tbase) - _relative(c, cbase)
                if abs(differential) > number(plan.pretrend_tolerance):
                    raise ValueError("处理组与对照组前期趋势不平行")
                if rule.role == "LOAD" and (
                    abs(_relative(t, tbase)) > number(rule.threshold)
                    or abs(_relative(c, cbase)) > number(rule.threshold)
                ):
                    raise ValueError("前期业务负载与冻结基线不可比")
            result[rule.key] = [{"treatment": str(t), "control": str(c)} for t, c in values]
    return result


def apply_control_comparison(*, result, frozen, baseline, after, reports,
                             control_after_ref, controlled_execution, direct_status,
                             baseline_ref, after_ref):
    """匹配对照支持措施效果，但不声称随机化、完全隔离或已确认根因。"""
    plan = EffectControlPlan.model_validate(frozen["controlled_comparison"])
    gaps = set(result["gap_codes"])
    analysis = {"method": plan.method, "status": "INCONCLUSIVE", "signals": {},
                "after_artifact_id": control_after_ref,
                "baseline_artifact_id": str(plan.baseline_artifact_id)}
    try:
        if not control_after_ref or control_after_ref not in reports:
            raise ValueError("缺少同步对照后窗口")
        if control_after_ref in dict(control_source_ids(plan)) or control_after_ref == str(frozen["baseline_artifact_id"]):
            raise ValueError("对照后窗口不能复用冻结前窗口")
        analysis["pretrends"] = validate_control_baselines(frozen=frozen, baseline=baseline, reports=reports)
        control_before = reports[str(plan.baseline_artifact_id)]
        control_after = reports[control_after_ref]
        _aligned(after, control_after, plan)
        _same_series(control_before, control_after)
        counterparts = {item.key: item for item in plan.signals}
        from platform_core.contracts.aiops.effect import EffectSignalRule
        rules = tuple(EffectSignalRule.model_validate(item) for item in frozen["signals"])
        with localcontext() as context:
            context.prec = 80
            for rule in rules:
                tb, unit, formula, _ = _measurement(baseline, rule, baseline_ref.removeprefix("artifact:"), plan.minimum_statement_calls)
                ta, aunit, aformula, _ = _measurement(after, rule, after_ref.removeprefix("artifact:"), plan.minimum_statement_calls)
                cb, cunit, cformula, cbgraph = _measurement(control_before, counterparts[rule.key], str(plan.baseline_artifact_id), plan.minimum_statement_calls)
                ca, caunit, caformula, cagraph = _measurement(control_after, counterparts[rule.key], control_after_ref, plan.minimum_statement_calls)
                if len({(unit, formula), (aunit, aformula), (cunit, cformula), (caunit, caformula)}) != 1:
                    raise ValueError("后期对照指标的单位或公式发生变化")
                cchange = _relative(ca, cb)
                stable = abs(cchange) <= number(plan.control_stability_tolerance)
                if not stable:
                    gaps.add("EFFECT_CONTROL_NOT_STABLE")
                corrected = None
                passed = None
                if rule.role == "PRIMARY":
                    if min(tb, cb, ca) <= 0:
                        raise ValueError("主要效果归一化对照分母必须为正")
                    corrected = (ta / tb) / (ca / cb) - 1
                    passed = corrected <= -number(rule.threshold) if rule.comparison == "DECREASE_BY" else corrected >= number(rule.threshold)
                    if not passed:
                        gaps.add("EFFECT_CONTROL_ADJUSTED_EFFECT_NOT_MET")
                if rule.role == "LOAD" and abs(cchange) > number(rule.threshold):
                    gaps.add("EFFECT_CONTROL_WORKLOAD_NOT_COMPARABLE")
                # 对照组护栏也必须满足预登记的绝对值或相对变化条件。
                if rule.role == "GUARDRAIL":
                    threshold = number(rule.threshold)
                    guard_ok = {
                        "AT_MOST": ca <= threshold, "AT_LEAST": ca >= threshold,
                        "STABLE_WITHIN": abs(cchange) <= threshold,
                        "DECREASE_BY": cchange <= -threshold,
                        "INCREASE_BY": cchange >= threshold,
                    }[rule.comparison]
                    if not guard_ok:
                        gaps.add("EFFECT_CONTROL_GUARDRAIL_FAILED")
                analysis["signals"][rule.key] = {
                    "unit": unit, "control_before": str(cb), "control_after": str(ca),
                    "control_relative_change": str(cchange), "control_stable": stable,
                    "adjusted_relative_change": str(corrected) if corrected is not None else None,
                    "adjusted_passed": passed,
                    "formula": "(treatment_after / treatment_before) / (control_after / control_before) - 1" if rule.role == "PRIMARY" else None,
                    "control_baseline_calculation": cbgraph, "control_after_calculation": cagraph,
                }
        analysis["status"] = "SUPPORTED" if not gaps else "NOT_SUPPORTED"
    except (KeyError, TypeError, ValueError) as exc:
        gaps.add("EFFECT_CONTROL_EVIDENCE_INVALID")
        analysis["reason"] = str(exc)
    supported = (not gaps and result["measured_effect"] == "IMPROVED"
                 and controlled_execution and direct_status == "VERIFIED")
    analysis["status"] = "SUPPORTED" if supported else "NOT_SUPPORTED"
    result.update(control_analysis=analysis, gap_codes=tuple(sorted(gaps)))
    if supported:
        result["attribution_strength"] = "SUPPORTED"
    result["causal_limitations"] = (
        "预登记匹配对照支持本次措施的效果，不等于随机试验或已确认根因；不能完全排除未观测并行变更及共享基础设施影响",
    )
    return result
