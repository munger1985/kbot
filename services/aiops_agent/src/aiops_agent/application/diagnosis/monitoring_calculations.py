"""监控摘要与冻结计算从同一精确数据生成，容量比值只连接同源同刻同单位序列。"""

import json
from decimal import Decimal, localcontext

from aiops_agent.contracts.series_calculation import SeriesComputation
from aiops_agent.domain.evidence.observations import numeric_wire, parse_numeric_sample
from aiops_agent.domain.evidence.series_calculation import summarize_series, series_number


MAX_TRACE_INPUTS = 8192
MAX_SUMMARY_ROWS = 128


def metric_source_quality_flags(observation):
    """留存层聚合不能由点数或管理员映射升级成原始覆盖证明。"""
    return (("PROVIDER_RETENTION_ROLLUP_POSSIBLE", "COVERAGE_UNKNOWN")
        if observation.provenance.get("history_resolution") == "PROVIDER_MANAGED" else ())


def _scope(observation, series):
    return (str(observation.target_id), str(observation.source_id), observation.source_version,
        str(observation.binding_id), observation.external_target_fingerprint,
        tuple(sorted(series.dimensions.items())), observation.unit)


def build_monitoring_rows(*, artifact_id, result, forecast_horizon_days=None):
    rows, warnings, traces = [], [], []
    maxima = {}
    remaining_inputs = MAX_TRACE_INPUTS
    for observation in result.observations:
        if observation.semantic_role != "STORAGE_MAX" or observation.value_kind != "GAUGE":
            continue
        for series in observation.series:
            valid = [(point.observed_at, number) for point in series.points
                if point.quality == "GOOD" and observation.window_start <= point.observed_at <= observation.window_end
                and (number := parse_numeric_sample(point.value)) is not None]
            if valid and not observation.truncated and observation.coverage_ratio == 1:
                stamp = max(at for at, _ in valid)
                maxima.setdefault((_scope(observation, series), stamp), set()).update(value for at, value in valid if at == stamp)
    for metric_index, observation in enumerate(result.observations):
        warnings.extend(observation.warnings)
        if str(observation.target_id) != str(result.target_id) or str(observation.binding_id) != str(result.binding_id) or (
            result.source_id is not None and str(observation.source_id) != str(result.source_id)
        ):
            warnings.append("MONITORING_SCOPE_MISMATCH：拒绝不属于冻结来源与目标的序列")
            continue
        for series_index, series in enumerate(observation.series):
            if len(rows) >= MAX_SUMMARY_ROWS:
                warnings.append("MONITORING_SERIES_BUDGET_EXCEEDED：摘要仅保留前 128 条序列")
                break
            points, flags = [], list(metric_source_quality_flags(observation))
            for point in series.points:
                number = parse_numeric_sample(point.value)
                if point.quality != "GOOD" or number is None or not observation.window_start <= point.observed_at <= observation.window_end:
                    flags.append("INVALID_OR_OUTSIDE_SAMPLE")
                    continue
                points.append((point.observed_at, number))
            if not series.points:
                continue
            if observation.truncated:
                flags.append("TRUNCATED")
            if observation.coverage_ratio < 1:
                flags.append("PARTIAL_COVERAGE")
            if observation.unit.upper() in {"UNKNOWN", "UNSPECIFIED", ""}:
                flags.append("UNIT_UNKNOWN")
            if len(points) > 4096 or len(points) > remaining_inputs:
                flags.append("SERIES_COMPUTATION_BUDGET_EXCEEDED")
                computed = None
            else:
                computed = summarize_series(tuple(points), value_kind=observation.value_kind)
            summary, daily = computed if computed is not None else ({}, ())
            if computed is None:
                warnings.append(f"{observation.metric_code}: SERIES_COMPUTATION_REJECTED：冲突、无效样本或计算预算不足")
            horizon = None
            confidence = None
            requested_horizon = parse_numeric_sample(forecast_horizon_days)
            elapsed = series_number(summary, "elapsed_days")
            if (computed is not None and not flags and observation.value_kind == "GAUGE"
                and requested_horizon is not None and 0 < requested_horizon <= 365
                and int(summary.get("representative_sample_count") or 0) >= 7
                and elapsed is not None and elapsed >= min(Decimal(7), requested_horizon)):
                horizon = requested_horizon
                summary, daily = summarize_series(tuple(points), value_kind="GAUGE", forecast_horizon_days=horizon)
                confidence = "MEDIUM" if elapsed >= horizon * Decimal("0.8") else "LOW"
            if observation.value_kind != "GAUGE":
                warnings.append(f"{observation.metric_code}: TREND_SEMANTICS_UNSUPPORTED：原始计数器或状态仅保留样本统计，不生成普通趋势预测")
            estimated, utilization = None, None
            with localcontext() as context:
                context.prec = max(80, 2 * max((
                    len(number.as_tuple().digits) + abs(number.as_tuple().exponent)
                    for _, number in points
                ), default=1) + 40)
                slope = series_number(summary, "trend_slope_per_day")
                positive = series_number(summary, "positive_change_ratio")
                future = series_number(summary, "forecast_value")
                latest = series_number(summary, "latest")
                if observation.semantic_role == "STORAGE_USED" and confidence is not None:
                    stamp = max(at for at, _ in points)
                    candidates = maxima.get((_scope(observation, series), stamp), set())
                    maximum = next(iter(candidates)) if len(candidates) == 1 else None
                    if maximum is not None and maximum > 0 and future is not None:
                        utilization = numeric_wire(future / maximum * 100)
                    if maximum is not None and latest is not None and maximum > latest and slope is not None and slope > 0 and positive is not None and positive >= Decimal("0.6"):
                        estimated = numeric_wire((maximum - latest) / slope)
                elif observation.semantic_role == "STORAGE_UTILIZATION" and confidence is not None:
                    utilization = summary.get("forecast_value")
            dimensions = ", ".join(f"{key}={value}" for key, value in sorted(series.dimensions.items()))
            rows.append((
                observation.metric_code, dimensions or "-", summary.get("latest"),
                summary.get("average"), summary.get("maximum"), observation.unit,
                observation.window_start.isoformat(), observation.window_end.isoformat(),
                observation.coverage_ratio,
                *(summary.get(key) for key in ("first", "minimum", "change", "change_per_day",
                    "trend_slope_per_day", "positive_change_ratio")),
                summary.get("sample_count", 0), summary.get("representative_sample_count", 0),
                summary.get("elapsed_days"), confidence, summary.get("forecast_horizon_days"),
                summary.get("forecast_change"), summary.get("forecast_value"), utilization, estimated,
                observation.semantic_role, observation.value_kind,
                json.dumps(daily, ensure_ascii=False, separators=(",", ":")),
                summary.get("sample_p95"),
            ))
            if computed is not None:
                remaining_inputs -= len(points)
                traces.append(SeriesComputation(
                    input_ref=f"artifact:{artifact_id}#/observations/{metric_index}/series/{series_index}/points",
                    row_index=len(rows) - 1, target_id=str(observation.target_id),
                    source_id=str(observation.source_id), source_version=observation.source_version,
                    binding_id=str(observation.binding_id), metric_code=observation.metric_code,
                    unit=observation.unit, value_kind=observation.value_kind,
                    dimensions=tuple(sorted(series.dimensions.items())),
                    window_start=observation.window_start, window_end=observation.window_end,
                    samples=tuple(points), forecast_horizon_days=horizon, summary=summary,
                    quality_flags=tuple(dict.fromkeys(flags)),
                ))
    return tuple(rows), tuple(dict.fromkeys(warnings)), tuple(traces)

async def validate_monitoring_trace_sources(*, uow, run, payload):
    """报告和案例重新核验原始 Artifact；有效计算自述不能替代原始输入。"""
    from uuid import UUID
    from aiops_agent.contracts.evidence import ObservationSet
    from aiops_agent.contracts.turn_answer import TurnEvidenceFact
    from aiops_agent.domain.evidence.series_calculation import replay_series_computation
    from aiops_agent.playbooks import canonical_hash

    sources = {}
    try:
        for raw_fact in payload.get("evidence", ()):
            if not raw_fact.get("series_computations"):
                continue
            fact = TurnEvidenceFact.model_validate(raw_fact)
            for trace in fact.series_computations:
                if str(trace.target_id) != str(run.target_id):
                    return False
                if str(fact.artifact_id) not in sources:
                    artifact = await uow.runs.get_artifact(artifact_id=UUID(str(fact.artifact_id)))
                    if (artifact is None or str(artifact.ops_run_id) != str(run.ops_run_id)
                        or artifact.schema_version != "OBSERVATION_SET.v1"
                        or canonical_hash(artifact.payload_json) != artifact.content_hash):
                        return False
                    sources[str(fact.artifact_id)] = ObservationSet.model_validate(artifact.payload_json)
                source = sources[str(fact.artifact_id)]
                if str(source.target_id) != str(trace.target_id) or str(source.binding_id) != str(trace.binding_id) or str(source.source_id) != str(trace.source_id):
                    return False
                pointer = trace.input_ref.split("#/observations/", 1)
                if len(pointer) != 2 or pointer[0] != f"artifact:{fact.artifact_id}":
                    return False
                metric_index, _, series_index, _ = pointer[1].split("/")
                observation = source.observations[int(metric_index)]
                series = observation.series[int(series_index)]
                expected = {
                    "target_id": str(observation.target_id), "source_id": str(observation.source_id),
                    "source_version": observation.source_version, "binding_id": str(observation.binding_id),
                    "metric_code": observation.metric_code, "unit": observation.unit,
                    "value_kind": observation.value_kind, "dimensions": tuple(sorted(series.dimensions.items())),
                    "window_start": observation.window_start, "window_end": observation.window_end,
                }
                if any(getattr(trace, key) != value for key, value in expected.items()):
                    return False
                original = tuple((point.observed_at, number) for point in series.points
                    if point.quality == "GOOD" and observation.window_start <= point.observed_at <= observation.window_end
                    and (number := parse_numeric_sample(point.value)) is not None)
                if trace.samples != original:
                    return False
                flags = set(metric_source_quality_flags(observation))
                if observation.truncated:
                    flags.add("TRUNCATED")
                if observation.coverage_ratio < 1:
                    flags.add("PARTIAL_COVERAGE")
                if observation.unit.upper() in {"UNKNOWN", "UNSPECIFIED", ""}:
                    flags.add("UNIT_UNKNOWN")
                if len(original) != len(series.points):
                    flags.add("INVALID_OR_OUTSIDE_SAMPLE")
                if set(trace.quality_flags) != flags:
                    return False
                replay_series_computation(trace)
    except (ValueError, TypeError, IndexError, KeyError, AttributeError):
        return False
    return True
