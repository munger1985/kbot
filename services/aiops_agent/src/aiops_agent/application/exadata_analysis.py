"""只基于冻结的设备状态和同维历史基线分析，不以偏离冒充故障阈值。"""

from decimal import Decimal, localcontext
from .exadata_status import match_status

from platform_core.contracts.aiops.exadata import ExadataComparison, ExadataFinding


def comparisons(samples, baseline):
    if baseline.truncated or baseline.unavailable_reason:
        return ()
    averages = {tuple(sorted(item.dimensions.items())): item for item in baseline.series}
    latest = {}
    for sample in samples:
        identity = tuple(sorted(sample.dimensions.items()))
        if identity not in latest or sample.observed_at >= latest[identity].observed_at:
            latest[identity] = sample
    result = []
    with localcontext() as context:
        context.prec = 40
        for identity, sample in latest.items():
            average = averages.get(identity)
            if average is None or sample.quality != "OBSERVED" or sample.value in (None, ""):
                continue
            current, mean = Decimal(sample.value), Decimal(average.average)
            if not current.is_finite() or not mean.is_finite():
                continue
            difference = current - mean
            result.append(ExadataComparison(dimensions=dict(identity), observed_at=sample.observed_at,
                current=sample.value, average=average.average, sample_count=average.sample_count,
                difference=str(difference), deviation_percent=str(difference / abs(mean) * 100) if mean else None))
    return tuple(result)


def analyze_view(view):
    """状态异常与数值观察分别解释，每一条规则结论绑定原始样本索引。"""
    definitions = {item.code: item for item in view.definitions}
    names = {item.target_id: item.display_name for item in view.members}
    latest = {}
    for index, sample in enumerate(view.samples):
        key = (sample.component_id, sample.metric_code, tuple(sorted(sample.dimensions.items())))
        if key not in latest or sample.observed_at >= latest[key][1].observed_at:
            latest[key] = (index, sample)
    findings = []
    for index, sample in latest.values():
        definition = definitions[sample.metric_code]
        identity = f"{names.get(sample.component_id, sample.component_id)} · {definition.display_name}"
        if sample.dimensions:
            identity += " · " + ", ".join(f"{key}={value}" for key, value in sorted(sample.dimensions.items()))
        if sample.quality != "OBSERVED" or sample.value in (None, ""):
            code, severity, detail = "CURRENT_EVIDENCE_INSUFFICIENT", "INCONCLUSIVE", f"{identity}：当前证据为空、过旧或截断，需核对 OEM 采集时间与采集错误，不能判定可用性。"
        elif definition.status:
            status = match_status(definition, sample.value)
            if status is None:
                code, severity, detail = "STATUS_UNKNOWN", "INCONCLUSIVE", f"{identity}：未识别状态，需核对原厂状态字典。"
            else:
                bad = status.tone in {"danger", "warning"}
                code, severity = ("COMPONENT_STATE_FINDING", "WARNING") if bad else ("COMPONENT_STATE_OBSERVED", "INCONCLUSIVE")
                detail = f"{identity}：{status.label}。" + (
                    "建议核对该设备/访问点的原厂事件和相关服务；此状态不证明其他组件或数据库异常。" if bad else
                    "仅解释此设备或访问点状态，不据此判定整机或数据库健康。")
        else:
            continue
        findings.append(ExadataFinding(code=code, severity=severity, component_id=sample.component_id,
            metric_code=sample.metric_code, evidence_indexes=(index,), detail=detail))
    return findings


def baseline_observations(result):
    """返回逐维观察及可执行的下一步取证建议，不猜测异常分界线。"""
    if result.baseline.unavailable_reason:
        return [result.baseline.unavailable_reason]
    lines = []
    for item in result.comparisons:
        dimension = ", ".join(f"{key}={value}" for key, value in sorted(item.dimensions.items())) or "无维度"
        delta_unit = "个百分点" if result.unit == "%" else (result.unit or "原生单位")
        deviation = (f"相对偏离 {Decimal(item.deviation_percent):+.2f}%" if item.deviation_percent is not None
                     else "均值为零，相对偏离不可定义")
        lines.append(f"{dimension}：比较采样 {item.current}，历史均值 {item.average}（{item.sample_count} 条留存采样），"
                     f"差值 {item.difference} {delta_unit}，{deviation}。")
    if not lines:
        lines.append("当前有效采样与历史均值没有可比较的完整维度；旧值、缺值不计算偏离。")
    lines.append("偏离只用于定位变化，不是故障阈值；需结合相同时段的负载、延迟、等待和原厂事件进一步核对，不能据此确定根因。")
    return lines
