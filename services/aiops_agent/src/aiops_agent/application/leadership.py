"""领导简报投影；只保留影响、风险和建议，不下钻 SID。"""

from __future__ import annotations

import re
from typing import Any

from platform_core.contracts.aiops.findings import FindingCard


_SEVERITY_RANK = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
    "INFO": 4,
}
_GRADE_RISK = {
    "CONFIRMED": "HIGH",
    "PROBABLE": "HIGH",
    "POSSIBLE": "MEDIUM",
    "INCONCLUSIVE": "LOW",
}
_FINDING_IMPACT = {
    "LOCK_WAIT": "存在会话阻塞，业务事务可能被拉长等待。",
    "LONG_SESSION": "存在长时间活动会话，可能占用连接和计算资源。",
    "DG_LAG": "备库复制延迟扩大，故障切换窗口被拉长。",
    "REPLICATION_LAG": "复制延迟扩大，从库数据新鲜度下降。",
    "WAIT_CLASS": "前台等待升高，业务响应可能变慢。",
    "TABLESPACE": "存储余量不足，可能影响写入和扩展。",
    "SQL_STATS_STALE": "统计信息过期，执行计划可能偏离实际数据分布。",
    "EXACHECK_FAIL": "平台健康检查失败，基础设施风险需要关注。",
    "EXACHECK_WARNING": "平台健康检查出现警告，存在潜在基础设施风险。",
    "DEAD_TUPLES": "表膨胀升高，查询和清理成本可能增加。",
    "AUTOVACUUM": "事务 ID 年龄偏高，存在 wraparound 风险。",
    "IDLE_SESSION": "空闲会话占用连接，可能挤压业务并发。",
    "CONNECTION_USAGE": "连接资源紧张，新业务会话可能被拒绝。",
    "QUERY_RATE": "查询速率或回滚率异常，业务吞吐可能受影响。",
}
_FINDING_RISK = {
    "LOCK_WAIT": "阻塞持续会扩大超时、堆积和连锁等待。",
    "LONG_SESSION": "长会话不释放会持续占用会话与资源配额。",
    "DG_LAG": "延迟过高时，切换或只读流量会读到过期数据。",
    "REPLICATION_LAG": "延迟过高时，从库查询会偏离主库最新状态。",
    "WAIT_CLASS": "等待类持续升高会放大业务时延。",
    "TABLESPACE": "空间耗尽后写入会失败，影响业务连续性。",
    "SQL_STATS_STALE": "过期统计可能导致低效计划并放大负载。",
    "EXACHECK_FAIL": "基础设施检查失败可能演变为可用性事件。",
    "EXACHECK_WARNING": "未处理的平台警告可能在峰值时暴露为故障。",
    "DEAD_TUPLES": "膨胀持续会拖慢查询并增加维护窗口。",
    "AUTOVACUUM": "wraparound 风险会迫使停写或紧急清理。",
    "IDLE_SESSION": "连接被空闲会话占满后，新业务接入会失败。",
    "CONNECTION_USAGE": "连接耗尽后新会话会被拒绝。",
    "QUERY_RATE": "异常吞吐可能预示应用故障或雪崩。",
}
_SAFE_METRICS = {
    "TABLESPACE": ("used_percent", "当前使用率 {value}%。"),
    "CONNECTION_USAGE": ("utilization_percent", "当前连接使用率 {value}%。"),
    "DG_LAG": ("lag_seconds", "当前复制延迟 {value} 秒。"),
    "REPLICATION_LAG": ("lag_seconds", "当前复制延迟 {value} 秒。"),
}
_IDENTIFIER_PATTERNS = (
    re.compile(r"\bSID\s*[:=#]?\s*\S+", re.IGNORECASE),
    re.compile(r"\bSerial#?\s*[:=]?\s*\S+", re.IGNORECASE),
    re.compile(r"\bSQL[_ ]?ID\s*[:=]?\s*\S+", re.IGNORECASE),
    re.compile(r"\binstance_id\s*[:=]?\s*\S+", re.IGNORECASE),
    re.compile(r"\bsession_id\s*[:=]?\s*\S+", re.IGNORECASE),
    re.compile(r"\b会话\s+\d+"),
)
_ALLOWED_KEYS = (
    "schema_version",
    "risk_level",
    "business_impact",
    "risks",
    "recommendations",
)


def _unique(items: list[str], *, limit: int = 5) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item).split())
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _sanitize(value: object) -> str:
    text = " ".join(str(value or "").split())
    for pattern in _IDENTIFIER_PATTERNS:
        text = pattern.sub("", text)
    text = " ".join(text.split())
    return re.sub(r"^[\s：:;；,，]+|[\s：:;；,，]+$", "", text)


def _metric_note(finding: FindingCard) -> str:
    spec = _SAFE_METRICS.get(str(finding.finding_type))
    if spec is None:
        return ""
    field_name, template = spec
    raw = dict(finding.fields or {}).get(field_name)
    if isinstance(raw, bool) or raw in {None, ""}:
        return ""
    try:
        number = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if number != number or number in {float("inf"), float("-inf")}:
        return ""
    value = str(int(number)) if number == int(number) else str(number)
    return template.format(value=value)


def _finding_impacts(findings: tuple[FindingCard, ...]) -> list[str]:
    items: list[str] = []
    for finding in findings:
        finding_type = str(finding.finding_type)
        text = _FINDING_IMPACT.get(finding_type, "已确认存在需要关注的业务影响。")
        note = _metric_note(finding)
        if note:
            text = f"{text.rstrip('。')}，{note}"
        items.append(text)
    return _unique(items)


def _finding_risks(findings: tuple[FindingCard, ...]) -> list[str]:
    return _unique([
        _FINDING_RISK.get(str(finding.finding_type), "该问题持续可能扩大业务中断窗口。")
        for finding in findings
    ])


def _risk_level(
    *,
    findings: tuple[FindingCard, ...],
    payload: dict[str, Any],
) -> str:
    if findings:
        return str(min(
            findings,
            key=lambda item: _SEVERITY_RANK.get(str(item.severity), 4),
        ).severity)
    scope = dict(payload.get("scope") or {})
    grade = str(scope.get("root_cause_grade") or "INCONCLUSIVE")
    if str(payload.get("status") or "") == "PARTIAL" and grade == "INCONCLUSIVE":
        return "MEDIUM"
    return _GRADE_RISK.get(grade, "LOW")


def _fallback_impacts(payload: dict[str, Any]) -> list[str]:
    items = [_sanitize(payload.get("summary"))]
    for fact in list(payload.get("facts") or ()):
        if not isinstance(fact, dict):
            continue
        items.append(_sanitize(fact.get("summary") or fact.get("fact_summary")))
    cleaned = _unique([item for item in items if item])
    if cleaned:
        return cleaned
    if str(payload.get("report_type") or "").startswith("INSPECTION_"):
        return ["本次巡检未发现需要升级的业务影响。"]
    return ["当前没有需要向领导升级的业务影响。"]


def _fallback_risks(payload: dict[str, Any], *, risk_level: str) -> list[str]:
    items: list[str] = []
    if str(payload.get("status") or "") == "PARTIAL":
        items.append("证据不完整，结论仍有不确定性。")
    if risk_level in {"CRITICAL", "HIGH"}:
        items.append("根因已较明确，业务中断窗口可能继续扩大。")
    elif risk_level == "MEDIUM":
        items.append("存在需要关注的不确定性，建议在业务窗口内复核。")
    else:
        items.append("当前风险可控，维持既定观察即可。")
    return _unique(items)


def _recommendations(payload: dict[str, Any]) -> list[str]:
    items = [
        _sanitize(item)
        for item in list(payload.get("recommendations") or ())
        if _sanitize(item)
    ]
    cleaned = _unique(items)
    if cleaned:
        return cleaned
    if str(payload.get("report_type") or "").startswith("INSPECTION_"):
        return ["继续按既定周期执行巡检。"]
    return ["按正式报告中的处置建议评估后执行。"]


def project_leadership_briefing(
    *,
    payload: dict[str, Any],
    findings: tuple[FindingCard, ...] | list[FindingCard] = (),
) -> dict[str, Any]:
    """从正式报告和可选 Finding 投影领导简报，不输出 SID/SQL。"""
    cards = tuple(findings or ())
    risk_level = _risk_level(findings=cards, payload=payload)
    impacts = _finding_impacts(cards) if cards else _fallback_impacts(payload)
    risks = _finding_risks(cards) if cards else _fallback_risks(
        payload, risk_level=risk_level
    )
    if str(payload.get("status") or "") == "PARTIAL":
        risks = _unique([*risks, "证据不完整，结论仍有不确定性。"])
    briefing = {
        "schema_version": "LEADERSHIP_BRIEFING.v1",
        "risk_level": risk_level,
        "business_impact": impacts,
        "risks": risks,
        "recommendations": _recommendations(payload),
    }
    return {key: briefing[key] for key in _ALLOWED_KEYS}
