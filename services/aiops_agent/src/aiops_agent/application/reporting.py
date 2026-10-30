"""AIOps 正式报告的模板解析、展示投影与无状态 PDF 渲染。"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from importlib.resources import as_file, files
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from aiops_agent.application.errors import validation_failed
from aiops_agent.application.display_dictionary import (
    display_field_name,
    display_field_value,
    display_metric_dimensions,
    display_metric_name,
    display_name,
    display_text,
    format_metric_value,
)
from aiops_agent.application.leadership import project_leadership_briefing


@dataclass(frozen=True)
class ReportTemplate:
    """报告生成器使用的冻结模板定义。"""

    template_ref: str
    version: str
    display_name: str
    applicable_source_kinds: tuple[str, ...]
    allowed_period_kinds: tuple[str, ...]
    sections: tuple[str, ...]
    definition: dict[str, Any]
    report_types: tuple[str, ...] = ()

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.definition,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


_REQUIRED_SECTIONS = ("EXECUTIVE_SUMMARY", "EVIDENCE_BOUNDARY")
_ALLOWED_SECTIONS = frozenset(
    {
        "EXECUTIVE_SUMMARY", "SCOPE", "ALERT_TIMELINE",
        "INSPECTION_COVERAGE", "RISK_OVERVIEW", "TREND", "FINDINGS",
        "ROOT_CAUSE", "RECOMMENDATIONS", "ACTIONS", "EVIDENCE_BOUNDARY",
        "EVIDENCE_APPENDIX",
    }
)



def _report_display_text(value: object) -> str:
    """使用统一 AIOps 展示字典翻译报告文本。"""
    return display_text(value)


def _target_name_for_presentation(
    *, stored_name: object, target_id: object, report_context: dict[str, Any],
) -> str:
    """保留有效冻结名称，并为历史 UUID 名称投影当前 Target 名称。"""

    normalized_id = str(target_id or "").strip()

    def is_technical_identifier(value: object) -> bool:
        normalized = str(value or "").strip()
        if not normalized or (normalized_id and normalized == normalized_id):
            return True
        try:
            UUID(normalized)
        except (ValueError, AttributeError):
            return False
        return True

    if not is_technical_identifier(stored_name):
        return str(stored_name).strip()
    current_name = report_context.get("target_display_name")
    if not is_technical_identifier(current_name):
        return str(current_name).strip()
    return "目标已删除或无权访问"


def _report_gap_item(value: object) -> str:
    """把证据缺口代码及其上下文转换为可读说明。"""
    if not isinstance(value, dict):
        return _report_display_text(value)
    code = str(value.get("code") or "")
    label = _report_display_text(code or "未知证据缺口")
    turn_no = value.get("turn_no")
    detail = str(value.get("detail") or "").strip()
    if code == "INVESTIGATION_EVIDENCE_GAP" and detail:
        label = _report_display_text(detail)
        detail = ""
    if turn_no is not None:
        label = f"第 {turn_no} 轮：{label}"
    if detail and detail not in label:
        label = f"{label}：{_report_display_text(detail)}"
    category = str(value.get("category") or "")
    category_label = {
        "PRODUCT_CONTRACT": "产品证据契约",
        "MONITORING": "监控采集",
        "REPORT_CONTRACT": "报告契约",
        "SOURCE_ACCESS": "数据库访问",
        "SOURCE_CAPABILITY": "版本或能力",
        "DATA_SOURCE": "数据源取证",
    }.get(category)
    if category_label:
        label = f"[{category_label}] {label}"
    return label


def _report_gap_items(values: list[object]) -> list[str]:
    """按用户可见文本去重，避免不同采集步骤重复报告同一缺口。"""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _report_gap_item(value).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _report_briefing_display(briefing: dict[str, Any]) -> dict[str, Any]:
    """翻译领导简报正文中的稳定枚举，保留原始等级合同。"""
    if not briefing:
        return {}
    result = dict(briefing)
    result["risk_level_display"] = _report_display_text(
        briefing.get("risk_level") or "LOW"
    )
    for key in ("business_impact", "risks", "recommendations"):
        result[key] = [
            _report_display_text(item) for item in briefing.get(key) or ()
        ]
    return result


def _markdown_report_items(value: str) -> list[str]:
    """将巡检 Markdown 归一为报告章节可展示的段落项。"""
    items: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            items.append(" ".join(paragraph))
            paragraph.clear()

    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            title = line.lstrip("#").strip()
            if title and not line.startswith("# "):
                items.append(f"【{title}】")
            continue
        if line.startswith(("- ", "* ")):
            flush()
            items.append(line[2:].strip())
            continue
        if len(line) > 3 and line[0].isdigit() and ". " in line[:4]:
            flush()
            items.append(line)
            continue
        paragraph.append(line)
    flush()
    return [
        item.replace("**", "").replace("`", "")
        for item in items
        if item
    ]


def _comparison_report_items(fact: dict[str, Any]) -> list[str]:
    """把处置验证事实投影为 FINDINGS 可读段落。"""
    items = [f"对比结论：{fact.get('result') or 'INCONCLUSIVE'}"]
    signals = fact.get("primary_signals")
    if isinstance(signals, dict):
        signal_names = {
            "target_absent": "目标对象已消失",
            "blocking_absent": "阻塞关系已消失",
        }
        items.extend(
            f"{signal_names.get(key, key)}："
            f"{('是' if value else '否') if value is not None else '未知'}"
            for key, value in signals.items()
        )
    rationale = [
        str(item) for item in list(fact.get("rationale_codes") or ()) if item
    ]
    if rationale:
        items.append("判定依据：" + "、".join(rationale))
    return items


def _report_fact_items(facts: list[dict[str, Any]]) -> list[str]:
    """按事实语义生成可直接进入 FINDINGS 的可读段落。"""
    result: list[str] = []
    for fact in facts:
        kind = str(fact.get("kind") or "")
        if kind == "agent_health_inspection":
            result.extend(_markdown_report_items(str(fact.get("markdown") or "")))
            continue
        if kind == "comparison_result":
            result.extend(_comparison_report_items(fact))
            continue
        summary = str(
            fact.get("summary") or fact.get("fact_summary") or fact
        )
        observations = [
            item for item in fact.get("observations") or ()
            if isinstance(item, dict)
        ]
        if observations:
            rows = [
                "，".join(
                    f"{display_field_name(key)}：{display_field_value(key, value)}"
                    for key, value in row.items()
                )
                for row in observations
            ]
            summary = f"{summary} 关键观测：{'；'.join(rows)}"
        result.append(summary)
    return result


def _number(value: object) -> float | None:
    """将冻结证据中的数值安全收敛为展示计算可用的浮点数。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _window_days(start: object, end: object) -> float | None:
    """计算监控窗口长度；历史报告缺少窗口字段时不伪造七日基线。"""
    try:
        left = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        right = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return max(0.0, (right - left).total_seconds() / 86_400)


def _daily_metric_projection(fact: dict[str, Any]) -> dict[str, Any]:
    """把一条监控事实投影为七日基线、偏离和小倍数序列。"""
    current = _number(fact.get("latest"))
    baseline = _number(fact.get("average"))
    coverage = _number(fact.get("coverage_ratio"))
    window_days = _window_days(
        fact.get("window_start"), fact.get("window_end")
    )
    value_kind = str(fact.get("value_kind") or "")
    baseline_available = (
        value_kind != "STATE"
        and current is not None
        and baseline is not None
        and baseline != 0
        and coverage is not None
        and coverage >= 0.8
        and window_days is not None
        and window_days >= 6.5
    )
    deviation = current - baseline if baseline_available else None
    deviation_percent = (
        deviation / baseline * 100
        if deviation is not None and baseline is not None
        else None
    )
    unit = str(fact.get("unit") or "")
    dimensions = str(fact.get("dimensions") or "-")
    points = []
    for item in fact.get("daily_points") or ():
        if not isinstance(item, dict):
            continue
        value = _number(item.get("value"))
        date = str(item.get("date") or "")
        if value is None or not date:
            continue
        points.append({
            "date": date,
            "value": value,
            "baseline_index": (
                round(value / baseline * 100, 2)
                if baseline_available and baseline is not None else None
            ),
        })
    metric_code = str(fact.get("metric_code") or "")
    dimension_label = display_metric_dimensions(dimensions).strip()
    return {
        "metric_code": metric_code,
        "semantic_role": str(fact.get("semantic_role") or ""),
        "display_name": display_metric_name(metric_code),
        "dimension_label": dimension_label,
        "unit": unit,
        "current": current,
        "current_display": format_metric_value(current, unit),
        "baseline_average": baseline if baseline_available else None,
        "baseline_display": (
            format_metric_value(baseline, unit)
            if baseline_available else "基线不足"
        ),
        "deviation": deviation,
        "deviation_display": (
            format_metric_value(deviation, unit)
            if deviation is not None else "基线不足"
        ),
        "deviation_percent": (
            round(deviation_percent, 2)
            if deviation_percent is not None else None
        ),
        "baseline_status": (
            "AVAILABLE" if baseline_available else "INSUFFICIENT"
        ),
        "coverage_ratio": coverage,
        "coverage_percent": (
            round(coverage * 100, 2) if coverage is not None else None
        ),
        "window_start": fact.get("window_start"),
        "window_end": fact.get("window_end"),
        "sample_count": fact.get("sample_count"),
        "trend_points": points,
    }


def _daily_inspection_presentation(
    *,
    payload: dict[str, Any],
    report_context: dict[str, Any],
) -> dict[str, Any]:
    """生成日常巡检专用的正式报告结构，文字编辑不能覆盖指标证据。"""
    scope = dict(payload.get("scope") or {})
    target = dict(scope.get("target") or {})
    overrides = dict(payload.get("presentation_overrides") or {})
    metrics = [
        _daily_metric_projection(item)
        for item in payload.get("facts") or ()
        if isinstance(item, dict)
        and item.get("kind") == "inspection_monitoring_metric"
    ]
    metrics.sort(key=lambda item: (
        item["baseline_status"] != "AVAILABLE",
        -abs(item["deviation_percent"] or 0),
        item["display_name"],
    ))
    capacity_roles = {
        "STORAGE_UTILIZATION", "STORAGE_USED", "STORAGE_FREE",
        "STORAGE_MAX", "DATABASE_SIZE", "FILESYSTEM_UTILIZATION",
    }
    capacity_rows = [
        item for item in metrics
        if item.get("semantic_role") in capacity_roles
    ]
    attention_statuses = {"FINDING", "PARTIAL", "GAP", "MISSING", "BREACHED"}
    finding_rows = []
    for fact in payload.get("facts") or ():
        if not isinstance(fact, dict):
            continue
        status = str(fact.get("check_status") or "")
        if status not in attention_statuses:
            continue
        severities = [
            str(item.get("severity") or "")
            for item in fact.get("findings") or ()
            if isinstance(item, dict)
        ]
        finding_rows.append({
            "finding_id": f"F{len(finding_rows) + 1:02d}",
            "title": str(fact.get("title") or "巡检发现"),
            "status": status,
            "severity": severities[0] if severities else "INFO",
            "summary": _report_display_text(
                fact.get("summary") or "已记录需复核的巡检发现。"
            ),
        })
    gaps = _report_gap_items(list(payload.get("gaps") or ()))
    available_metrics = [
        item for item in metrics if item["baseline_status"] == "AVAILABLE"
    ]
    top_deviation = available_metrics[0] if available_metrics else None
    health_status = str(scope.get("health_status") or "UNKNOWN")
    conclusion = str(payload.get("summary") or "本次日常巡检已完成。")
    if top_deviation is not None:
        sign = "+" if float(top_deviation["deviation_percent"]) >= 0 else ""
        conclusion += (
            f" 最大可计算偏离为{top_deviation['display_name']}"
            f"{top_deviation['dimension_label']} {sign}"
            f"{top_deviation['deviation_percent']:.2f}%，"
            "该偏离只表示相对七日基线的变化，需结合指标方向和业务时段复核。"
        )
    elif metrics:
        conclusion += " 当前监控证据不足以形成可靠的过去七日基线偏离。"
    edited_conclusion = overrides.get("EXECUTIVE_SUMMARY")
    if isinstance(edited_conclusion, (list, tuple)) and edited_conclusion:
        conclusion = " ".join(str(item) for item in edited_conclusion)
    environment = _report_display_text(target.get("environment") or "未记录")
    database_type = str(
        target.get("database_type") or scope.get("database_type") or "未记录"
    )
    database_version = str(target.get("version_code") or "未记录")
    db_role = _report_display_text(target.get("db_role") or "数据库运维")
    report_id = str(report_context.get("report_id") or "")
    report_date = str(payload.get("period_end") or "")[:10].replace("-", "")
    report_number = (
        f"AIOPS-DI-{report_date}-{report_id[-8:].upper()}"
        if report_id and report_date else str(payload.get("report_key") or "inspection.daily")
    )
    check_counts = dict(scope.get("check_status_counts") or {})
    total_checks = int(scope.get("evidence_total_count") or sum(check_counts.values()))
    return {
        "schema_version": "DAILY_INSPECTION_PRESENTATION.v1",
        "header": {
            "title": str(payload.get("title") or "数据库日常巡检报告"),
            "report_time": report_context.get("created_at") or payload.get("period_end"),
            "observation_start": (
                metrics[0].get("window_start") if metrics else payload.get("period_start")
            ),
            "observation_end": (
                metrics[0].get("window_end") if metrics else payload.get("period_end")
            ),
            "target_name": _target_name_for_presentation(
                stored_name=target.get("display_name"),
                target_id=target.get("target_id") or payload.get("target_id"),
                report_context=report_context,
            ),
            "target_id": str(target.get("target_id") or payload.get("target_id") or ""),
            "environment": environment,
            "database_type": database_type,
            "database_version": database_version,
            "report_version": int(report_context.get("report_version") or 1),
            "report_number": report_number,
            "responsibility_domain": f"数据库平台 / {db_role} / {environment}运维",
        },
        "conclusion": {
            "text": conclusion,
            "health_status": health_status,
            "health_status_display": _report_display_text(health_status),
            "total_checks": total_checks,
            "attention_count": len(finding_rows),
            "gap_count": len(gaps),
            "evidence_coverage_percent": scope.get("evidence_coverage_percent"),
            "monitoring_coverage_percent": scope.get(
                "monitoring_sample_coverage_percent"
            ),
        },
        "baseline_metrics": metrics,
        "capacity_rows": capacity_rows,
        "findings": finding_rows,
        "finding_notes": [
            _report_display_text(item)
            for item in overrides.get("FINDINGS") or ()
        ],
        "recommendations": [
            _report_display_text(item)
            for item in (
                overrides.get("RECOMMENDATIONS")
                or payload.get("recommendations")
                or ()
            )
        ],
        "gaps": gaps,
        "evidence_boundary": {
            "methodology": (
                "当前值取七日监控窗口内最新有效样本；基线取同一窗口、同一维度和同一口径的样本均值。"
                "采样覆盖率低于 80%、窗口不足 6.5 天、基线为零或状态类指标均不计算偏离。"
            ),
            "sources": [
                f"{item.get('artifact_id', 'evidence')} · {item.get('content_hash', '未提供哈希')}"
                for item in payload.get("evidence_refs") or ()
                if isinstance(item, dict)
            ],
        },
    }


_CAPACITY_METRIC_ROLES = {
    "STORAGE_UTILIZATION", "STORAGE_USED", "STORAGE_FREE", "STORAGE_MAX",
    "DATABASE_SIZE", "FILESYSTEM_UTILIZATION",
}
_BACKUP_RECOVERY_FINDING_TYPES = {
    "BACKUP_MISSING", "BACKUP_FAILED", "RECOVERY_CONFIGURATION",
}


def _percentile_95(values: list[float]) -> float | None:
    """用线性插值计算可复现的 P95；单点序列返回该点。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def is_monthly_backup_recovery_item(value: object) -> bool:
    """识别由未来备份恢复模块接管、不得进入本期月报正文的内容。"""
    fact = value if isinstance(value, dict) else {}
    finding_types = {
        str(item.get("finding_type") or "").upper()
        for item in fact.get("findings") or () if isinstance(item, dict)
    }
    if finding_types & _BACKUP_RECOVERY_FINDING_TYPES:
        return True
    identity = " ".join(str(fact.get(key) or "") for key in (
        "tool_id", "code", "finding_type", "extension_key",
    )).lower()
    if any(token in identity for token in ("backup", "recovery", "rman")):
        return True
    if isinstance(value, str):
        text = value.lower()
        return any(token in text for token in ("rman", "备份", "恢复演练"))
    return False


def monthly_inspection_summary(
    *, facts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    period_start: datetime, period_end: datetime, timezone: str = "UTC",
    run_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从冻结日报事实形成自然月统计，结果可直接随月报冻结。"""
    try:
        report_timezone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        report_timezone = ZoneInfo("UTC")
    start_date = period_start.astimezone(report_timezone).date()
    end_date = period_end.astimezone(report_timezone).date()
    expected_days = max(0, (end_date - start_date).days)
    metric_groups: dict[str, dict[str, Any]] = {}
    issues: dict[str, dict[str, Any]] = {}
    check_counts: dict[str, int] = {}

    for raw_fact in facts:
        if not isinstance(raw_fact, dict) or is_monthly_backup_recovery_item(raw_fact):
            continue
        kind = str(raw_fact.get("kind") or "")
        observed_at = str(raw_fact.get("_source_observed_at") or "")
        if kind == "inspection_monitoring_metric":
            metric_code = str(raw_fact.get("metric_code") or "")
            semantic_role = str(raw_fact.get("semantic_role") or "")
            dimensions = str(raw_fact.get("dimensions") or "-")
            unit = str(raw_fact.get("unit") or "")
            source_target_id = str(raw_fact.get("_source_target_id") or "")
            source_target_name = str(raw_fact.get("_source_target_name") or "")
            key = "|".join((
                source_target_id, metric_code, dimensions, semantic_role, unit,
            ))
            group = metric_groups.setdefault(key, {
                "metric_key": key,
                "metric_code": metric_code,
                "semantic_role": semantic_role,
                "display_name": display_metric_name(metric_code),
                "dimensions": dimensions,
                "dimension_label": " · ".join(filter(None, (
                    source_target_name,
                    display_metric_dimensions(dimensions).strip(),
                ))),
                "target_id": source_target_id or None,
                "target_name": source_target_name or None,
                "unit": unit,
                "value_kind": str(raw_fact.get("value_kind") or ""),
                "days": {},
            })
            for point in raw_fact.get("daily_points") or ():
                if not isinstance(point, dict):
                    continue
                date_text = str(point.get("date") or "")[:10]
                value = _number(point.get("value"))
                try:
                    point_date = datetime.fromisoformat(date_text).date()
                except ValueError:
                    continue
                if value is None or not start_date <= point_date < end_date:
                    continue
                previous = group["days"].get(date_text)
                if previous is None or observed_at >= previous[0]:
                    group["days"][date_text] = (observed_at, {
                        "date": date_text,
                        "value": value,
                        "sample_count": point.get("sample_count"),
                        "minimum": _number(point.get("minimum")),
                        "maximum": _number(point.get("maximum")),
                    })
        elif kind == "inspection_check":
            status = str(raw_fact.get("check_status") or "UNKNOWN")
            check_counts[status] = check_counts.get(status, 0) + 1
            tool_id = str(raw_fact.get("tool_id") or "")
            for finding in raw_fact.get("findings") or ():
                if not isinstance(finding, dict):
                    continue
                finding_type = str(finding.get("finding_type") or "")
                if finding_type.upper() in _BACKUP_RECOVERY_FINDING_TYPES:
                    continue
                fingerprint = str(
                    finding.get("finding_id")
                    or f"{finding_type or 'FINDING'}:{tool_id or raw_fact.get('title') or 'unknown'}"
                )
                issue = issues.setdefault(fingerprint, {
                    "finding_id": fingerprint,
                    "title": str(finding.get("title") or raw_fact.get("title") or "巡检发现"),
                    "summary": str(finding.get("summary") or raw_fact.get("summary") or "已记录巡检发现。"),
                    "severity": str(finding.get("severity") or "INFO"),
                    "occurrence_count": 0,
                    "dates": set(),
                })
                issue["occurrence_count"] += 1
                severity_rank = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}
                severity = str(finding.get("severity") or "INFO")
                if severity_rank.get(severity, 0) > severity_rank.get(issue["severity"], 0):
                    issue["severity"] = severity
                try:
                    issue["dates"].add(
                        datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                        .astimezone(report_timezone).date().isoformat()
                    )
                except ValueError:
                    pass

    metrics: list[dict[str, Any]] = []
    for group in metric_groups.values():
        points = [item[1] for _, item in sorted(group["days"].items())]
        values = [float(item["value"]) for item in points]
        if not values:
            continue
        average = sum(values) / len(values)
        first, last = values[0], values[-1]
        net_change = last - first
        change_percent = net_change / first * 100 if first != 0 else None
        coverage_ratio = len(values) / expected_days if expected_days else None
        unit = str(group["unit"])
        metrics.append({
            **{key: value for key, value in group.items() if key != "days"},
            "monthly_average": round(average, 6),
            "monthly_average_display": format_metric_value(average, unit),
            "p95": round(_percentile_95(values) or 0.0, 6),
            "p95_display": format_metric_value(_percentile_95(values), unit),
            "minimum": min(values),
            "minimum_display": format_metric_value(min(values), unit),
            "maximum": max(values),
            "maximum_display": format_metric_value(max(values), unit),
            "first": first,
            "first_display": format_metric_value(first, unit),
            "last": last,
            "last_display": format_metric_value(last, unit),
            "net_change": round(net_change, 6),
            "net_change_display": format_metric_value(net_change, unit),
            "change_percent": round(change_percent, 2) if change_percent is not None else None,
            "effective_days": len(values),
            "expected_days": expected_days,
            "coverage_ratio": round(coverage_ratio, 6) if coverage_ratio is not None else None,
            "coverage_percent": round(coverage_ratio * 100, 2) if coverage_ratio is not None else None,
            "daily_points": points,
            "is_capacity": group["semantic_role"] in _CAPACITY_METRIC_ROLES,
        })
    metrics.sort(key=lambda item: (
        item["coverage_percent"] is None or item["coverage_percent"] < 80,
        -abs(item.get("change_percent") or 0), item["display_name"],
    ))
    issue_rows = []
    for issue in issues.values():
        dates = sorted(issue.pop("dates"))
        issue_rows.append({
            **issue,
            "affected_days": len(dates),
            "first_seen": dates[0] if dates else None,
            "last_seen": dates[-1] if dates else None,
            "lifecycle": "REPEATED" if len(dates) > 1 else "NEW",
        })
    issue_rows.sort(key=lambda item: (-item["affected_days"], item["finding_id"]))
    effective_metric_days = len({
        point["date"] for metric in metrics for point in metric["daily_points"]
    })
    return {
        "schema_version": "MONTHLY_INSPECTION_SUMMARY.v1",
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "timezone": timezone,
        "metrics": metrics,
        "issues": issue_rows,
        "database_checks": {
            "status_counts": check_counts,
            "observation_count": sum(check_counts.values()),
        },
        "coverage": {
            "expected_days": expected_days,
            "effective_metric_days": effective_metric_days,
            "metric_day_coverage_percent": (
                round(effective_metric_days * 100 / expected_days, 2)
                if expected_days else None
            ),
            **dict(run_summary or {}),
        },
    }


def compare_monthly_inspection_summaries(
    current: dict[str, Any], previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """冻结相同口径指标和问题台账的环比结果。"""
    if not previous:
        return {
            "status": "NO_PREVIOUS_REPORT",
            "status_display": "无上月报告",
            "metric_changes": [], "new_issues": [],
            "persistent_issues": [], "resolved_issues": [],
        }
    if previous.get("schema_version") != "MONTHLY_INSPECTION_SUMMARY.v1":
        return {
            "status": "INCOMPARABLE",
            "status_display": "上月报告口径不可比较",
            "metric_changes": [], "new_issues": [],
            "persistent_issues": [], "resolved_issues": [],
        }
    previous_metrics = {
        str(item.get("metric_key")): item for item in previous.get("metrics") or ()
        if isinstance(item, dict)
    }
    metric_changes = []
    for metric in current.get("metrics") or ():
        before = previous_metrics.get(str(metric.get("metric_key")))
        if not before:
            continue
        current_average = _number(metric.get("monthly_average"))
        previous_average = _number(before.get("monthly_average"))
        if current_average is None or previous_average is None:
            continue
        change = current_average - previous_average
        metric_changes.append({
            "metric_key": metric.get("metric_key"),
            "display_name": metric.get("display_name"),
            "dimension_label": metric.get("dimension_label"),
            "unit": metric.get("unit"),
            "current_average": current_average,
            "current_average_display": metric.get("monthly_average_display"),
            "previous_average": previous_average,
            "previous_average_display": before.get("monthly_average_display"),
            "change": round(change, 6),
            "change_percent": round(change * 100 / previous_average, 2) if previous_average else None,
        })
    current_issues = {
        str(item.get("finding_id")): item for item in current.get("issues") or ()
    }
    previous_issues = {
        str(item.get("finding_id")): item for item in previous.get("issues") or ()
    }
    return {
        "status": "AVAILABLE",
        "status_display": "已完成上月环比",
        "metric_changes": metric_changes,
        "new_issues": [current_issues[key] for key in sorted(current_issues.keys() - previous_issues.keys())],
        "persistent_issues": [current_issues[key] for key in sorted(current_issues.keys() & previous_issues.keys())],
        "resolved_issues": [previous_issues[key] for key in sorted(previous_issues.keys() - current_issues.keys())],
    }


def _monthly_inspection_presentation(
    *, payload: dict[str, Any], report_context: dict[str, Any],
) -> dict[str, Any]:
    """生成与日报同骨架、但按自然月统计的正式展示投影。"""
    scope = dict(payload.get("scope") or {})
    target = dict(scope.get("target") or {})
    summary = dict(scope.get("monthly_summary") or {})
    comparison = dict(scope.get("previous_month_comparison") or {})
    metrics = list(summary.get("metrics") or ())
    coverage = dict(summary.get("coverage") or {})
    issues = list(summary.get("issues") or ())
    report_id = str(report_context.get("report_id") or "")
    report_date = str(payload.get("period_end") or "")[:10].replace("-", "")
    report_number = (
        f"AIOPS-MI-{report_date}-{report_id[-8:].upper()}"
        if report_id and report_date else str(payload.get("report_key") or "inspection.monthly")
    )
    scope_type = str(scope.get("scope_type") or "TARGET")
    stored_scope_name = (
        scope.get("scope_name") or target.get("display_name")
        or payload.get("target_id")
    )
    scope_name = (
        _target_name_for_presentation(
            stored_name=stored_scope_name,
            target_id=target.get("target_id") or payload.get("target_id"),
            report_context=report_context,
        )
        if scope_type == "TARGET"
        else str(stored_scope_name or "未记录")
    )
    conclusion = (
        f"本月覆盖 {coverage.get('completed_run_count', 0)} 次已完成巡检，"
        f"形成 {len(metrics)} 组可统计指标和 {len(issues)} 项问题。"
    )
    if comparison.get("status") == "AVAILABLE":
        conclusion += (
            f" 环比新增问题 {len(comparison.get('new_issues') or ())} 项，"
            f"持续问题 {len(comparison.get('persistent_issues') or ())} 项，"
            f"已解决 {len(comparison.get('resolved_issues') or ())} 项。"
        )
    return {
        "schema_version": "MONTHLY_INSPECTION_PRESENTATION.v1",
        "header": {
            "title": str(payload.get("title") or "数据库月度巡检报告"),
            "report_time": report_context.get("created_at") or payload.get("period_end"),
            "period_start": payload.get("period_start"),
            "period_end": payload.get("period_end"),
            "scope_type": scope_type,
            "scope_name": scope_name,
            "target_count": int(scope.get("target_count") or 1),
            "database_types": list(scope.get("database_types") or [
                target.get("database_type") or scope.get("database_type") or "未记录"
            ]),
            "environment": _report_display_text(target.get("environment") or "未记录"),
            "report_version": int(report_context.get("report_version") or 1),
            "report_number": report_number,
            "responsibility_domain": _report_display_text(
                scope.get("responsibility_domain") or "数据库平台 / 月度运维"
            ),
        },
        "conclusion": {
            "text": conclusion,
            "issue_count": len(issues),
            "metric_count": len(metrics),
            "coverage_percent": coverage.get("metric_day_coverage_percent"),
        },
        "period_comparison": comparison,
        "coverage": coverage,
        "metrics": metrics,
        "capacity_rows": [item for item in metrics if item.get("is_capacity")],
        "issues": issues,
        "database_checks": dict(summary.get("database_checks") or {}),
        "recommendations": [
            _report_display_text(item) for item in payload.get("recommendations") or ()
        ],
        "gaps": _report_gap_items(list(payload.get("gaps") or ())),
        "extensions": list(scope.get("monthly_report_extensions") or ({
            "schema_version": "MONTHLY_REPORT_EXTENSION.v1",
            "extension_key": "backup_recovery",
            "status": "RESERVED",
            "items": [],
        },)),
        "evidence_boundary": {
            "methodology": (
                "指标按自然月、同一指标与维度逐日去重后统计月均值、P95、峰谷和变化；"
                "覆盖不足和口径不可比较项单独标记。生成日瞬时值不进入月度结论。"
            ),
            "sources": [
                f"{item.get('artifact_id', 'evidence')} · {item.get('content_hash', '未提供哈希')}"
                for item in payload.get("evidence_refs") or () if isinstance(item, dict)
            ],
        },
    }


def closed_period_window(
    *, period_kind: str, timezone: str, now: datetime,
) -> tuple[datetime, datetime]:
    """计算最近一个已完整闭合的自然报告周期，返回 UTC 半开区间。"""
    if period_kind not in {"MONTHLY", "QUARTERLY", "ANNUAL"}:
        raise validation_failed("周期报告类型必须为月度、季度或年度")
    try:
        local_now = now.astimezone(ZoneInfo(timezone))
    except ZoneInfoNotFoundError as exc:
        raise validation_failed("报告时区必须是有效的 IANA 时区") from exc
    if period_kind == "MONTHLY":
        end = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start = (end - timedelta(days=1)).replace(day=1)
    elif period_kind == "QUARTERLY":
        month = ((local_now.month - 1) // 3) * 3 + 1
        end = local_now.replace(month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        start = end.replace(year=end.year - 1, month=10) if end.month == 1 else end.replace(month=end.month - 3)
    else:
        end = local_now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        start = end.replace(year=end.year - 1)
    return start.astimezone(UTC), end.astimezone(UTC)


_INSPECTION_SECTIONS = (
    "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
    "RISK_OVERVIEW", "FINDINGS", "RECOMMENDATIONS",
    "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
)
_PERIOD_INSPECTION_SECTIONS = (
    "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
    "RISK_OVERVIEW", "TREND", "FINDINGS", "ACTIONS",
    "RECOMMENDATIONS", "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
)


def _system_template(
    *, key: str, name: str, source_kinds: tuple[str, ...],
    periods: tuple[str, ...], sections: tuple[str, ...],
    report_types: tuple[str, ...],
) -> ReportTemplate:
    definition = {
        "schema_version": "REPORT_LAYOUT.v1",
        "display_name": name,
        "applicable_source_kinds": list(source_kinds),
        "allowed_period_kinds": list(periods),
        "sections": [{"kind": item} for item in sections],
    }
    return ReportTemplate(
        template_ref=f"system:{key}", version="1", display_name=name,
        applicable_source_kinds=source_kinds, allowed_period_kinds=periods,
        sections=sections, definition=definition, report_types=report_types,
    )


SYSTEM_REPORT_TEMPLATES = {
    item.template_ref: item
    for item in (
        _system_template(
            key="oracle.workload", name="Oracle 周期快照工作负载报告（非 AWR）",
            source_kinds=("INSPECTION",), periods=("CUSTOM",),
            report_types=("ORACLE_WORKLOAD",),
            sections=("EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE", "TREND", "FINDINGS", "RECOMMENDATIONS", "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX"),
        ),
        _system_template(
            key="oracle.workload_diff", name="Oracle 周期快照工作负载对比（非 AWR）",
            source_kinds=("INSPECTION",), periods=("CUSTOM",),
            report_types=("ORACLE_WORKLOAD_DIFF",),
            sections=("EXECUTIVE_SUMMARY", "SCOPE", "TREND", "FINDINGS", "RECOMMENDATIONS", "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX"),
        ),
        _system_template(
            key="diagnosis.standard", name="标准诊断报告",
            source_kinds=("CHAT", "ALERT"), periods=("AD_HOC",),
            report_types=("INCIDENT", "PERFORMANCE"),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "ALERT_TIMELINE",
                "ROOT_CAUSE", "FINDINGS", "RECOMMENDATIONS", "ACTIONS",
                "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="comparison.business", name="业务效果验证报告",
            source_kinds=("CHAT", "ALERT"), periods=("AD_HOC",),
            report_types=("COMPARISON",),
            sections=("EXECUTIVE_SUMMARY", "SCOPE", "FINDINGS", "ACTIONS", "RECOMMENDATIONS", "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX"),
        ),
        _system_template(
            key="mysql.workload", name="MySQL Workload Report",
            source_kinds=("INSPECTION",), periods=("CUSTOM",),
            report_types=("MYSQL_WORKLOAD",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
                "TREND", "FINDINGS", "RECOMMENDATIONS",
                "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="mysql.workload_diff", name="MySQL Workload Diff Report",
            source_kinds=("INSPECTION",), periods=("CUSTOM",),
            report_types=("MYSQL_WORKLOAD_DIFF",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "TREND", "FINDINGS",
                "RECOMMENDATIONS", "EVIDENCE_BOUNDARY",
                "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="mysql.activity", name="MySQL Activity Report",
            source_kinds=("INSPECTION",), periods=("CUSTOM",),
            report_types=("MYSQL_ACTIVITY",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
                "FINDINGS", "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="mysql.sql_healthcheck", name="MySQL SQL Healthcheck",
            source_kinds=("CHAT",), periods=("AD_HOC",),
            report_types=("MYSQL_SQL_HEALTHCHECK",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "ROOT_CAUSE", "FINDINGS",
                "RECOMMENDATIONS", "ACTIONS", "EVIDENCE_BOUNDARY",
                "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="postgresql.workload", name="PostgreSQL Workload Report",
            source_kinds=("INSPECTION",), periods=("CUSTOM",),
            report_types=("POSTGRESQL_WORKLOAD",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
                "TREND", "FINDINGS", "RECOMMENDATIONS",
                "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="postgresql.workload_diff", name="PostgreSQL Workload Diff Report",
            source_kinds=("INSPECTION",), periods=("CUSTOM",),
            report_types=("POSTGRESQL_WORKLOAD_DIFF",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "TREND", "FINDINGS",
                "RECOMMENDATIONS", "EVIDENCE_BOUNDARY",
                "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="postgresql.activity", name="PostgreSQL Activity Report",
            source_kinds=("INSPECTION",), periods=("CUSTOM",),
            report_types=("POSTGRESQL_ACTIVITY",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
                "FINDINGS", "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="postgresql.sql_healthcheck", name="PostgreSQL SQL Healthcheck",
            source_kinds=("CHAT",), periods=("AD_HOC",),
            report_types=("POSTGRESQL_SQL_HEALTHCHECK",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "ROOT_CAUSE", "FINDINGS",
                "RECOMMENDATIONS", "ACTIONS", "EVIDENCE_BOUNDARY",
                "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="postgresql.pgbadger", name="PostgreSQL pgBadger Report",
            source_kinds=("EXTERNAL_REPORT",), periods=("CUSTOM",),
            report_types=("POSTGRESQL_PGBADGER",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "FINDINGS",
                "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="inspection.daily", name="日常巡检报告",
            source_kinds=("INSPECTION",), periods=("DAILY",),
            report_types=("INSPECTION_DAILY",),
            sections=_INSPECTION_SECTIONS,
        ),
        _system_template(
            key="inspection.weekly", name="周度巡检报告",
            source_kinds=("INSPECTION",), periods=("WEEKLY",),
            report_types=("INSPECTION_WEEKLY",),
            sections=_INSPECTION_SECTIONS,
        ),
        _system_template(
            key="inspection.custom", name="定期巡检报告",
            source_kinds=("INSPECTION",), periods=("CUSTOM",),
            report_types=("INSPECTION_CUSTOM",),
            sections=_INSPECTION_SECTIONS,
        ),
        _system_template(
            key="inspection.monthly", name="月度巡检报告",
            source_kinds=("INSPECTION",), periods=("MONTHLY",),
            report_types=("INSPECTION_MONTHLY",),
            sections=_PERIOD_INSPECTION_SECTIONS,
        ),
        _system_template(
            key="inspection.quarterly", name="季度巡检报告",
            source_kinds=("INSPECTION",), periods=("QUARTERLY",),
            report_types=("INSPECTION_QUARTERLY",),
            sections=_PERIOD_INSPECTION_SECTIONS,
        ),
        _system_template(
            key="inspection.annual", name="年度巡检报告",
            source_kinds=("INSPECTION",), periods=("ANNUAL",),
            report_types=("INSPECTION_ANNUAL",),
            sections=_PERIOD_INSPECTION_SECTIONS,
        ),
        _system_template(
            key="comparison.standard", name="处置验证报告",
            source_kinds=("CHAT", "ALERT"), periods=("CUSTOM",),
            report_types=("COMPARISON",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "FINDINGS", "ACTIONS",
                "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
    )
}

REPORT_TYPE_SYSTEM_TEMPLATES = {
    report_type: template
    for template in SYSTEM_REPORT_TEMPLATES.values()
    for report_type in template.report_types
}


def list_system_templates() -> list[dict[str, Any]]:
    """返回页面选择器所需的系统模板摘要。"""
    return [template_summary(item) for item in SYSTEM_REPORT_TEMPLATES.values()]


def template_summary(template: ReportTemplate) -> dict[str, Any]:
    return {
        "template_ref": template.template_ref,
        "version": template.version,
        "display_name": template.display_name,
        "applicable_source_kinds": list(template.applicable_source_kinds),
        "allowed_period_kinds": list(template.allowed_period_kinds),
        "sections": list(template.sections),
        "content_hash": template.content_hash,
        "system_defined": template.template_ref.startswith("system:"),
    }


def resolve_system_template(template_ref: str) -> ReportTemplate | None:
    return SYSTEM_REPORT_TEMPLATES.get(template_ref)


def resolve_report_template_reference(template_ref: str) -> ReportTemplate | None:
    """解析报告冻结的系统模板标识，兼容早期未带 system 前缀的快照。"""
    direct = resolve_system_template(template_ref)
    if direct is not None or template_ref.startswith("system:"):
        return direct
    return resolve_system_template(f"system:{template_ref}")


def resolve_report_type_template(report_type: str) -> ReportTemplate | None:
    """按正式报告类型解析对应的系统模板。"""
    return REPORT_TYPE_SYSTEM_TEMPLATES.get(report_type)


def resolve_historical_report_template(
    *, template_ref: str, report_type: str,
) -> ReportTemplate | None:
    """为缺少模板快照的历史报告解析确定性的系统报告模板。"""
    resolved = resolve_report_template_reference(template_ref)
    if resolved is not None:
        return resolved
    return resolve_report_type_template(report_type)


def template_snapshot(template: ReportTemplate) -> dict[str, Any]:
    """冻结报告使用的模板引用、版本和完整定义。"""
    return {
        "template_ref": template.template_ref,
        "version": template.version,
        "content_hash": template.content_hash,
        "definition": template.definition,
    }


def validate_template_definition(definition: dict[str, Any]) -> ReportTemplate:
    """校验 Domain 模板的受控章节 DSL。"""
    source_kinds = tuple(str(item) for item in definition.get(
        "applicable_source_kinds", ("CHAT", "ALERT", "INSPECTION")
    ))
    periods = tuple(str(item) for item in definition.get(
        "allowed_period_kinds", ("AD_HOC", "DAILY", "MONTHLY", "QUARTERLY", "ANNUAL")
    ))
    raw_sections = definition.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise validation_failed("报告模板必须至少包含一个章节")
    sections = tuple(
        str(item.get("kind")) if isinstance(item, dict) else ""
        for item in raw_sections
    )
    if any(item not in _ALLOWED_SECTIONS for item in sections):
        raise validation_failed("报告模板包含不支持的章节类型")
    if any(item not in sections for item in _REQUIRED_SECTIONS):
        raise validation_failed("报告模板必须保留摘要和证据边界章节")
    if not source_kinds or any(item not in {"CHAT", "ALERT", "INSPECTION"} for item in source_kinds):
        raise validation_failed("报告模板适用入口无效")
    if not periods or any(item not in {"AD_HOC", "DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "ANNUAL", "CUSTOM"} for item in periods):
        raise validation_failed("报告模板适用周期无效")
    name = str(definition.get("display_name") or "自定义报告模板").strip()
    return ReportTemplate(
        template_ref="", version="", display_name=name,
        applicable_source_kinds=source_kinds, allowed_period_kinds=periods,
        sections=sections, definition=definition,
    )


def normalize_report_source(
    *, schema_version: str, payload: dict[str, Any], source_kind: str,
) -> dict[str, Any]:
    """将三个入口的最终产物归一为报告装配所需的公开事实。"""
    if schema_version == "DIAGNOSIS_CLOSURE.v1":
        profile = dict(payload.get("profile") or {})
        root_cause = dict(profile.get("root_cause") or {})
        verification = dict(profile.get("verification") or {})
        return {
            "status": "READY",
            "root_cause": {"effective_level": str(root_cause.get("effective_level") or "INCONCLUSIVE")},
            "diagnosis_rationale": str(root_cause.get("summary") or ""),
            "facts": tuple({
                "kind": "diagnosis_closure",
                "summary": str(item),
                "trust_level": "USER_PROVIDED",
            } for item in dict(profile.get("problem_signature") or {}).get(
                "normalized_symptoms", ()
            )),
            "gaps": (),
            "solution": {},
            "evidence_refs": tuple({
                "artifact_id": str(item["artifact_id"]), "source_run_id": str(item["run_id"]),
                "schema_version": item["schema_version"], "content_hash": item["content_hash"],
                "observed_at": item["observed_at"],
            } for item in payload.get("source_refs", ())) or tuple({"artifact_id": str(item)} for item in (
                *tuple(root_cause.get("evidence_refs") or ()),
                *tuple(verification.get("before_refs") or ()),
                *tuple(verification.get("after_refs") or ()),
            )),
            "executed_actions": tuple(
                str(item.get("summary") or "")
                for item in profile.get("actions", ())
                if isinstance(item, dict) and item.get("summary")
            ),
            "verification": verification,
            "action_source": "USER_PROVIDED",
            "closure_artifact_id": payload.get("artifact_id"),
            "conversation": {
                "conversation_id": payload.get("conversation_id"),
                "included_turn_ids": list(payload.get("source_turn_ids") or ()),
                "turn_count": len(payload.get("source_turn_ids") or ()),
            },
        }
    if schema_version == "DIAGNOSIS_REPORT_DRAFT.v1":
        return {
            "status": str(payload.get("status") or "PARTIAL"),
            "root_cause": dict(payload.get("root_cause") or {}),
            "diagnosis_rationale": str(
                payload.get("diagnosis_rationale") or ""
            ),
            "facts": tuple(dict(item) for item in payload.get("facts", ())),
            "gaps": tuple(payload.get("gaps", ())),
            "solution": dict(payload.get("solution") or {}),
            "model_receipt_hashes": tuple(
                payload.get("model_receipt_hashes", ())
            ),
            "report_decision_reasons": tuple(
                payload.get("report_decision_reasons", ())
            ),
        }
    if schema_version == "AIOPS_TURN_RESULT.v1":
        from aiops_agent.application.diagnosis.investigation_record import investigation_report_source
        from aiops_agent.contracts.investigation import InvestigationRecord

        evidence = tuple(dict.fromkeys(
            str(ref) for block in payload.get("blocks", ())
            for ref in dict(block).get("evidence_refs", ())
        ))
        if payload.get("investigation_record") is not None:
            record = InvestigationRecord.model_validate(payload["investigation_record"])
            # 再验不可变记录与本结果证据一致，报告不能接受游离主张。
            from aiops_agent.application.diagnosis.investigation_record import validate_diagnosis
            from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
            from aiops_agent.contracts.turn_answer import TurnEvidenceFact
            facts = tuple(TurnEvidenceFact.model_validate(item) for item in payload.get("evidence", ()))
            record.validate_observations(facts)
            validate_diagnosis(DiagnosisAnswerDraft(
                claims=record.claims, recommendations=record.recommendations,
                goal_satisfied=False,
            ), facts)
            source = investigation_report_source(record)
            if payload.get("status") != "COMPLETED":
                source["status"] = "PARTIAL"
            source["gaps"] = (*source["gaps"], *payload.get("evidence_gaps", ()))
            return source
        body, solutions = [], []
        for block in payload.get("blocks", ()):
            kind = str(dict(block).get("block_type"))
            markdown = str(dict(block).get("payload", {}).get("markdown") or "").strip()
            if kind in {"MARKDOWN", "ANALYSIS_MARKDOWN"} and markdown:
                body.append(markdown)
            elif kind == "SOLUTION_MARKDOWN" and markdown:
                solutions.append(markdown)
        gaps = list(payload.get("evidence_gaps", ()))
        if not body:
            gaps.append({"code": "REPORT_BODY_MISSING", "detail": "源结果未包含可报告的分析正文"})
        available = {str(item.get("evidence_ref")) for item in payload.get("evidence", ())}
        if not set(evidence) <= available:
            gaps.append({"code": "REPORT_REFERENCE_MISSING", "detail": "正文引用不在冻结证据集合"})
        return {
            "status": "READY" if payload.get("status") == "COMPLETED" and body and not gaps else "PARTIAL",
            "root_cause": {"effective_level": "INCONCLUSIVE"},
            "diagnosis_rationale": "\n\n".join(body) or "源结果正文缺失，报告内容尚未完整。",
            "facts": (),
            "gaps": tuple(gaps),
            "solution": {"long_term_remediations": tuple(solutions)},
            "evidence_refs": evidence,
        }
    if schema_version == "REPORT_CONTENT.v1":
        scope = dict(payload.get("scope") or {})
        return {
            "status": str(payload.get("status") or "PARTIAL"),
            "root_cause": {
                "effective_level": str(
                    scope.get("root_cause_grade") or "INCONCLUSIVE"
                ),
            },
            "diagnosis_rationale": str(scope.get("diagnosis_rationale") or payload.get("summary") or ""),
            "facts": tuple(
                dict(item) for item in payload.get("facts", ())
                if isinstance(item, dict)
            ),
            "gaps": tuple(payload.get("gaps", ())),
            "solution": {
                "long_term_remediations": tuple(
                    payload.get("recommendations", ())
                ),
            },
            "evidence_refs": tuple(payload.get("evidence_refs", ())),
            "inspection_coverage": scope.get("inspection_coverage"),
            "investigation_goals": tuple(scope.get("investigation_goals") or ()),
            "executed_actions": tuple(scope.get("executed_actions") or ()),
            "action_source": scope.get("action_source"),
            "verification": dict(scope.get("verification") or {}),
        }
    if schema_version == "DB_DIAGNOSTIC_REPORT.v1" and source_kind == "INSPECTION":
        observation_count = int(payload.get("observation_count") or 0)
        return {
            "status": "PARTIAL" if payload.get("status") == "PARTIAL" else "READY",
            "root_cause": {"effective_level": "INCONCLUSIVE"},
            "diagnosis_rationale": (
                f"本次巡检完成 {observation_count} 项观测，"
                f"记录 {int(payload.get('gap_count') or 0)} 个数据缺口。"
            ),
            "facts": ({
                "kind": "inspection_coverage",
                "summary": f"已完成 {observation_count} 项观测",
                "tools": list(payload.get("tools") or ()),
            },),
            "gaps": tuple(payload.get("gaps", ())),
            "solution": {},
        }
    raise validation_failed("当前诊断结果不支持生成正式报告")


def report_presentation(
    *, payload: dict[str, Any], template: ReportTemplate,
    findings: tuple[Any, ...] | list[Any] = (),
    report_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把不可变报告内容投影为前端预览和文档渲染的共同输入。"""
    facts = list(payload.get("facts") or ())
    gaps = list(payload.get("gaps") or ())
    recommendations = list(payload.get("recommendations") or ())
    overrides = dict(payload.get("presentation_overrides") or {})
    scope = dict(payload.get("scope") or {})
    root_grade = str(scope.get("root_cause_grade") or "INCONCLUSIVE")
    section_data: list[dict[str, Any]] = []
    for kind in template.sections:
        if kind == "EXECUTIVE_SUMMARY":
            body = [str(payload.get("summary") or "未形成摘要")]
        elif kind == "SCOPE":
            body = [f"报告时间窗：{payload.get('period_start')} 至 {payload.get('period_end')}"]
            from aiops_agent.application.report_lifecycle import report_goal_items
            body.extend(report_goal_items(scope))
            conversation = dict(scope.get("conversation") or {})
            if conversation:
                body.append(
                    f"会话：{conversation.get('title') or '未命名诊断'}；"
                    f"覆盖 {conversation.get('turn_count') or 0} 个 Turn。"
                )
                body.extend(
                    f"诊断问题：{item}"
                    for item in conversation.get("question_summaries") or ()
                )
        elif kind == "ALERT_TIMELINE":
            body = [str(scope.get("alert_summary") or "本报告未关联告警时间线")]
        elif kind == "INSPECTION_COVERAGE":
            body = [str(scope.get("inspection_coverage") or "本报告未包含巡检覆盖统计")]
        elif kind == "RISK_OVERVIEW":
            body = [f"根因评估等级：{root_grade}"]
        elif kind == "TREND":
            body = [str(item) for item in list(scope.get("trends") or ())] or ["当前报告范围内没有可复现的趋势数据。"]
        elif kind == "FINDINGS":
            body = _report_fact_items(facts) or ["未记录已验证发现。"]
        elif kind == "ROOT_CAUSE":
            body = [f"根因评估等级：{root_grade}", str(scope.get("diagnosis_rationale") or "当前结论以证据边界章节为准。")]
        elif kind == "RECOMMENDATIONS":
            body = [str(item) for item in recommendations] or ["当前没有可执行建议。"]
        elif kind == "ACTIONS":
            from aiops_agent.application.report_lifecycle import report_action_items
            body = report_action_items(scope) or ["当前没有已记录的处置或验证动作。"]
        elif kind == "EVIDENCE_BOUNDARY":
            body = _report_gap_items(gaps) or [
                "未发现额外的数据缺口。"
            ]
        else:
            body = [
                f"{item.get('artifact_id', 'evidence')} · {item.get('content_hash', '未提供哈希')}"
                for item in list(payload.get("evidence_refs") or ())
            ] or ["未记录可公开的证据索引。"]
        if kind == "EVIDENCE_APPENDIX":
            from aiops_agent.application.diagnosis.sql_plans import controlled_plan_appendix
            body.extend(controlled_plan_appendix(scope.get("sql_plan_assessments", ())))
        if kind not in {"EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX"}:
            edited = overrides.get(kind)
            if isinstance(edited, (list, tuple)) and edited:
                body = [str(item) for item in edited]
        body = [_report_display_text(item) for item in body]
        section_data.append({
            "kind": kind,
            "display_name": display_name(kind),
            "items": body,
            "human_edited": kind in overrides,
        })
    briefing = project_leadership_briefing(
        payload=payload, findings=findings,
    )
    presentation = {
        "schema_version": "REPORT_PRESENTATION.v1",
        "title": payload.get("title") or template.display_name,
        "status": payload.get("status"),
        "status_display": _report_display_text(
            payload.get("status") or "UNKNOWN"
        ),
        "template": {**template_summary(template), "definition": template.definition},
        "report": payload,
        "sections": section_data,
        "leadership_briefing": _report_briefing_display(briefing),
    }
    if (
        str(payload.get("report_type") or "") == "INSPECTION_DAILY"
        or template.template_ref == "system:inspection.daily"
    ):
        presentation["daily_inspection"] = _daily_inspection_presentation(
            payload=payload,
            report_context=dict(report_context or {}),
        )
    if (
        str(payload.get("report_type") or "") == "INSPECTION_MONTHLY"
        or template.template_ref == "system:inspection.monthly"
    ):
        presentation["monthly_inspection"] = _monthly_inspection_presentation(
            payload=payload,
            report_context=dict(report_context or {}),
        )
    return presentation


def _pdf_text(value: str) -> str:
    """将 PDF 字体不能表达的控制字符替换为可见占位符。"""
    return "".join(character if 0x20 <= ord(character) <= 0xFFFF else "?" for character in value)


@lru_cache(maxsize=1)
def _pdf_report_font_name() -> str:
    """注册随服务发布的中文 TrueType 字体，供标准 PDF 生成器使用。"""
    font_name = "KBotWQYMicroHei"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        resource = files("aiops_agent").joinpath("resources/fonts/wqy-microhei.ttc")
        with as_file(resource) as path:
            pdfmetrics.registerFont(TTFont(font_name, str(path), subfontIndex=0))
    return font_name


def _pdf_paragraph(value: object) -> str:
    """将报告原文安全转换为 PDF 段落文本。"""
    from html import escape

    return escape(_pdf_text(str(value))).replace("\n", "<br/>")


def _report_page_chrome(canvas: Canvas, document: SimpleDocTemplate) -> None:
    """绘制正式报告的统一页眉、页脚与页码。"""
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#1D4E6D"))
    canvas.setLineWidth(0.6)
    canvas.line(20 * mm, height - 16 * mm, width - 20 * mm, height - 16 * mm)
    canvas.setFont(_pdf_report_font_name(), 7.5)
    canvas.setFillColor(colors.HexColor("#466274"))
    canvas.drawString(20 * mm, height - 12 * mm, "KBot AIOps  ·  正式诊断报告")
    canvas.drawRightString(width - 20 * mm, 12 * mm, f"第 {document.page} 页")
    canvas.setStrokeColor(colors.HexColor("#CAD4DB"))
    canvas.line(20 * mm, 16 * mm, width - 20 * mm, 16 * mm)
    canvas.restoreState()


def _pdf_trend_chart(metric: dict[str, Any]) -> Drawing:
    """用七日基线指数绘制可缩放、可打印的小倍数趋势。"""
    width = 31 * mm
    height = 11 * mm
    drawing = Drawing(width, height)
    drawing.add(Line(0, height / 2, width, height / 2, strokeColor=colors.HexColor("#B7C1C8"), strokeWidth=0.5))
    points = [
        item for item in metric.get("trend_points") or ()
        if isinstance(item, dict) and _number(item.get("baseline_index")) is not None
    ]
    if len(points) < 2:
        return drawing
    values = [float(item["baseline_index"]) for item in points]
    low = min(min(values), 90.0)
    high = max(max(values), 110.0)
    span = max(high - low, 1.0)
    coordinates = []
    for index, value in enumerate(values):
        x = width * index / (len(values) - 1)
        y = 1.5 * mm + (height - 3 * mm) * (value - low) / span
        coordinates.extend((x, y))
    drawing.add(PolyLine(
        coordinates,
        strokeColor=colors.HexColor("#244C73"),
        strokeWidth=1.2,
    ))
    drawing.add(Circle(
        coordinates[-2], coordinates[-1], 1.2 * mm,
        fillColor=colors.HexColor("#244C73"), strokeColor=None,
    ))
    return drawing


def _daily_pdf_story(
    *,
    presentation: dict[str, Any],
    font_name: str,
    body: ParagraphStyle,
    section: ParagraphStyle,
    cover_title: ParagraphStyle,
) -> list[Any]:
    """按正式日常巡检版式生成表格和图形化 PDF story。"""
    daily = dict(presentation.get("daily_inspection") or {})
    header = dict(daily.get("header") or {})
    conclusion = dict(daily.get("conclusion") or {})
    story: list[Any] = [
        Paragraph(_pdf_paragraph(header.get("title") or presentation.get("title")), cover_title),
        Paragraph("数据库日常巡检 · 正式运维报告", body),
        Spacer(1, 5 * mm),
    ]
    metadata = [
        ["报告时间", header.get("report_time") or "未记录", "报告编号", header.get("report_number") or "未记录"],
        ["观测范围", f"{header.get('observation_start') or '未记录'} 至 {header.get('observation_end') or '未记录'}", "报告版本", f"v{header.get('report_version') or 1}"],
        ["巡检目标", header.get("target_name") or "未记录", "运行环境", header.get("environment") or "未记录"],
        ["数据库", f"{header.get('database_type') or '未记录'} {header.get('database_version') or ''}".strip(), "责任域", header.get("responsibility_domain") or "数据库运维"],
    ]
    metadata_table = Table(
        [[Paragraph(_pdf_paragraph(cell), body) for cell in row] for row in metadata],
        colWidths=(23 * mm, 61 * mm, 23 * mm, 63 * mm),
    )
    metadata_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF1F2")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EEF1F2")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#52616A")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#52616A")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C6CDD1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend((metadata_table, Spacer(1, 7 * mm)))
    story.append(Paragraph("报告结论", section))
    story.append(Paragraph(_pdf_paragraph(conclusion.get("text") or "未形成结论"), body))
    ledger = [
        ["整体健康", conclusion.get("health_status_display") or "未判定"],
        ["数据库检查覆盖率", f"{conclusion.get('evidence_coverage_percent') if conclusion.get('evidence_coverage_percent') is not None else '未记录'}%"],
        ["监控采样覆盖率", f"{conclusion.get('monitoring_coverage_percent') if conclusion.get('monitoring_coverage_percent') is not None else '未记录'}%"],
        ["需关注 / 数据缺口", f"{conclusion.get('attention_count') or 0} / {conclusion.get('gap_count') or 0}"],
    ]
    ledger_table = Table(
        [[Paragraph(_pdf_paragraph(cell), body) for cell in row] for row in ledger],
        colWidths=(48 * mm, 122 * mm),
        hAlign="LEFT",
    )
    ledger_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9BA8AF")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D3D9DC")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F8F7")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend((ledger_table, Spacer(1, 6 * mm)))

    metrics = list(daily.get("baseline_metrics") or ())
    story.append(Paragraph("关键指标与过去 7 日基线偏离", section))
    metric_rows: list[list[Any]] = [[
        "指标", "范围", "当前值", "七日均值", "偏离", "覆盖率", "七日趋势",
    ]]
    for metric in metrics:
        deviation = (
            f"{float(metric['deviation_percent']):+.2f}%"
            if metric.get("deviation_percent") is not None else "基线不足"
        )
        metric_rows.append([
            Paragraph(_pdf_paragraph(metric.get("display_name") or "指标"), body),
            Paragraph(_pdf_paragraph(metric.get("dimension_label") or "全部"), body),
            Paragraph(_pdf_paragraph(metric.get("current_display") or "无有效值"), body),
            Paragraph(_pdf_paragraph(metric.get("baseline_display") or "基线不足"), body),
            Paragraph(_pdf_paragraph(deviation), body),
            Paragraph(_pdf_paragraph(
                f"{metric.get('coverage_percent')}%" if metric.get("coverage_percent") is not None else "未记录"
            ), body),
            _pdf_trend_chart(metric),
        ])
    metric_table = Table(
        metric_rows,
        repeatRows=1,
        colWidths=(29 * mm, 27 * mm, 25 * mm, 25 * mm, 19 * mm, 17 * mm, 31 * mm),
    )
    metric_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECEE")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#415965")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#D0D5D8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend((metric_table, Spacer(1, 5 * mm)))

    capacity = list(daily.get("capacity_rows") or ())
    if capacity:
        story.append(Paragraph("七日趋势与容量明细", section))
        capacity_rows: list[list[Any]] = [["容量指标", "范围", "当前值", "七日均值", "偏离"]]
        for metric in capacity:
            capacity_rows.append([
                metric.get("display_name") or "容量指标",
                metric.get("dimension_label") or "全部",
                metric.get("current_display") or "无有效值",
                metric.get("baseline_display") or "基线不足",
                (
                    f"{float(metric['deviation_percent']):+.2f}%"
                    if metric.get("deviation_percent") is not None else "基线不足"
                ),
            ])
        capacity_table = Table(
            [[Paragraph(_pdf_paragraph(cell), body) for cell in row] for row in capacity_rows],
            repeatRows=1,
            colWidths=(35 * mm, 46 * mm, 31 * mm, 31 * mm, 27 * mm),
        )
        capacity_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECEE")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D0D4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend((capacity_table, Spacer(1, 5 * mm)))

    for title, key, empty in (
        ("发现问题", "findings", "未发现需要单列的问题。"),
        ("处置建议", "recommendations", "当前没有新增处置建议。"),
        ("数据缺口", "gaps", "未发现额外的数据缺口。"),
    ):
        story.append(Paragraph(title, section))
        items = list(daily.get(key) or ())
        if not items:
            items = [empty]
        for item in items:
            if isinstance(item, dict):
                value = f"{item.get('finding_id', '')} {item.get('title', '')}：{item.get('summary', '')}".strip()
            else:
                value = str(item)
            story.append(Paragraph(_pdf_paragraph(value), body, bulletText="•"))
    boundary = dict(daily.get("evidence_boundary") or {})
    story.append(Paragraph("证据边界", section))
    story.append(Paragraph(_pdf_paragraph(boundary.get("methodology") or "以冻结证据为准。"), body))
    for item in boundary.get("sources") or ():
        story.append(Paragraph(_pdf_paragraph(item), body, bulletText="•"))
    return story


def _monthly_pdf_story(
    *, presentation: dict[str, Any], font_name: str,
    body: ParagraphStyle, section: ParagraphStyle,
    cover_title: ParagraphStyle,
) -> list[Any]:
    """按日报出版物骨架输出月度统计、环比与问题台账。"""
    monthly = dict(presentation.get("monthly_inspection") or {})
    header = dict(monthly.get("header") or {})
    conclusion = dict(monthly.get("conclusion") or {})
    comparison = dict(monthly.get("period_comparison") or {})
    coverage = dict(monthly.get("coverage") or {})
    story: list[Any] = [
        Paragraph(_pdf_paragraph(header.get("title") or "数据库月度巡检报告"), cover_title),
        Paragraph("数据库月度巡检 · 正式运维报告", body),
        Spacer(1, 5 * mm),
    ]
    metadata = [
        ["报告时间", header.get("report_time") or "未记录", "报告编号", header.get("report_number") or "未记录"],
        ["统计周期", f"{header.get('period_start') or '未记录'} 至 {header.get('period_end') or '未记录'}", "报告版本", f"v{header.get('report_version') or 1}"],
        ["运维范围", header.get("scope_name") or "未记录", "范围类型", header.get("scope_type") or "TARGET"],
        ["数据库类型", "、".join(header.get("database_types") or ()) or "未记录", "责任域", header.get("responsibility_domain") or "数据库月度运维"],
    ]
    metadata_table = Table(
        [[Paragraph(_pdf_paragraph(cell), body) for cell in row] for row in metadata],
        colWidths=(23 * mm, 61 * mm, 23 * mm, 63 * mm),
    )
    metadata_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF1F2")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EEF1F2")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C6CDD1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend((metadata_table, Spacer(1, 7 * mm)))
    story.append(Paragraph("月度结论", section))
    story.append(Paragraph(_pdf_paragraph(conclusion.get("text") or "未形成结论"), body))
    ledger = [
        ["完成巡检", str(coverage.get("completed_run_count") or 0)],
        ["指标日覆盖率", f"{coverage.get('metric_day_coverage_percent') if coverage.get('metric_day_coverage_percent') is not None else '未记录'}%"],
        ["问题总数", str(conclusion.get("issue_count") or 0)],
        ["上月对比", comparison.get("status_display") or "无上月报告"],
    ]
    ledger_table = Table(
        [[Paragraph(_pdf_paragraph(cell), body) for cell in row] for row in ledger],
        colWidths=(48 * mm, 122 * mm),
    )
    ledger_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9BA8AF")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D3D9DC")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F8F7")),
    ]))
    story.extend((ledger_table, Spacer(1, 6 * mm)))

    story.append(Paragraph("上月对比", section))
    if comparison.get("status") == "AVAILABLE":
        story.append(Paragraph(_pdf_paragraph(
            f"新增问题 {len(comparison.get('new_issues') or ())} 项；"
            f"持续问题 {len(comparison.get('persistent_issues') or ())} 项；"
            f"已解决问题 {len(comparison.get('resolved_issues') or ())} 项。"
        ), body))
    else:
        story.append(Paragraph(_pdf_paragraph(comparison.get("status_display") or "无可比较的上月报告。"), body))

    story.append(Paragraph("月度关键指标", section))
    rows: list[list[Any]] = [["指标", "范围", "月均值", "P95", "最小 / 最大", "月内变化", "覆盖率"]]
    for metric in monthly.get("metrics") or ():
        change = metric.get("change_percent")
        rows.append([
            metric.get("display_name") or "指标",
            metric.get("dimension_label") or "全部",
            metric.get("monthly_average_display") or "无有效值",
            metric.get("p95_display") or "无有效值",
            f"{metric.get('minimum_display') or '-'} / {metric.get('maximum_display') or '-'}",
            f"{float(change):+.2f}%" if change is not None else "不可计算",
            f"{metric.get('coverage_percent')}%" if metric.get("coverage_percent") is not None else "未记录",
        ])
    metric_table = Table(
        [[Paragraph(_pdf_paragraph(cell), body) for cell in row] for row in rows],
        repeatRows=1, colWidths=(27 * mm, 27 * mm, 25 * mm, 22 * mm, 32 * mm, 20 * mm, 19 * mm),
    )
    metric_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECEE")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D0D4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend((metric_table, Spacer(1, 5 * mm)))

    for title, key, empty in (
        ("问题台账", "issues", "本月未形成需跟踪的问题。"),
        ("处置建议与下月重点", "recommendations", "当前没有新增处置建议。"),
        ("数据缺口", "gaps", "未发现额外的数据缺口。"),
    ):
        story.append(Paragraph(title, section))
        items = list(monthly.get(key) or ()) or [empty]
        for item in items:
            if isinstance(item, dict):
                value = (
                    f"{item.get('title') or item.get('finding_id') or '问题'}："
                    f"{item.get('summary') or ''}（影响 {item.get('affected_days') or 0} 天，"
                    f"出现 {item.get('occurrence_count') or 0} 次）"
                )
            else:
                value = str(item)
            story.append(Paragraph(_pdf_paragraph(value), body, bulletText="•"))
    story.append(Paragraph("扩展接口状态", section))
    for extension in monthly.get("extensions") or ():
        story.append(Paragraph(_pdf_paragraph(
            f"{extension.get('extension_key') or '扩展'}：{extension.get('status') or 'RESERVED'}"
        ), body, bulletText="•"))
    boundary = dict(monthly.get("evidence_boundary") or {})
    story.append(Paragraph("证据边界", section))
    story.append(Paragraph(_pdf_paragraph(boundary.get("methodology") or "以冻结证据为准。"), body))
    return story


def render_pdf(presentation: dict[str, Any]) -> bytes:
    """以标准 PDF 生成器输出可显示、复制和搜索的中文报告。"""
    font_name = _pdf_report_font_name()
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=25 * mm, bottomMargin=23 * mm,
        title=str(presentation.get("title") or "AIOps 正式报告"),
        author="KBot AIOps",
    )
    body = ParagraphStyle(
        "报告正文", fontName=font_name, fontSize=9.5, leading=16,
        textColor=colors.HexColor("#263238"), wordWrap="CJK", spaceAfter=4,
    )
    section = ParagraphStyle(
        "章节标题", parent=body, fontSize=14, leading=20,
        textColor=colors.HexColor("#163E59"), spaceBefore=10, spaceAfter=8,
    )
    cover_title = ParagraphStyle(
        "封面标题", parent=body, fontSize=24, leading=32,
        textColor=colors.HexColor("#163E59"), spaceAfter=7,
    )
    cover_subtitle = ParagraphStyle(
        "封面副标题", parent=body, fontSize=10, leading=16,
        textColor=colors.HexColor("#557080"), spaceAfter=20,
    )
    if presentation.get("daily_inspection"):
        document.build(
            _daily_pdf_story(
                presentation=presentation,
                font_name=font_name,
                body=body,
                section=section,
                cover_title=cover_title,
            ),
            onFirstPage=_report_page_chrome,
            onLaterPages=_report_page_chrome,
        )
        return buffer.getvalue()
    if presentation.get("monthly_inspection"):
        document.build(
            _monthly_pdf_story(
                presentation=presentation,
                font_name=font_name,
                body=body,
                section=section,
                cover_title=cover_title,
            ),
            onFirstPage=_report_page_chrome,
            onLaterPages=_report_page_chrome,
        )
        return buffer.getvalue()
    story = [Spacer(1, 38 * mm)]
    story.append(Paragraph(_pdf_paragraph(presentation.get("title") or "AIOps 正式报告"), cover_title))
    template = dict(presentation.get("template") or {})
    story.append(Paragraph(_pdf_paragraph(template.get("display_name") or "系统诊断报告"), cover_subtitle))
    report = dict(presentation.get("report") or {})
    metadata = [
        ["报告状态", _pdf_paragraph(
            presentation.get("status_display")
            or _report_display_text(presentation.get("status") or "UNKNOWN")
        )],
        ["报告周期", _pdf_paragraph(f"{report.get('period_start') or '未提供'} 至 {report.get('period_end') or '未提供'}")],
        ["报告模板", _pdf_paragraph(template.get("template_ref") or "未提供")],
        ["模板版本", _pdf_paragraph(template.get("version") or "未提供")],
    ]
    info_table = Table(metadata, colWidths=(32 * mm, 128 * mm), hAlign="LEFT")
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 15),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#466274")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#1E2D36")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F3")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CAD4DB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend((info_table, Spacer(1, 16 * mm)))
    briefing = dict(presentation.get("leadership_briefing") or {})
    if briefing:
        story.append(Paragraph("领导简报", section))
        risk_display = briefing.get(
            "risk_level_display"
        ) or _report_display_text(briefing.get("risk_level") or "LOW")
        story.append(Paragraph(
            _pdf_paragraph(f"风险等级：{risk_display}"),
            body,
        ))
        for label, key in (
            ("影响", "business_impact"),
            ("风险", "risks"),
            ("建议", "recommendations"),
        ):
            story.append(Paragraph(_pdf_paragraph(label), body))
            for item in briefing.get(key) or ():
                story.append(Paragraph(_pdf_paragraph(item), body, bulletText="•"))
        story.append(Spacer(1, 8 * mm))
    summary = next((item for item in presentation.get("sections") or () if item.get("kind") == "EXECUTIVE_SUMMARY"), None)
    if summary:
        story.append(Paragraph("执行摘要", section))
        for item in summary.get("items") or ():
            story.append(Paragraph(_pdf_paragraph(item), body, bulletText="•"))
    story.append(PageBreak())
    story.append(Paragraph("详细诊断", cover_title))
    for item in presentation.get("sections") or ():
        kind = str(item.get("kind") or "章节")
        if kind == "EXECUTIVE_SUMMARY":
            continue
        if kind == "EVIDENCE_APPENDIX":
            story.append(PageBreak())
        label = item.get("display_name") or display_name(kind)
        story.append(Paragraph(_pdf_paragraph(label), section))
        for detail in item.get("items") or ():
            if str(detail).startswith("【") and str(detail).endswith("】"):
                story.append(Paragraph(_pdf_paragraph(str(detail)[1:-1]), section))
            else:
                story.append(Paragraph(_pdf_paragraph(detail), body, bulletText="•"))
    document.build(story, onFirstPage=_report_page_chrome, onLaterPages=_report_page_chrome)
    return buffer.getvalue()
