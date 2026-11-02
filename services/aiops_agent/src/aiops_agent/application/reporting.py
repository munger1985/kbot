"""AIOps 正式报告的模板解析、展示投影与无状态 PDF 渲染。"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from math import isfinite
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
from aiops_agent.application.inspections.monthly import monthly_business_review


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
    *, payload: dict[str, Any], report_context: dict[str, Any],
) -> dict[str, Any]:
    """正式日报以冻结业务判定为主，技术质量只进入附录。"""
    scope = dict(payload.get("scope") or {})
    target = dict(scope.get("target") or {})
    assessment = dict(scope.get("inspection_assessment") or {})
    facts = [item for item in payload.get("facts") or () if isinstance(item, dict)]
    metric_results = {
        (str(item.get("metric_code") or ""), str(item.get("dimensions") or "-")): item
        for item in facts if item.get("kind") == "inspection_metric_check"
    }
    metrics = []
    for fact in facts:
        if fact.get("kind") != "inspection_monitoring_metric":
            continue
        metric = _daily_metric_projection(fact)
        checked = metric_results.get((str(fact.get("metric_code") or ""), str(fact.get("dimensions") or "-")), {})
        metric.update({
            "average_display": format_metric_value(_number(fact.get("average")), str(fact.get("unit") or "")),
            "maximum_display": format_metric_value(_number(fact.get("maximum")), str(fact.get("unit") or "")),
            "status_display": "异常" if any(
                item.get("severity") in {"HIGH", "CRITICAL"} for item in checked.get("findings") or ()
            ) else {"CLEAR": "正常", "FINDING": "需关注", "MISSING": "未完成", "PARTIAL": "未完成"}.get(
                str(checked.get("check_status") or ""), "仅观测"
            ),
            "basis": str(checked.get("basis") or "辅助观测，不以相对增长自动判定异常。"),
        })
        metrics.append(metric)
    metrics.sort(key=lambda item: (
        {"异常": 0, "需关注": 1, "未完成": 2, "正常": 3}.get(item["status_display"], 4),
        item["display_name"], item["dimension_label"],
    ))
    identity = next((item for item in facts if item.get("kind") == "inspection_identity"), {})
    identity_rows = list(identity.get("observations") or ())
    identity_row = identity_rows[0] if identity_rows and isinstance(identity_rows[0], dict) else {}
    health = str(assessment.get("health_status") or scope.get("health_status") or "INCOMPLETE")
    label = str(assessment.get("health_status_display") or {
        "NORMAL": "正常", "ATTENTION": "需关注", "ABNORMAL": "异常",
    }.get(health, "检查未完成"))
    report_id = str(report_context.get("report_id") or "")
    report_date = str(payload.get("period_end") or "")[:10].replace("-", "")
    number = f"AIOPS-DI-{report_date}-{report_id[-8:].upper()}" if report_id else str(payload.get("report_key") or "")
    gaps = _report_gap_items(list(payload.get("gaps") or ()))
    return {
        "schema_version": "DAILY_INSPECTION_PRESENTATION.v1",
        "header": {
            "title": str(payload.get("title") or "数据库日常巡检报告"),
            "report_time": report_context.get("created_at") or payload.get("period_end"),
            "observation_start": payload.get("period_start"),
            "observation_end": payload.get("period_end"),
            "timezone": scope.get("timezone") or "UTC",
            "business_system_name": scope.get("business_system_name") or "",
            "target_name": _target_name_for_presentation(
                stored_name=target.get("display_name"),
                target_id=target.get("target_id") or payload.get("target_id"),
                report_context=report_context,
            ),
            "target_id": str(target.get("target_id") or payload.get("target_id") or ""),
            "environment": _report_display_text(target.get("environment") or "未记录"),
            "database_type": str(target.get("database_type") or scope.get("database_type") or "未记录"),
            "database_version": str(target.get("version_code") or identity_row.get("version") or "未记录"),
            "report_version": int(report_context.get("report_version") or 1),
            "report_number": number, "responsibility_domain": "数据库运维",
        },
        "conclusion": {
            "text": str(assessment.get("text") or payload.get("summary") or "本次未形成完整检查结论。"),
            "health_status": health, "health_status_display": label,
            "completion_display": assessment.get("completion_display") or "未记录",
            "required_check_count": assessment.get("required_check_count"),
            "completed_required_count": assessment.get("completed_required_count"),
            "attention_count": len(assessment.get("findings") or ()),
            "incomplete_count": len(assessment.get("limitations") or ()),
        },
        "checks": list(assessment.get("checks") or ()),
        "metrics": metrics,
        "capacity_rows": [item for item in metrics if item.get("semantic_role") in _CAPACITY_METRIC_ROLES],
        "findings": list(assessment.get("findings") or ()),
        "limitations": list(assessment.get("limitations") or ()),
        "recommendations": list(assessment.get("recommendations") or ()),
        "gaps": gaps,
        "evidence_boundary": {
            "methodology": (
                "数据库检查按冻结规则判断；当前快照只说明检查时状态，不证明全天未发生异常。"
                "监控按冻结 Profile 阈值判读最新有效样本，单点超限不等于持续故障。"
                "采样质量和覆盖率不代替数据库健康结论。"
            ),
            "sources": [
                f"证据编号：{item.get('artifact_id') or '未记录'}\n内容指纹（Hash）：{item.get('content_hash') or '未提供哈希'}"
                for item in payload.get("evidence_refs") or () if isinstance(item, dict)
            ],
            "quality": {
                key: scope.get(key) for key in (
                    "evidence_coverage_percent", "tool_execution_coverage_percent",
                    "monitoring_sample_coverage_percent", "evidence_gap_count",
                )
            },
        },
    }


_CAPACITY_METRIC_ROLES = {
    "STORAGE_UTILIZATION", "STORAGE_USED", "STORAGE_FREE", "STORAGE_MAX",
    "DATABASE_SIZE", "FILESYSTEM_UTILIZATION",
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
    metric_limitations: list[str] = []

    for raw_fact in facts:
        if not isinstance(raw_fact, dict):
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
                "samples": {},
                "conflicts": set(),
            })
            sample_points = raw_fact.get("sample_points") or ()
            if sample_points:
                for point in sample_points:
                    if not isinstance(point, dict):
                        continue
                    value = _number(point.get("value"))
                    try:
                        moment = datetime.fromisoformat(str(point.get("observed_at") or "").replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if moment.tzinfo is None or value is None or not isfinite(value):
                        continue
                    if not period_start <= moment < period_end:
                        continue
                    stamp = moment.astimezone(UTC)
                    previous = group["samples"].get(stamp)
                    if previous is not None and previous != value:
                        group["conflicts"].add(stamp)
                    group["samples"][stamp] = value
                continue
            if raw_fact.get("daily_points") and raw_fact.get("daily_points_timezone") != timezone:
                metric_limitations.append(
                    f'{source_target_name or "数据库"} · {group["display_name"]}：仅有其他时区或时区未记录的日聚合，未纳入本月指标统计。'
                )
                continue
            for point in raw_fact.get("daily_points") or ():
                if not isinstance(point, dict):
                    continue
                date_text = str(point.get("date") or "")[:10]
                value = _number(point.get("value"))
                try:
                    point_date = datetime.fromisoformat(date_text).date()
                except ValueError:
                    continue
                if value is None or not isfinite(value) or not start_date <= point_date < end_date:
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
    metrics: list[dict[str, Any]] = []
    for group in metric_groups.values():
        if group["samples"]:
            days: dict[str, list[float]] = {}
            for stamp, value in sorted(group["samples"].items()):
                if stamp not in group["conflicts"]:
                    days.setdefault(stamp.astimezone(report_timezone).date().isoformat(), []).append(value)
            points = [
                {"date": date, "value": sum(values) / len(values),
                 "sample_count": len(values), "minimum": min(values), "maximum": max(values)}
                for date, values in sorted(days.items())
            ]
        else:
            points = [item[1] for _, item in sorted(group["days"].items())]
        if group["conflicts"]:
            metric_limitations.append(
                f'{group["target_name"] or "数据库"} · {group["display_name"]}：存在同刻不同值的冲突采样，已排除，不据此推导峰值。'
            )
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
            **{key: value for key, value in group.items() if key not in {"days", "samples", "conflicts"}},
            "statistic_basis": "DAILY_AGGREGATES",
            "statistic_basis_display": "日均值分布，非原始采样分布",
            "monthly_average": round(average, 6),
            "monthly_average_display": ("状态日均 " if group["value_kind"] == "STATE" else "") + format_metric_value(average, unit),
            "p95": round(_percentile_95(values) or 0.0, 6),
            "p95_display": format_metric_value(round(_percentile_95(values) or 0.0, 6), unit),
            "minimum": min(values),
            "minimum_display": "日均最小 " + format_metric_value(min(values), unit),
            "maximum": max(values),
            "maximum_display": "日均最大 " + format_metric_value(max(values), unit),
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
    review = monthly_business_review(
        facts=facts, period_start=period_start, period_end=period_end, timezone=timezone,
    )
    effective_metric_days = len({
        point["date"] for metric in metrics for point in metric["daily_points"]
    })
    return {
        "schema_version": "MONTHLY_INSPECTION_SUMMARY.v1",
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "timezone": timezone,
        "metrics": metrics,
        **review,
        "metric_limitations": list(dict.fromkeys(metric_limitations)),
        "limitations": list(dict.fromkeys((*review["limitations"], *metric_limitations))),
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
        if not before or (
            metric.get("statistic_basis") != before.get("statistic_basis")
            or metric.get("unit") != before.get("unit")
        ):
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
        "status_display": "已完成上月环比" if metric_changes else "问题台账已对照；无同口径可比指标",
        "metric_comparison_status": "AVAILABLE" if metric_changes else "INCOMPARABLE",
        "metric_changes": metric_changes,
        "new_issues": [current_issues[key] for key in sorted(current_issues.keys() - previous_issues.keys())],
        "persistent_issues": [current_issues[key] for key in sorted(current_issues.keys() & previous_issues.keys())],
        "resolved_issues": [
            item for item in current_issues.values()
            if item.get("lifecycle") == "VERIFIED_RESOLVED"
        ],
        "pending_verification_issues": [
            {**previous_issues[key], "lifecycle": "PENDING_VERIFICATION", "lifecycle_display": "待验证"}
            for key in sorted(previous_issues.keys() - current_issues.keys())
            if previous_issues[key].get("lifecycle") != "VERIFIED_RESOLVED"
        ],
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
    conclusion = dict(summary.get("conclusion") or {})
    if not conclusion:
        conclusion = {"text": "本月未形成完整基础检查结论。", "health_status_display": "检查未完成"}

    conclusion["issue_count"] = len(issues)
    return {
        "schema_version": "MONTHLY_INSPECTION_PRESENTATION.v1",
        "header": {
            "title": str(payload.get("title") or "数据库月度巡检报告"),
            "report_time": report_context.get("created_at") or payload.get("period_end"),
            "period_start": payload.get("period_start"),
            "period_end": payload.get("period_end"),
            "timezone": scope.get("timezone") or "UTC",
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
        "conclusion": conclusion,
        "period_comparison": comparison,
        "coverage": coverage,
        "metrics": metrics,
        "capacity_rows": [item for item in metrics if item.get("is_capacity")],
        "issues": issues,
        "target_results": list(summary.get("target_results") or ()),
        "limitations": list(summary.get("limitations") or ()),
        "database_checks": dict(summary.get("database_checks") or {}),
        "recommendations": [
            _report_display_text(item) for item in summary.get("recommendations") or ()
        ],
        "gaps": _report_gap_items(list(payload.get("gaps") or ())),
        "evidence_boundary": {
            "methodology": (
                "指标按数据库、指标、维度及单位隔离，逐日去重；日均值的平均和百分位不等于原始采样统计。"
                "快照状态日均值不代表精确月度可用率；未再发现问题不等于已验证恢复。"
                "生成日瞬时值不进入月度结论。"
            ),
            "sources": [
                f"证据编号：{item.get('artifact_id') or '未记录'}\n内容指纹（Hash）：{item.get('content_hash') or '未提供哈希'}"
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
            **{key: scope[key] for key in (
                "target", "scope_type", "scope_name", "target_count", "database_types",
                "monthly_summary", "previous_month_summary", "previous_month_comparison",
                "business_system_id", "business_system_name", "timezone", "health_status",
                "inspection_assessment", "report_policy", "member_targets",
            ) if key in scope},
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
    report_label = "正式巡检报告" if getattr(document, "inspection_report", False) else "正式诊断报告"
    canvas.drawString(20 * mm, height - 12 * mm, f"KBot AIOps  ·  {report_label}")
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
    *, presentation: dict[str, Any], font_name: str, body: ParagraphStyle,
    section: ParagraphStyle, cover_title: ParagraphStyle,
) -> list[Any]:
    """日报 PDF 与网页复用同一业务投影。"""
    from aiops_agent.application.inspection_pdf import inspection_pdf_story

    return inspection_pdf_story(
        data=dict(presentation.get("daily_inspection") or {}), monthly=False,
        font_name=font_name, body=body, section=section, cover_title=cover_title,
    )


def _monthly_pdf_story(
    *, presentation: dict[str, Any], font_name: str, body: ParagraphStyle,
    section: ParagraphStyle, cover_title: ParagraphStyle,
) -> list[Any]:
    """月报 PDF 与网页复用同一业务投影。"""
    from aiops_agent.application.inspection_pdf import inspection_pdf_story

    return inspection_pdf_story(
        data=dict(presentation.get("monthly_inspection") or {}), monthly=True,
        font_name=font_name, body=body, section=section, cover_title=cover_title,
    )



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
    document.inspection_report = bool(presentation.get("daily_inspection") or presentation.get("monthly_inspection"))
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
