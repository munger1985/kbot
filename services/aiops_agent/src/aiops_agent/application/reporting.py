"""AIOps 正式报告的模板解析、展示投影与无状态 PDF 渲染。"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from importlib.resources import as_file, files
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from aiops_agent.application.errors import validation_failed
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

_REPORT_DISPLAY_NAMES = {
    "EXECUTIVE_SUMMARY": "执行摘要",
    "SCOPE": "报告范围",
    "ALERT_TIMELINE": "告警时间线",
    "INSPECTION_COVERAGE": "巡检覆盖情况",
    "RISK_OVERVIEW": "风险概览",
    "TREND": "趋势分析",
    "FINDINGS": "核验发现",
    "ROOT_CAUSE": "根因分析",
    "RECOMMENDATIONS": "处置建议",
    "ACTIONS": "已执行动作",
    "EVIDENCE_BOUNDARY": "证据边界",
    "EVIDENCE_APPENDIX": "证据附录",
    "CONFIRMED": "已确认",
    "PROBABLE": "很可能",
    "POSSIBLE": "可能",
    "INCONCLUSIVE": "证据不足，无法定论",
    "GENERATING": "生成中",
    "READY": "已完成",
    "PARTIAL": "部分完成",
    "FAILED": "失败",
    "CRITICAL": "严重",
    "HIGH": "高",
    "MEDIUM": "中",
    "LOW": "低",
    "INFO": "提示",
    "RESOLVED": "已解决",
    "IMPROVED": "已改善",
    "UNCHANGED": "未改善",
    "DEGRADED": "已恶化",
    "MISSING_ASH": "缺少 ASH 历史会话数据",
    "MISSING_PRIMARY_RUN": "缺少主诊断运行记录",
    "MISSING_FINAL_RESULT": "缺少最终诊断结果",
    "UNREPORTABLE_FINAL_RESULT": "最终结果暂不支持生成报告",
    "MISSING_FINAL_ARTIFACT": "缺少最终报告产物",
    "USER_RESULT_UNAVAILABLE": "用户提供的结果不可用",
    "EVIDENCE_FACT_LIMIT_REACHED": "已达到证据事实数量上限",
    "VERIFICATION_EVIDENCE_MISSING": "缺少处置验证证据",
    "ACTION_VERIFIER_UNAVAILABLE": "动作验证器不可用",
    "EVIDENCE_NOT_COMPARABLE": "处理前后证据不可比较",
    "VERIFICATION_STATE_UNSUPPORTED": "当前验证状态不受支持",
    "SOURCE_AUTH_FAILED": "诊断源认证失败",
    "SOURCE_UNREACHABLE": "诊断源不可达",
    "DIAGNOSTIC_POLICY_DENIED": "诊断策略不允许数据库直连",
    "DB_DIRECT_NOT_CONFIGURED": "未配置数据库直连",
    "TARGET_INACTIVE": "Target 未启用",
    "DIAGNOSTIC_SECRET_MISSING": "缺少诊断凭据",
    "TARGET_ENDPOINT_MISSING": "缺少 Target 连接地址",
    "TARGET_CONNECTIVITY_UNAVAILABLE": "Target 当前不可连接",
    "VERSION_UNSUPPORTED": "数据库版本不受支持",
    "CAPABILITY_UNAVAILABLE": "所需诊断能力不可用",
    "INSPECTION_OBSERVATION_MISSING": "缺少巡检观测结果",
    "INSPECTION_TEMPLATE_STEPS_MISSING": "巡检模板缺少检查步骤",
    "METRIC_SOURCE_UNAVAILABLE": "指标源不可用",
    "ACTION_TEMPLATE_UNAVAILABLE": "动作模板不可用",
    "EXECUTION_SECRET_MISSING": "缺少执行凭据",
    "EXECUTION_UNAVAILABLE_ADVISORY_PROVIDED": "无法自动执行，已提供人工建议",
    "MUTATION_EXECUTION_UNAVAILABLE": "受控变更执行不可用",
    "OPERATIONS_KNOWLEDGE_UNAVAILABLE": "运维知识不可用",
    "OPS_DELEGATION_RESULT_NOT_READY": "委派结果尚未就绪",
    "POLICY_MISSING": "缺少执行策略",
    "SECRET_UNAVAILABLE": "凭据不可用",
    "VERIFIED_ACTION_PARAMETERS_UNAVAILABLE": "缺少已验证的动作参数",
    "EXTERNAL_SOURCE_NOT_SOURCE_VERIFIED": "外部数据源未通过来源验证",
    "DISCONTINUITY": "数据采样不连续",
    "METRIC_COUNTER_RESET": "指标计数器已重置",
    "SAMPLING_BOUNDARY": "采样边界限制",
    "RPO_NOT_CONFIGURED": "未配置业务 RPO",
    "RTO_NOT_CONFIGURED": "未配置业务 RTO",
    "RESTORE_NOT_DEMONSTRATED": "恢复能力尚未实证",
    "LATEST_DRILL_FAILED": "最近一次恢复演练失败",
    "DRILL_STALE": "最近合格成功演练已超过要求周期",
    "POLICY_CHANGED_SINCE_DRILL": "演练后恢复策略已变更",
    "RPO_MEASUREMENT_MISSING": "合格演练缺少实测 RPO",
    "RTO_MEASUREMENT_MISSING": "合格演练缺少实测 RTO",
    "RPO_TARGET_BREACHED": "实测 RPO 超出业务目标",
    "RTO_TARGET_BREACHED": "实测 RTO 超出业务目标",
    "BACKUP_SOURCE_NOT_VERIFIED": "必需备份来源尚未验证",
    "VERIFICATION_ADVERSE": "验证发现不利变化",
    "ACTION_EFFECT_VERIFIED": "已验证动作达到预期效果",
    "EXPECTED_DIRECT_EFFECT_NOT_OBSERVED": "未观测到预期直接效果",
    "TARGET_ABSENT": "目标对象已消失",
    "BLOCKING_ABSENT": "阻塞关系已消失",
}
_REPORT_CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{2,}")


def _report_display_text(value: object) -> str:
    """仅替换已登记的报告枚举，未知技术文本保持原样。"""
    return _REPORT_CODE_PATTERN.sub(
        lambda match: _REPORT_DISPLAY_NAMES.get(
            match.group(0), match.group(0)
        ),
        str(value),
    )


def _report_gap_item(value: object) -> str:
    """把证据缺口代码及其上下文转换为可读说明。"""
    if not isinstance(value, dict):
        return _report_display_text(value)
    label = _report_display_text(value.get("code") or "未知证据缺口")
    turn_no = value.get("turn_no")
    detail = str(value.get("detail") or "").strip()
    if turn_no is not None:
        label = f"第 {turn_no} 轮：{label}"
    if detail and detail not in label:
        label = f"{label}：{_report_display_text(detail)}"
    return label


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
        result.append(str(fact.get("summary") or fact.get("fact_summary") or fact))
    return result


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
                "EXECUTIVE_SUMMARY", "SCOPE", "FINDINGS",
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
        markdown = "\n\n".join(
            str(dict(block).get("payload", {}).get("markdown") or "")
            for block in payload.get("blocks", ())
            if str(dict(block).get("block_type")) == "MARKDOWN"
        ).strip()
        evidence = []
        for block in payload.get("blocks", ()):
            evidence.extend(dict(block).get("evidence_refs") or ())
        return {
            "status": "READY" if payload.get("status") == "COMPLETED" else "PARTIAL",
            "root_cause": {"effective_level": "INCONCLUSIVE"},
            "diagnosis_rationale": markdown or "Agent 未生成可用于汇报的文字结论。",
            "facts": (),
            "gaps": tuple(payload.get("evidence_gaps", ())),
            "solution": {},
            "evidence_refs": tuple(evidence),
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
            "diagnosis_rationale": str(payload.get("summary") or ""),
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
            body = [str(item) for item in list(scope.get("actions") or ())] or ["当前没有已记录的处置或验证动作。"]
        elif kind == "EVIDENCE_BOUNDARY":
            body = [_report_gap_item(item) for item in gaps] or [
                "未发现额外的数据缺口。"
            ]
        else:
            body = [
                f"{item.get('artifact_id', 'evidence')} · {item.get('content_hash', '未提供哈希')}"
                for item in list(payload.get("evidence_refs") or ())
            ] or ["未记录可公开的证据索引。"]
        if kind not in {"EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX"}:
            edited = overrides.get(kind)
            if isinstance(edited, (list, tuple)) and edited:
                body = [str(item) for item in edited]
        body = [_report_display_text(item) for item in body]
        section_data.append({
            "kind": kind,
            "display_name": _REPORT_DISPLAY_NAMES.get(kind, kind),
            "items": body,
            "human_edited": kind in overrides,
        })
    briefing = project_leadership_briefing(
        payload=payload, findings=findings,
    )
    return {
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
        label = item.get("display_name") or _REPORT_DISPLAY_NAMES.get(
            kind, kind
        )
        story.append(Paragraph(_pdf_paragraph(label), section))
        for detail in item.get("items") or ():
            if str(detail).startswith("【") and str(detail).endswith("】"):
                story.append(Paragraph(_pdf_paragraph(str(detail)[1:-1]), section))
            else:
                story.append(Paragraph(_pdf_paragraph(detail), body, bulletText="•"))
    document.build(story, onFirstPage=_report_page_chrome, onLaterPages=_report_page_chrome)
    return buffer.getvalue()
