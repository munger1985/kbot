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


def _system_template(
    *, key: str, name: str, source_kinds: tuple[str, ...],
    periods: tuple[str, ...], sections: tuple[str, ...],
) -> ReportTemplate:
    definition = {
        "schema_version": "REPORT_TEMPLATE.v1",
        "display_name": name,
        "applicable_source_kinds": list(source_kinds),
        "allowed_period_kinds": list(periods),
        "sections": [{"kind": item} for item in sections],
    }
    return ReportTemplate(
        template_ref=f"system:{key}", version="1", display_name=name,
        applicable_source_kinds=source_kinds, allowed_period_kinds=periods,
        sections=sections, definition=definition,
    )


SYSTEM_REPORT_TEMPLATES = {
    item.template_ref: item
    for item in (
        _system_template(
            key="diagnosis.standard", name="标准诊断报告",
            source_kinds=("CHAT", "ALERT"), periods=("AD_HOC",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "ALERT_TIMELINE",
                "ROOT_CAUSE", "FINDINGS", "RECOMMENDATIONS", "ACTIONS",
                "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="inspection.daily", name="日常巡检报告",
            source_kinds=("INSPECTION",), periods=("DAILY",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
                "RISK_OVERVIEW", "FINDINGS", "RECOMMENDATIONS",
                "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="inspection.monthly", name="月度巡检报告",
            source_kinds=("INSPECTION",), periods=("MONTHLY",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
                "RISK_OVERVIEW", "TREND", "FINDINGS", "ACTIONS",
                "RECOMMENDATIONS", "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="inspection.quarterly", name="季度巡检报告",
            source_kinds=("INSPECTION",), periods=("QUARTERLY",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
                "RISK_OVERVIEW", "TREND", "FINDINGS", "ACTIONS",
                "RECOMMENDATIONS", "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
        _system_template(
            key="inspection.annual", name="年度巡检报告",
            source_kinds=("INSPECTION",), periods=("ANNUAL",),
            sections=(
                "EXECUTIVE_SUMMARY", "SCOPE", "INSPECTION_COVERAGE",
                "RISK_OVERVIEW", "TREND", "FINDINGS", "ACTIONS",
                "RECOMMENDATIONS", "EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX",
            ),
        ),
    )
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
    if not periods or any(item not in {"AD_HOC", "DAILY", "MONTHLY", "QUARTERLY", "ANNUAL", "CUSTOM"} for item in periods):
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
        elif kind == "ALERT_TIMELINE":
            body = [str(scope.get("alert_summary") or "本报告未关联告警时间线")]
        elif kind == "INSPECTION_COVERAGE":
            body = [str(scope.get("inspection_coverage") or "本报告未包含巡检覆盖统计")]
        elif kind == "RISK_OVERVIEW":
            body = [f"根因评估等级：{root_grade}"]
        elif kind == "TREND":
            body = [str(item) for item in list(scope.get("trends") or ())] or ["当前报告范围内没有可复现的趋势数据。"]
        elif kind == "FINDINGS":
            body = [str(item.get("summary") or item.get("fact_summary") or item) for item in facts] or ["未记录已验证发现。"]
        elif kind == "ROOT_CAUSE":
            body = [f"根因评估等级：{root_grade}", str(scope.get("diagnosis_rationale") or "当前结论以证据边界章节为准。")]
        elif kind == "RECOMMENDATIONS":
            body = [str(item) for item in recommendations] or ["当前没有可执行建议。"]
        elif kind == "ACTIONS":
            body = [str(item) for item in list(scope.get("actions") or ())] or ["当前没有已记录的处置或验证动作。"]
        elif kind == "EVIDENCE_BOUNDARY":
            body = [str(item.get("code") or item) for item in gaps] or ["未发现额外的数据缺口。"]
        else:
            body = [
                f"{item.get('artifact_id', 'evidence')} · {item.get('content_hash', '未提供哈希')}"
                for item in list(payload.get("evidence_refs") or ())
            ] or ["未记录可公开的证据索引。"]
        if kind not in {"EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX"}:
            edited = overrides.get(kind)
            if isinstance(edited, (list, tuple)) and edited:
                body = [str(item) for item in edited]
        section_data.append({
            "kind": kind,
            "items": body,
            "human_edited": kind in overrides,
        })
    return {
        "schema_version": "REPORT_PRESENTATION.v1",
        "title": payload.get("title") or template.display_name,
        "status": payload.get("status"),
        "template": {**template_summary(template), "definition": template.definition},
        "report": payload,
        "sections": section_data,
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


_REPORT_SECTION_NAMES = {
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
}


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
        ["报告状态", _pdf_paragraph(presentation.get("status") or "UNKNOWN")],
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
        label = _REPORT_SECTION_NAMES.get(kind, kind)
        story.append(Paragraph(_pdf_paragraph(label), section))
        for detail in item.get("items") or ():
            story.append(Paragraph(_pdf_paragraph(detail), body, bulletText="•"))
    document.build(story, onFirstPage=_report_page_chrome, onLaterPages=_report_page_chrome)
    return buffer.getvalue()
