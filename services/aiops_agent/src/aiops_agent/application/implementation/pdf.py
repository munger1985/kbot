"""数据库实施 Runbook 的无状态 PDF 渲染器。"""

from __future__ import annotations

from html import escape
from io import BytesIO
import textwrap
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

from aiops_agent.application.reporting import (
    _pdf_report_font_name,
    _pdf_text,
)


def _items(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _paragraph(value: object) -> str:
    return escape(_pdf_text(str(value or ""))).replace("\n", "<br/>")


def _wrapped_code(value: object) -> str:
    lines: list[str] = []
    for source_line in str(value or "").strip().splitlines():
        wrapped = textwrap.wrap(
            source_line,
            width=104,
            replace_whitespace=False,
            drop_whitespace=False,
            subsequent_indent="  ",
        )
        lines.extend(wrapped or [""])
    return _pdf_text("\n".join(lines))


def _page_chrome(canvas: Canvas, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#315B82"))
    canvas.line(18 * mm, height - 15 * mm, width - 18 * mm, height - 15 * mm)
    canvas.setFont(_pdf_report_font_name(), 7.5)
    canvas.setFillColor(colors.HexColor("#536B7A"))
    canvas.drawString(18 * mm, height - 11 * mm, "KBot AIOps  ·  数据库实施操作文档")
    canvas.drawRightString(width - 18 * mm, 11 * mm, f"第 {document.page} 页")
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.restoreState()


def render_implementation_runbook_pdf(payload: dict[str, Any]) -> bytes:
    """直接渲染持久化 Block，兼容历史 Runbook schema。"""
    font_name = _pdf_report_font_name()
    buffer = BytesIO()
    title = str(payload.get("title") or "数据库实施操作文档")
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=23 * mm,
        bottomMargin=21 * mm,
        title=title,
        author="KBot AIOps",
    )
    body = ParagraphStyle(
        "实施正文",
        fontName=font_name,
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#27343D"),
        wordWrap="CJK",
        spaceAfter=4,
    )
    title_style = ParagraphStyle(
        "实施标题",
        parent=body,
        fontSize=21,
        leading=29,
        textColor=colors.HexColor("#173F5F"),
        spaceAfter=7,
    )
    phase_style = ParagraphStyle(
        "阶段标题",
        parent=body,
        fontSize=14,
        leading=20,
        textColor=colors.HexColor("#173F5F"),
        spaceBefore=10,
        spaceAfter=6,
    )
    step_style = ParagraphStyle(
        "步骤标题",
        parent=body,
        fontSize=11,
        leading=17,
        textColor=colors.HexColor("#275D82"),
        spaceBefore=7,
        spaceAfter=4,
    )
    code_style = ParagraphStyle(
        "命令",
        fontName=font_name,
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor("#1E2A30"),
        backColor=colors.HexColor("#F0F3F5"),
        borderColor=colors.HexColor("#C8D2D8"),
        borderWidth=0.4,
        borderPadding=7,
        leftIndent=2,
        rightIndent=2,
        spaceBefore=3,
        spaceAfter=7,
    )
    story: list[Any] = [Spacer(1, 16 * mm), Paragraph(_paragraph(title), title_style)]
    metadata = " · ".join(
        item
        for item in (
            str(payload.get("profile") or ""),
            str(payload.get("status") or ""),
            str(payload.get("schema_version") or ""),
        )
        if item
    )
    if metadata:
        story.append(Paragraph(_paragraph(metadata), body))
    policy = payload.get("execution_policy")
    if policy:
        story.append(Paragraph(_paragraph(policy), body))
    story.append(Spacer(1, 7 * mm))

    group_labels = (
        ("commands", "实施命令"),
        ("verification_commands", "验证命令"),
        ("rollback", "回退命令"),
    )
    phases = _items(payload.get("phases"))
    for phase_index, phase_value in enumerate(phases, start=1):
        phase = _mapping(phase_value)
        story.append(Paragraph(
            _paragraph(f"{phase_index}. {phase.get('title') or phase.get('phase_id') or '实施阶段'}"),
            phase_style,
        ))
        if phase.get("objective"):
            story.append(Paragraph(_paragraph(phase["objective"]), body))
        for step_index, step_value in enumerate(_items(phase.get("steps")), start=1):
            step = _mapping(step_value)
            status = str(step.get("applicability") or "")
            step_title = step.get("title") or step.get("step_id") or "操作步骤"
            story.append(Paragraph(
                _paragraph(f"{phase_index}.{step_index} {step_title} [{status}]"),
                step_style,
            ))
            if step.get("rationale"):
                story.append(Paragraph(_paragraph(step["rationale"]), body))
            required_inputs = _items(step.get("required_inputs"))
            if required_inputs:
                story.append(Paragraph(
                    _paragraph("所需输入：" + "、".join(map(str, required_inputs))),
                    body,
                ))
            for key, label in group_labels:
                commands = _items(step.get(key))
                if not commands:
                    continue
                story.append(Paragraph(_paragraph(label), body))
                for command_value in commands:
                    command = _mapping(command_value)
                    heading = " · ".join(
                        item
                        for item in (
                            str(command.get("command_type") or ""),
                            str(command.get("title") or ""),
                        )
                        if item
                    )
                    if heading:
                        story.append(Paragraph(_paragraph(heading), body))
                    story.append(Preformatted(
                        _wrapped_code(command.get("content")),
                        code_style,
                    ))
                    for note in _items(command.get("notes")):
                        story.append(Paragraph(_paragraph(note), body, bulletText="•"))
            for risk in _items(step.get("risks")):
                story.append(Paragraph(_paragraph(f"注意：{risk}"), body))

    story.append(PageBreak())
    story.append(Paragraph("附录：当前状态与输入", phase_style))
    for item_value in _items(payload.get("current_state")):
        item = _mapping(item_value)
        story.append(Paragraph(
            _paragraph(
                f"{item.get('label') or '-'}：{item.get('value') or '-'} "
                f"[{item.get('status') or 'UNKNOWN'}]"
            ),
            body,
            bulletText="•",
        ))
    for item_value in _items(payload.get("resolved_parameters")):
        item = _mapping(item_value)
        story.append(Paragraph(
            _paragraph(
                f"{item.get('label') or item.get('key') or '-'}："
                f"{item.get('value') or '-'}"
            ),
            body,
            bulletText="•",
        ))
    for item_value in _items(payload.get("required_inputs")):
        item = _mapping(item_value)
        story.append(Paragraph(
            _paragraph(
                f"待补齐 {item.get('label') or item.get('key') or '-'}："
                f"{item.get('description') or item.get('placeholder') or '-'}"
            ),
            body,
            bulletText="•",
        ))
    stop_conditions = _items(payload.get("stop_conditions"))
    if stop_conditions:
        story.append(Paragraph("停止条件", step_style))
        for item in stop_conditions:
            story.append(Paragraph(_paragraph(item), body, bulletText="•"))

    document.build(story, onFirstPage=_page_chrome, onLaterPages=_page_chrome)
    return buffer.getvalue()
