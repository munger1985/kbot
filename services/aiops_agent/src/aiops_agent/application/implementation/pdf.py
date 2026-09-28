"""数据库实施 Runbook 的无状态 PDF 渲染器。"""

from __future__ import annotations

from html import escape
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)
from reportlab.platypus.tableofcontents import TableOfContents

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
    """保留命令换行，仅替换 PDF 字体无法表达的控制字符。"""
    return "".join(
        character
        if character in {"\n", "\t"} or 0x20 <= ord(character) <= 0xFFFF
        else "?"
        for character in str(value or "").strip()
    )


def _formatted_code(
    value: object,
    command_type: object,
    *,
    width: int = 90,
) -> str:
    """按命令类型生成可见且仍可复制执行的 PDF 换行。"""
    command = _wrapped_code(value)
    is_shell = str(command_type or "").upper() == "SHELL"
    rendered: list[str] = []
    for source_line in command.splitlines() or [""]:
        remaining = source_line.rstrip()
        continuation_indent = ""
        while len(continuation_indent + remaining) > width:
            available = max(20, width - len(continuation_indent))
            split_at = remaining.rfind(" ", 0, available + 1)
            compact_split = False
            if split_at <= 0 and is_shell:
                split_at = max(
                    remaining.rfind(character, 0, available + 1) + 1
                    for character in ",;"
                )
                compact_split = split_at > 0
            if split_at <= 0:
                break
            segment = remaining[:split_at].rstrip()
            remaining = remaining[
                split_at if compact_split else split_at + 1:
            ].lstrip()
            continuation = (
                "\\" if is_shell and compact_split
                else (" \\" if is_shell else "")
            )
            rendered.append(
                f"{continuation_indent}{segment}{continuation}"
            )
            continuation_indent = "" if compact_split else "  "
        rendered.append(f"{continuation_indent}{remaining}")
    return "\n".join(rendered)


class _RunbookCodeBlock(Flowable):
    """把标题和命令正文绘制成可分页的单一代码块。"""

    header_height = 20
    code_font_size = 7
    code_leading = 10
    body_padding_top = 7
    body_padding_bottom = 7
    body_padding_horizontal = 9

    def __init__(
        self,
        *,
        title: str,
        code: str,
        text_font_name: str,
        continuation: bool = False,
    ) -> None:
        super().__init__()
        self.title = f"{title}（续）" if continuation else title
        self.lines = code.splitlines() or [""]
        self.text_font_name = text_font_name
        self.code_font_name = (
            "Courier" if code.isascii() else text_font_name
        )
        self.spaceBefore = 5
        self.spaceAfter = 7
        self._available_width = 0.0

    def _body_height(self, line_count: int | None = None) -> float:
        count = len(self.lines) if line_count is None else line_count
        return (
            self.body_padding_top
            + count * self.code_leading
            + self.body_padding_bottom
        )

    def wrap(
        self,
        available_width: float,
        available_height: float,
    ) -> tuple[float, float]:
        self._available_width = available_width
        return available_width, self.header_height + self._body_height()

    def split(self, available_width: float, available_height: float) -> list[Flowable]:
        self._available_width = available_width
        full_height = self.header_height + self._body_height()
        if full_height <= available_height:
            return [self]
        fixed_height = (
            self.header_height
            + self.body_padding_top
            + self.body_padding_bottom
        )
        line_capacity = int(
            (available_height - fixed_height) // self.code_leading
        )
        if line_capacity < 1:
            return []
        first = _RunbookCodeBlock(
            title=self.title,
            code="\n".join(self.lines[:line_capacity]),
            text_font_name=self.text_font_name,
        )
        first.spaceAfter = 0
        remainder = _RunbookCodeBlock(
            title=self.title.removesuffix("（续）"),
            code="\n".join(self.lines[line_capacity:]),
            text_font_name=self.text_font_name,
            continuation=True,
        )
        remainder.spaceBefore = 0
        return [first, remainder]

    def draw(self) -> None:
        canvas = self.canv
        width = self._available_width
        body_height = self._body_height()
        total_height = self.header_height + body_height

        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#F0F3F5"))
        canvas.setStrokeColor(colors.HexColor("#C8D2D8"))
        canvas.setLineWidth(0.5)
        canvas.rect(0, 0, width, body_height, fill=1, stroke=1)

        canvas.setFillColor(colors.HexColor("#2B3549"))
        canvas.setStrokeColor(colors.HexColor("#2B3549"))
        canvas.rect(
            0,
            body_height,
            width,
            self.header_height,
            fill=1,
            stroke=1,
        )
        canvas.setFillColor(colors.white)
        canvas.setFont(self.text_font_name, 8)
        canvas.drawString(
            self.body_padding_horizontal,
            total_height - 13,
            _pdf_text(self.title),
        )

        canvas.setFillColor(colors.HexColor("#1E2A30"))
        text = canvas.beginText()
        text.setTextOrigin(
            self.body_padding_horizontal,
            body_height - self.body_padding_top - self.code_font_size,
        )
        text.setFont(self.code_font_name, self.code_font_size)
        text.setLeading(self.code_leading)
        for line in self.lines:
            text.textLine(line.expandtabs(4))
        canvas.drawText(text)
        canvas.restoreState()


class _RunbookDocTemplate(SimpleDocTemplate):
    """为实施文档生成目录条目、书签和 PDF 大纲。"""

    def afterFlowable(self, flowable: object) -> None:
        entry = getattr(flowable, "_runbook_toc_entry", None)
        if not entry:
            return
        level, title, bookmark = entry
        self.canv.bookmarkPage(bookmark)
        self.canv.addOutlineEntry(title, bookmark, level=level, closed=False)
        self.notify("TOCEntry", (level, title, self.page, bookmark))


def _toc_heading(
    value: str,
    style: ParagraphStyle,
    *,
    level: int,
    bookmark: str,
) -> Paragraph:
    """创建同时进入正文、目录和 PDF 大纲的标题。"""
    paragraph = Paragraph(_paragraph(value), style)
    paragraph._runbook_toc_entry = (level, value, bookmark)
    return paragraph


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
    document = _RunbookDocTemplate(
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
        keepWithNext=True,
    )
    step_style = ParagraphStyle(
        "步骤标题",
        parent=body,
        fontSize=11,
        leading=17,
        textColor=colors.HexColor("#275D82"),
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True,
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
    story.extend((Spacer(1, 7 * mm), PageBreak()))

    story.append(Paragraph("目录", title_style))
    contents = TableOfContents()
    contents.dotsMinLevel = 0
    contents.levelStyles = (
        ParagraphStyle(
            "目录阶段",
            parent=body,
            fontSize=10,
            leading=16,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
        ),
        ParagraphStyle(
            "目录步骤",
            parent=body,
            fontSize=8.5,
            leading=13,
            leftIndent=12 * mm,
            firstLineIndent=0,
            textColor=colors.HexColor("#536B7A"),
        ),
    )
    story.extend((contents, PageBreak()))

    group_labels = (
        ("commands", "实施命令"),
        ("verification_commands", "验证命令"),
        ("rollback", "回退命令"),
    )
    phases = _items(payload.get("phases"))
    for phase_index, phase_value in enumerate(phases, start=1):
        phase = _mapping(phase_value)
        phase_title = (
            f"{phase_index}. "
            f"{phase.get('title') or phase.get('phase_id') or '实施阶段'}"
        )
        story.append(_toc_heading(
            phase_title,
            phase_style,
            level=0,
            bookmark=f"phase-{phase_index}",
        ))
        if phase.get("objective"):
            story.append(Paragraph(_paragraph(phase["objective"]), body))
        for step_index, step_value in enumerate(_items(phase.get("steps")), start=1):
            step = _mapping(step_value)
            status = str(step.get("applicability") or "")
            step_title = step.get("title") or step.get("step_id") or "操作步骤"
            numbered_step_title = (
                f"{phase_index}.{step_index} {step_title} [{status}]"
            )
            story.append(_toc_heading(
                numbered_step_title,
                step_style,
                level=1,
                bookmark=f"phase-{phase_index}-step-{step_index}",
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
                groups = ((label, commands, False),)
                if key == "commands":
                    executable = [
                        item for item in commands
                        if "MANUAL" not in {
                            str(_mapping(item).get("command_type") or "").upper(),
                            str(_mapping(item).get("executor") or "").upper(),
                        }
                    ]
                    manual = [
                        item for item in commands
                        if "MANUAL" in {
                            str(_mapping(item).get("command_type") or "").upper(),
                            str(_mapping(item).get("executor") or "").upper(),
                        }
                    ]
                    groups = (
                        (label, executable, False),
                        ("人工确认项", manual, True),
                    )
                for group_label, group_commands, manual_group in groups:
                    if not group_commands:
                        continue
                    story.append(Paragraph(_paragraph(group_label), body))
                    for command_value in group_commands:
                        command = _mapping(command_value)
                        heading = " · ".join(
                            item
                            for item in (
                                str(command.get("executor") or command.get("command_type") or ""),
                                str(command.get("title") or ""),
                            )
                            if item
                        )
                        if manual_group:
                            story.append(Paragraph(
                                _paragraph(command.get("title") or "人工确认"),
                                step_style,
                            ))
                            story.append(Paragraph(
                                _paragraph(command.get("content")),
                                body,
                            ))
                        else:
                            formatted_code = _formatted_code(
                                command.get("content"),
                                command.get("command_type"),
                            )
                            story.append(_RunbookCodeBlock(
                                title=heading or "COMMAND",
                                code=formatted_code,
                                text_font_name=font_name,
                            ))
                        execution_metadata = " · ".join(
                            item for item in (
                                f"身份 {command.get('run_as')}" if command.get("run_as") else "",
                                f"节点 {'、'.join(map(str, _items(command.get('node_scope'))))}" if _items(command.get("node_scope")) else "",
                                f"风险 {command.get('risk_level')}" if command.get("risk_level") else "",
                            ) if item
                        )
                        if execution_metadata:
                            story.append(Paragraph(_paragraph(execution_metadata), body))
                        for expected in _items(command.get("expected_result")):
                            story.append(Paragraph(
                                _paragraph(f"预期：{expected}"), body, bulletText="•"
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
    missing_facts = _items(payload.get("missing_facts"))
    if missing_facts:
        story.append(Paragraph("缺失的必要事实", step_style))
        for item_value in missing_facts:
            item = _mapping(item_value)
            story.append(Paragraph(
                _paragraph(
                    f"{item.get('fact_key') or '-'}：{item.get('reason') or '-'}；"
                    f"补齐位置 {item.get('resolution_source') or '-'}"
                ),
                body,
                bulletText="•",
            ))
    artifacts = _items(payload.get("artifacts"))
    if artifacts:
        story.append(Paragraph("脚本与配置清单", step_style))
        for item_value in artifacts:
            item = _mapping(item_value)
            story.append(Paragraph(
                _paragraph(
                    f"{item.get('file_name') or '-'} → {item.get('target_path') or '-'}；"
                    f"{item.get('file_mode') or '-'} / {item.get('run_as') or '-'}；"
                    f"SHA256 {item.get('sha256') or '-'}"
                ),
                body,
                bulletText="•",
            ))
    stop_conditions = _items(payload.get("stop_conditions"))
    if stop_conditions:
        story.append(Paragraph("停止条件", step_style))
        for item in stop_conditions:
            story.append(Paragraph(_paragraph(item), body, bulletText="•"))

    document.multiBuild(
        story,
        onFirstPage=_page_chrome,
        onLaterPages=_page_chrome,
    )
    return buffer.getvalue()
