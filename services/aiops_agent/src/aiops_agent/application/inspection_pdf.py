"""日常与月度巡检的专业报告排版，消费与网页相同的冻结投影。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle


def inspection_pdf_story(
    *, data: dict[str, Any], monthly: bool, font_name, body, section, cover_title,
) -> list[Any]:
    """业务结论在前，正常明细可核查，原始来源和质量进入附录。"""
    header = dict(data.get("header") or {})
    conclusion = dict(data.get("conclusion") or {})
    story: list[Any] = []
    try:
        timezone = ZoneInfo(str(header.get("timezone") or "UTC"))
    except ZoneInfoNotFoundError:
        timezone = ZoneInfo("UTC")

    def paragraph(value: object, style=body):
        return Paragraph(escape(str(value if value is not None else "未记录")).replace("\n", "<br/>"), style)

    def moment(value: object) -> str:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return str(value or "未记录")

    def heading(title: str):
        story.append(paragraph(title, section))

    def table(headers, rows, widths):
        values = [[paragraph(value) for value in row] for row in [headers, *rows]]
        item = Table(values, repeatRows=1, colWidths=tuple(value * mm for value in widths), hAlign="LEFT")
        item.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF1")),
            ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#CDD7DC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend((item, Spacer(1, 4 * mm)))

    story.extend((paragraph(header.get("title") or "数据库巡检报告", cover_title), Spacer(1, 3 * mm)))
    metadata = [
        ["运维范围", header.get("scope_name") if monthly else header.get("target_name")],
        ["报告时间", moment(header.get("report_time"))],
        ["观察区间" if not monthly else "统计周期", moment(header.get("period_start") if monthly else header.get("observation_start")) + " 至 " + moment(header.get("period_end") if monthly else header.get("observation_end"))],
        ["报告时区", header.get("timezone") or "UTC"],
        ["报告编号", header.get("report_number")],
    ]
    if not monthly:
        metadata.insert(1, ["业务系统", header.get("business_system_name") or "未记录"])
        metadata.insert(2, ["数据库", str(header.get("database_type") or "") + " " + str(header.get("database_version") or "")])
    table(["项目", "内容"], metadata, (30, 140))
    heading("月度运行结论" if monthly else "巡检结论")
    story.append(paragraph(conclusion.get("health_status_display") or "检查未完成", section))
    story.append(paragraph(conclusion.get("text") or "未形成完整检查结论。"))
    limitations = list(data.get("limitations") or ())
    if limitations:
        heading("检查限制与未完成事项")
        for value in limitations:
            story.append(paragraph(value))
    if monthly:
        targets = list(data.get("target_results") or ())
        if targets:
            heading("系统与数据库概况")
            table(["数据库", "月度评价", "最近巡检", "问题与待办"], [
                [item.get("target_name"), item.get("health_status_display"),
                 item.get("latest_health_display"),
                 f"{item.get('issue_count') or 0} 项问题；" + "；".join(item.get("limitations") or ())]
                for item in targets
            ], (35, 25, 30, 80))
    findings = list(data.get("issues" if monthly else "findings") or ())
    if findings:
        heading("稳定性与问题闭环台账" if monthly else "问题与风险")
        for index, item in enumerate(findings, 1):
            story.append(paragraph(f"{index}. {item.get('target_name') or ''} {item.get('title') or '巡检发现'}", section))
            story.append(paragraph(item.get("summary") or item.get("impact") or ""))
            if monthly:
                story.append(paragraph(
                    f"首次发现 {item.get('first_seen') or '未记录'}；最近发现 {item.get('last_seen') or '未记录'}；"
                    f"观测 {item.get('observation_count') or 0} 次；状态：{item.get('lifecycle_display') or '待验证'}。"
                    + (f"已确认事件 {item['occurrence_count']} 次。" if item.get("occurrence_count") is not None else "观测次数不等于事故次数。")
                ))
            else:
                for label, key in (("判断依据", "basis"), ("影响对象", "object_name"), ("处置建议", "recommendation")):
                    if item.get(key):
                        story.append(paragraph(f"{label}：{item[key]}"))
    if not monthly:
        checks = list(data.get("checks") or ())
        heading("基础检查明细")
        if checks:
            table(["检查内容", "结果", "实测情况", "判断依据"], [
                [item.get("title"), item.get("status_display"),
                 item.get("actual_display") or item.get("summary"), item.get("basis")]
                for item in checks
            ], (31, 18, 61, 60))
        else:
            story.append(paragraph("本报告未记录可展示的基础检查明细，不能据此补造正常结果。"))
    metrics = list(data.get("metrics") or ())
    if metrics:
        heading("性能与运行趋势" if monthly else "关键运行指标")
        if monthly:
            story.append(paragraph("月均和 P95 基于日均值分布；日均峰谷不等于原始采样峰谷，状态日均值不代表精确月度可用率。"))
            table(["指标 / 数据库", "日均值平均", "日均值 P95", "日均最小 / 最大", "有效日期"], [
                [str(item.get("display_name") or "") + " " + str(item.get("dimension_label") or ""),
                 item.get("monthly_average_display"), item.get("p95_display"),
                 str(item.get("minimum_display") or "") + " / " + str(item.get("maximum_display") or ""),
                 f"{item.get('effective_days') or 0}/{item.get('expected_days') or 0}"]
                for item in metrics
            ], (44, 29, 29, 45, 23))
        else:
            table(["指标 / 范围", "最新样本", "观察期平均 / 最大", "判断"], [
                [str(item.get("display_name") or "") + " " + str(item.get("dimension_label") or ""),
                 item.get("current_display"),
                 str(item.get("average_display") or "") + " / " + str(item.get("maximum_display") or ""),
                 item.get("status_display")]
                for item in metrics
            ], (58, 32, 58, 22))
    if monthly and data.get("capacity_rows"):
        heading("容量与增长")
        table(["容量 / 数据库", "首个有效日", "末个有效日", "净变化"], [
            [str(item.get("display_name") or "") + " " + str(item.get("dimension_label") or ""),
             item.get("first_display"), item.get("last_display"), item.get("net_change_display")]
            for item in data["capacity_rows"]
        ], (65, 35, 35, 35))
    heading("处置建议与下月重点" if monthly else "处置事项")
    actions = list(data.get("recommendations") or ())
    if actions:
        for index, value in enumerate(actions, 1):
            story.append(paragraph(f"{index}. {value}"))
    else:
        story.append(paragraph("无额外处置事项。" if conclusion.get("health_status") == "NORMAL" else "请先完成本报告列明的检查和问题复核。"))
    heading("附录：统计方法与技术追溯")
    boundary = dict(data.get("evidence_boundary") or {})
    story.append(paragraph(boundary.get("methodology") or "以本报告冻结的检查范围和规则为准。"))
    if monthly:
        comparison = dict(data.get("period_comparison") or {})
        story.append(paragraph("上月对比：" + str(comparison.get("status_display") or "无可比上月报告")))
        if comparison.get("status") == "AVAILABLE":
            story.append(paragraph(
                f"新增 {len(comparison.get('new_issues') or ())} 项，"
                f"持续 {len(comparison.get('persistent_issues') or ())} 项，"
                f"已验证恢复 {len(comparison.get('resolved_issues') or ())} 项，"
                f"待验证 {len(comparison.get('pending_verification_issues') or ())} 项。"
            ))
    for value in data.get("gaps") or ():
        story.append(paragraph(value))
    for value in boundary.get("sources") or ():
        story.append(paragraph(value))
    return story
