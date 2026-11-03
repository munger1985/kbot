"""日常巡检共享分区 PDF：只消费正式投影，不根据数据库引擎重新排版。"""

from datetime import datetime
from math import isfinite
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from reportlab.graphics.shapes import Drawing, PolyLine
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle


def daily_inspection_pdf_story(*, data, font_name, body, section, cover_title):
    """目录、专题、SQL 分析与复查清单均保持网页同源。"""
    story = []
    header = data.get("header") or {}
    conclusion = data.get("conclusion") or {}
    regions = list(data.get("sections") or ())
    try:
        timezone = ZoneInfo(header.get("timezone") or "UTC")
    except ZoneInfoNotFoundError:
        timezone = ZoneInfo("UTC")

    def p(value, style=body, keep=False):
        item = Paragraph(escape(str(value if value is not None else "未记录")).replace("\n", "<br/>"), style)
        item.keepWithNext = keep
        return item

    def moment(value):
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return str(value or "未记录")

    def severity(value):
        return {"CRITICAL": "严重", "HIGH": "高", "MEDIUM": "中", "LOW": "低", "INFO": "参考"}.get(value, str(value or "未记录"))

    def heading(title, anchor=None):
        text = escape(title)
        if anchor:
            text = '<a name="' + anchor + '"/>' + text
        item = Paragraph(text, section)
        item.keepWithNext = True
        story.append(item)

    def table(headers, rows, widths):
        def cell(value):
            return value if isinstance(value, (Paragraph, Drawing)) else p(value)
        item = Table([[cell(value) for value in row] for row in [headers, *rows]],
                     repeatRows=1, splitInRow=1, colWidths=[width * mm for width in widths], hAlign="LEFT")
        item.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF1")),
            ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#CDD7DC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([item, Spacer(1, 4 * mm)])

    def check_table(checks):
        if not checks:
            return
        table(["检查内容", "结果", "实测情况", "判断依据"], [
            [check.get("title"), check.get("status_display"),
             (str(len(check["detail_records"])) + " 条对象记录，详见下方。") if check.get("detail_records") else check.get("actual_display") or check.get("summary"),
             check.get("basis")] for check in checks
        ], [31, 17, 73, 49])
        for check in checks:
            for index, record in enumerate(check.get("detail_records") or (), 1):
                heading(str(check.get("title") or "检查") + " · 对象 " + str(index))
                table(["项目", "实测值"], [[field["label"], field["value"]] for field in record], [50, 120])

    def trend(metric):
        points = list(metric.get("trend_points") or ())
        if metric.get("baseline_status") != "AVAILABLE" or len(points) < 2:
            return "暂无连续趋势"
        try:
            dates = [datetime.fromisoformat(point["date"]) for point in points]
            values = [float(point["value"]) for point in points]
            if not all(isfinite(value) for value in values) or any((b-a).days != 1 for a, b in zip(dates, dates[1:])):
                return "日期有缺测，未连接趋势线"
        except (KeyError, ValueError, TypeError):
            return "暂无连续趋势"
        width, height = 20 * mm, 10 * mm
        drawing = Drawing(width, height)
        low, span = min(values), max(max(values)-min(values), 1e-9)
        coords = []
        for index, value in enumerate(values):
            coords.extend((width*index/(len(values)-1), 2+(height-4)*(value-low)/span))
        drawing.add(PolyLine(coords, strokeColor=colors.HexColor("#244C73"), strokeWidth=1))
        return drawing

    story.append(p(header.get("title") or "数据库日常巡检报告", cover_title))
    if data.get("preview"):
        story.append(p("模拟版式预览 · 非真实巡检结果"))
    table(["项目", "内容"], [
        ["业务系统 / 数据库", str(header.get("business_system_name") or "未记录") + " / " + str(header.get("target_name") or "未记录")],
        ["数据库类型 / 版本", str(header.get("database_type") or "") + " / " + str(header.get("database_version") or "")],
        ["环境 / 时区", str(header.get("environment") or "") + " / " + str(header.get("timezone") or "UTC")],
        ["巡检时间", moment(header.get("report_time"))],
        ["观察区间", moment(header.get("observation_start")) + " 至 " + moment(header.get("observation_end"))],
        ["报告编号", str(header.get("report_number") or "") + " · v" + str(header.get("report_version") or 1)],
    ], [35, 135])
    heading("报告目录")
    table(["区域", "结论", "主要情况"], [
        [Paragraph('<a href="#daily-' + str(region["ordinal"]) + '">' + escape(f'{region["ordinal"]:02d} · {region["title"]}') + '</a>', body),
         region["status_display"], region["summary"]] for region in regions
    ], [50, 25, 95])

    for region in regions:
        key = region["section_id"]
        heading(f'{region["ordinal"]:02d} · {region["title"]}', "daily-" + str(region["ordinal"]))
        story.append(p(region["status_display"] + " · " + region["summary"]))
        if key == "OVERVIEW":
            story.append(p(conclusion.get("text") or "检查未完成。"))
            table(["运行评价", "必要检查完成", "问题与风险", "未完成事项"], [[
                conclusion.get("health_status_display"),
                f'{conclusion.get("completed_required_count", "未记录")} / {conclusion.get("required_check_count", "未记录")}',
                f'{conclusion.get("attention_count", 0)} 项', f'{conclusion.get("incomplete_count", 0)} 项',
            ]], [43, 43, 42, 42])
            if data.get("finding_index"):
                heading("重要问题定位")
                titles = {item["section_id"]: item["title"] for item in regions}
                table(["严重性", "问题 / 对象", "所属区域"], [
                    [severity(issue.get("severity")), issue.get("title", "") + " · " + str(issue.get("object_name") or issue.get("summary") or ""),
                     titles.get(issue.get("report_section"), "巡检总览")] for issue in data["finding_index"]
                ], [20, 100, 50])
            if data.get("limitations"):
                heading("检查限制与未完成事项")
                story.extend(p(value) for value in data["limitations"])
        else:
            story.append(p(region.get("description") or ""))
        linked = {item.get("problem_id") for group in region.get("sql_details") or () for item in group.get("statements") or ()}
        if region.get("findings"):
            for issue in region["findings"]:
                if issue.get("finding_id") in linked:
                    continue
                heading(str(issue.get("title") or "巡检发现") + " · " + severity(issue.get("severity")))
                story.append(p(issue.get("summary") or ""))
                for label, field in (("影响对象", "object_name"), ("判断依据", "basis"), ("建议", "recommendation")):
                    if issue.get(field):
                        story.append(p(label + "：" + str(issue[field])))
        if key not in {"SQL", "OVERVIEW", "BASELINE", "ACTIONS"}:
            checks = list(region.get("checks") or ())
            normal = [check for check in checks if check.get("status") in {"CLEAR", "COMPLIANT", "NOT_APPLICABLE"}]
            check_table([check for check in checks if check not in normal])
            if normal:
                heading(f"已完成的正常 / 不适用检查 · {len(normal)} 项")
                if region.get("normal_detail") == "FULL":
                    check_table(normal)
                else:
                    story.append(p("；".join(str(check.get("title")) for check in normal) + "。完整记录见附录。"))
            metrics = list(region.get("metrics") or ())
            if metrics:
                cols = ["指标 / 范围", "最新样本"] + (["采集窗口平均 / 最大"] if region.get("show_statistics") else []) + ["判断"]
                table(cols, [
                    [str(item.get("display_name")) + " / " + str(item.get("dimension_label") or ""), item.get("current_display")]
                    + ([str(item.get("average_display")) + " / " + str(item.get("maximum_display"))] if region.get("show_statistics") else [])
                    + [item.get("status_display")] for item in metrics
                ], [54, 30, 62, 24] if region.get("show_statistics") else [95, 45, 30])
        if key == "SQL" and not region.get("sql_details"):
            check_table(region.get("checks") or ())
        for group in region.get("sql_details") or ():
            story.append(p(group.get("summary") or "问题 SQL 检查尚未完成。"))
            story.append(p(group.get("note") or ""))
            for item in group.get("statements") or ():
                heading(f'{item["ordinal"]:02d} · {item.get("schema_name") or header.get("target_name")} · {severity(item.get("severity"))}')
                story.extend([p(item.get("problem_summary")), p("判断依据：" + str(item.get("basis") or "")),
                              p(item.get("text_label") or "SQL 文本"), p(item.get("statement_text") or "来源未返回 SQL 文本。")])
                table(["执行次数", "总耗时", "平均耗时", "最大耗时"], [[item.get(field) for field in
                      ("execution_count_display", "total_latency_display", "average_latency_display", "maximum_latency_display")]], [35, 55, 40, 40])
                review = item.get("agent_review") or {}
                heading("Agent 初步分析" + ("（未完成）" if review.get("status") != "COMPLETE" else ""))
                story.append(p(review.get("analysis") or "Agent 初步分析尚未生成。"))
                if review.get("status") == "COMPLETE":
                    for cause in review.get("possible_causes") or ():
                        story.extend([p("可能原因：" + cause["possible_cause"]), p("验证：" + cause["verification"])])
                    for advice in review.get("recommendations") or ():
                        story.extend([p("优化建议：" + advice["action"]), p(advice["rationale"]),
                                      p("验证：" + advice["validation"] + ("；实施前须审批。" if advice.get("requires_approval") else ""))])
                    story.extend(p(value) for value in review.get("limitations") or ())
                else:
                    heading("规则复核建议（非 Agent 分析）")
                    story.extend(p(value) for value in item.get("rule_recommendations") or ())
        if key == "BASELINE" and region.get("comparisons"):
            table(["指标 / 范围", "对比值", "七天均值", "变化量 / 变化率"] + (["日均趋势"] if region.get("show_trend") else []), [
                [str(item.get("display_name")) + " / " + str(item.get("dimension_label") or ""),
                 str(item.get("comparison_display")) + "\n" + str(item.get("comparison_label")),
                 str(item.get("baseline_display")) + "\n" + str(item.get("baseline_window_label") if item.get("baseline_status") == "AVAILABLE" else item.get("baseline_reason") or ""),
                 item.get("change_display")] + ([trend(item)] if region.get("show_trend") else [])
                for item in region["comparisons"]
            ], [38, 26, 41, 43, 22] if region.get("show_trend") else [45, 30, 50, 45])
        if key == "ACTIONS":
            if region.get("actions"):
                table(["优先级 / 对象", "处置事项", "复查依据"], [
                    [severity(item.get("priority")) + " / " + str(item.get("object_name") or ""), item["action"], item["validation"]]
                    for item in region["actions"]
                ], [35, 75, 60])
            else:
                story.append(p("无额外处置事项。" if conclusion.get("health_status") == "NORMAL" else "请先完成本报告列明的检查和问题复核。"))
    heading("附录：统计方法与技术追溯", "daily-appendix")
    boundary = data.get("evidence_boundary") or {}
    story.append(p(boundary.get("methodology") or "以本报告冻结的检查范围和规则为准。"))
    heading("完整检查索引")
    check_table(data.get("checks") or ())
    story.extend(p(value) for value in data.get("gaps") or ())
    if boundary.get("sources"):
        heading("技术追溯信息")
        story.append(p("证据编号用于定位原始记录，内容指纹用于核对记录内容；仅供技术核查，不属于巡检问题。"))
        story.extend(p(value) for value in boundary["sources"])
    return story
