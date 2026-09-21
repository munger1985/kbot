"""生成 知识检索 App 问文 Demo 的中文 PDF 文档。"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "documents"
FONT_NAME = "STSong-Light"


def _styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DemoTitle", parent=base["Title"], fontName=FONT_NAME, fontSize=20,
            leading=28, alignment=TA_CENTER, textColor=colors.HexColor("#16324F"),
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "DemoSubtitle", parent=base["Normal"], fontName=FONT_NAME, fontSize=10,
            leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#64748B"),
            spaceAfter=8 * mm,
        ),
        "heading": ParagraphStyle(
            "DemoHeading", parent=base["Heading2"], fontName=FONT_NAME, fontSize=13,
            leading=20, textColor=colors.HexColor("#16324F"), spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "DemoBody", parent=base["BodyText"], fontName=FONT_NAME, fontSize=10.5,
            leading=18, alignment=TA_LEFT, spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "DemoSmall", parent=base["BodyText"], fontName=FONT_NAME, fontSize=8.5,
            leading=13, textColor=colors.HexColor("#64748B"),
        ),
        "table": ParagraphStyle(
            "DemoTable", parent=base["BodyText"], fontName=FONT_NAME, fontSize=8.5,
            leading=13,
        ),
        "table_header": ParagraphStyle(
            "DemoTableHeader", parent=base["BodyText"], fontName=FONT_NAME, fontSize=8.5,
            leading=13, textColor=colors.white,
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def table(rows: list[list[str]], styles: dict[str, ParagraphStyle], widths: list[float]) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        item_style = styles["table_header"] if row_index == 0 else styles["table"]
        converted.append([p(str(value), item_style) for value in row])
    result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#245B78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def build_document(*, filename: str, title: str, date: str, customer: str, attendees: str,
                   purpose: str, discussion: list[list[str]], conclusions: list[str],
                   next_actions: list[list[str]], tags: str) -> None:
    styles = _styles()
    document = SimpleDocTemplate(
        str(OUTPUT / filename), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm, title=title, author="知识检索 App Demo",
    )
    story = [
        p(title, styles["title"]),
        p(f"客户：{customer}　|　会议日期：{date}　|　文档类型：会议纪要", styles["subtitle"]),
        table([
            ["会议主题", purpose],
            ["参会人员", attendees],
            ["检索标签", tags],
        ], styles, [32 * mm, 142 * mm]),
        Spacer(1, 4 * mm),
        p("一、讨论内容", styles["heading"]),
        table(discussion, styles, [35 * mm, 139 * mm]),
        p("二、会议结论", styles["heading"]),
    ]
    story.extend(p(f"• {item}", styles["body"]) for item in conclusions)
    story.extend([
        p("三、后续行动", styles["heading"]),
        table(next_actions, styles, [55 * mm, 55 * mm, 64 * mm]),
        Spacer(1, 5 * mm),
        p("本文件为 知识检索 App 知识库演示资料，内容为虚构的业务测试数据。沟通原文、需求解释和风险依据以本文件为准，CRM 中仅保留可统计的活动摘要。", styles["small"]),
    ])
    document.build(story)


def build_summary(*, filename: str, title: str, date: str, sections: list[tuple[str, str]],
                  evidence: list[list[str]]) -> None:
    styles = _styles()
    document = SimpleDocTemplate(
        str(OUTPUT / filename), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm, title=title, author="知识检索 App Demo",
    )
    story = [
        p(title, styles["title"]),
        p(f"统计周期：{date}　|　文档类型：客户需求与风险摘要", styles["subtitle"]),
    ]
    for heading, content in sections:
        story.extend([p(heading, styles["heading"]), p(content, styles["body"])])
    story.extend([
        p("重点客户证据索引", styles["heading"]),
        table(evidence, styles, [38 * mm, 38 * mm, 52 * mm, 46 * mm]),
        Spacer(1, 5 * mm),
        p("本文件用于演示跨文档检索和客户群体归纳，不替代 CRM 的精确统计结果。涉及金额、区域、客户等级和销售阶段时，应优先以问数结果为准。", styles["small"]),
    ])
    document.build(story)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_document(
        filename="CUS-0001_华辰智能制造_一期项目范围确认_20260910.pdf",
        title="智能工厂数据平台一期项目范围确认",
        date="2026-09-10", customer="华辰智能制造（CUS-0001）",
        attendees="高启明（客户信息化平台主管）；林晓峰（客户经理）；解决方案顾问",
        purpose="确认一期项目范围、交付边界和商务推进节点。",
        tags="华辰智能制造、智能工厂、数据平台、项目范围、采购推进",
        discussion=[
            ["议题", "讨论摘要"],
            ["一期范围", "客户确认优先覆盖生产经营看板、设备数据接入和管理层指标，不纳入二期的供应链预测。"],
            ["交付要求", "客户希望在项目启动后完成分阶段验收，并要求提供数据权限和审计说明。"],
            ["商务状态", "一期方案进入商务谈判，客户计划在十月中旬前完成内部评审。"],
        ],
        conclusions=["一期项目保持当前方案，二期扩展单独形成后续机会。", "客户对数据权限和审计能力有明确关注，需要在下一版方案中补充说明。"],
        next_actions=[["补充权限与审计说明", "解决方案顾问", "2026-09-18"], ["确认商务谈判安排", "林晓峰", "2026-09-22"]],
    )
    build_document(
        filename="CUS-0002_云峰连锁零售_预算风险沟通_20260728.pdf",
        title="门店经营分析平台预算风险沟通",
        date="2026-07-28", customer="云峰连锁零售（CUS-0002）",
        attendees="沈妍（客户数字化负责人）；周婉宁（客户经理）；售前顾问",
        purpose="了解预算调整对门店经营分析平台机会的影响。",
        tags="云峰连锁零售、门店分析、预算调整、风险、采购周期",
        discussion=[
            ["议题", "讨论摘要"],
            ["预算情况", "客户表示年度预算正在重新评估，门店经营分析项目可能分阶段采购。"],
            ["优先需求", "客户优先关注门店销售排名、区域对比和促销效果分析，暂缓复杂预测能力。"],
            ["推进风险", "如果预算审批延后，原定年内上线计划可能顺延。客户希望先获得轻量化方案和分期报价。"],
        ],
        conclusions=["项目仍有明确需求，但预算和采购周期构成主要风险。", "应准备分期报价，先覆盖核心门店分析场景。"],
        next_actions=[["提供分期报价方案", "周婉宁", "2026-09-20"], ["确认预算审批进展", "周婉宁", "2026-09-16"]],
    )
    build_document(
        filename="CUS-0003_智联供应链_物流方案演示_20260905.pdf",
        title="物流网络优化方案演示纪要",
        date="2026-09-05", customer="智联供应链（CUS-0003）",
        attendees="赵博文（客户供应链总监）；陈思远（客户经理）；方案顾问",
        purpose="演示物流网络优化方案，确认客户的评估重点。",
        tags="智联供应链、物流网络、路线优化、方案演示、评估",
        discussion=[
            ["议题", "讨论摘要"],
            ["方案反馈", "客户认可区域运输成本分析和异常线路识别能力，希望补充多仓协同的演示案例。"],
            ["数据准备", "客户需要确认历史订单、仓库和线路数据的可用范围，再决定试点区域。"],
            ["采购判断", "客户处于资格评估阶段，预计先进行小范围验证，再决定是否扩大项目。"],
        ],
        conclusions=["客户对方案持积极态度，但需要先完成数据可用性确认。", "下一次沟通应聚焦多仓协同和试点范围。"],
        next_actions=[["准备多仓协同案例", "方案顾问", "2026-09-19"], ["确认试点数据范围", "陈思远", "2026-09-19"]],
    )
    build_document(
        filename="CUS-0004_星河医疗_二期交付回顾_20260912.pdf",
        title="医疗数据中台二期交付回顾",
        date="2026-09-12", customer="星河医疗集团（CUS-0004）",
        attendees="许安宁（客户信息中心主任）；林晓峰（客户经理）；交付负责人",
        purpose="回顾二期交付效果，确认后续运营和扩展安排。",
        tags="星河医疗、医疗数据中台、交付回顾、运营、扩展",
        discussion=[
            ["交付情况", "二期核心功能已完成验收，客户认为数据统一和管理报表效率有明显改善。"],
            ["运营安排", "客户计划在下一季度扩大使用部门，并建立月度运营复盘机制。"],
            ["扩展方向", "客户提出希望后续支持科研数据主题和跨院区指标对比。"],
        ],
        conclusions=["二期交付反馈积极，客户处于活跃经营状态。", "科研数据和跨院区分析可作为后续扩展方向。"],
        next_actions=[["提交扩展方向清单", "林晓峰", "2026-09-25"], ["安排月度运营复盘", "交付负责人", "2026-09-30"]],
    )
    build_document(
        filename="CUS-0006_海岳新能源_生产监控预算讨论_20260802.pdf",
        title="新能源生产监控平台预算讨论",
        date="2026-08-02", customer="海岳新能源（CUS-0006）",
        attendees="韩立（客户数据平台主管）；陈思远（客户经理）；售前顾问",
        purpose="讨论生产监控平台预算、实施范围和采购节奏。",
        tags="海岳新能源、生产监控、预算、交付周期、风险",
        discussion=[
            ["预算情况", "客户认为方案价值明确，但当前年度预算无法覆盖完整范围，可能需要分两期建设。"],
            ["实施风险", "客户担心多工厂数据接入周期过长，希望先选择一个工厂验证。"],
            ["决策节奏", "客户内部需要重新确认预算和工厂优先级，采购时间存在不确定性。"],
        ],
        conclusions=["机会金额较高，但预算和多工厂交付复杂度带来较大风险。", "应将单工厂试点作为下一轮沟通重点。"],
        next_actions=[["设计单工厂试点方案", "陈思远", "2026-09-17"], ["确认预算审批状态", "陈思远", "2026-09-16"]],
    )
    build_document(
        filename="CUS-0008_蓝鲸软件_知识门户演示_20260906.pdf",
        title="研发协同与知识门户演示纪要",
        date="2026-09-06", customer="蓝鲸软件科技（CUS-0008）",
        attendees="顾言（客户产品副总裁）；周婉宁（客户经理）；知识管理顾问",
        purpose="演示研发知识门户，确认知识检索和权限管理需求。",
        tags="蓝鲸软件、知识门户、研发协同、知识检索、权限管理",
        discussion=[
            ["核心需求", "客户希望统一沉淀研发规范、项目复盘和产品资料，并按团队权限控制访问范围。"],
            ["问答场景", "客户重点关注按项目和产品检索资料，以及从会议纪要中归纳未解决事项。"],
            ["上线计划", "客户希望先选择一个研发团队试点，再评估是否推广到全公司。"],
        ],
        conclusions=["客户对知识门户和检索问答有明确兴趣，处于方案评估阶段。", "权限隔离和试点范围是下一步决策的关键。"],
        next_actions=[["确认试点团队范围", "顾言", "2026-09-21"], ["补充权限方案说明", "周婉宁", "2026-09-23"]],
    )
    build_document(
        filename="CUS-0010_启明消费金融_风控升级暂停_20260715.pdf",
        title="风控数据服务升级项目暂停沟通",
        date="2026-07-15", customer="启明消费金融（CUS-0010）",
        attendees="陆成（客户科技平台主管）；王若琳（客户经理）；交付与合规顾问",
        purpose="确认风控数据服务升级项目暂停原因及恢复条件。",
        tags="启明消费金融、风控、数据服务、预算审批、项目暂停",
        discussion=[
            ["暂停原因", "客户表示内部预算审批和合规评估尚未完成，暂时无法确认项目启动时间。"],
            ["保留事项", "客户仍认可客户画像和风险数据整合方向，希望保留现有方案材料。"],
            ["恢复条件", "预算审批通过、合规范围确认后，客户会重新安排项目评估。"],
        ],
        conclusions=["项目当前暂停，不能按原计划推进。", "客户需求仍然存在，但恢复时间取决于预算和合规审批。"],
        next_actions=[["跟进预算审批状态", "王若琳", "2026-09-16"], ["准备合规范围说明", "合规顾问", "2026-09-20"]],
    )
    build_summary(
        filename="2026Q3_重点客户需求与风险摘要.pdf",
        title="2026 年第三季度重点客户需求与风险摘要",
        date="2026-07-01 至 2026-09-12",
        sections=[
            ("一、总体观察", "本周期内，重点客户的主要关注集中在数据权限、分期实施、预算审批和试点范围。客户对数据分析、知识门户和经营管理场景整体保持积极兴趣，但高金额机会的推进速度明显受预算与合规流程影响。"),
            ("二、共性需求", "制造和能源客户关注生产经营数据整合；零售和物流客户关注区域分析与运营效率；软件客户关注知识检索、权限隔离和研发协同；金融客户关注风控数据整合和合规边界。"),
            ("三、重点风险", "云峰连锁零售存在预算调整风险；海岳新能源存在多工厂交付和预算风险；启明消费金融项目处于暂停状态；部分客户需要先完成数据可用性或内部审批，才能进入下一阶段。"),
            ("四、建议动作", "优先为高金额但存在风险的客户准备分期方案、试点方案和审批材料；对已经赢单或活跃运营的客户推进扩展需求访谈，形成后续机会。"),
        ],
        evidence=[
            ["云峰连锁零售", "预算与采购周期", "可能分期采购，年内上线存在不确定性", "会议纪要 2026-07-28"],
            ["海岳新能源", "单工厂试点", "完整范围预算不足，多工厂接入复杂", "会议纪要 2026-08-02"],
            ["蓝鲸软件科技", "权限与试点", "先选择研发团队，再评估推广", "会议纪要 2026-09-06"],
            ["启明消费金融", "预算与合规", "项目暂停，等待审批恢复", "会议纪要 2026-07-15"],
        ],
    )


if __name__ == "__main__":
    main()
