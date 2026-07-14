"""结构质量门 — 逐页评估 Docling 解析结构质量，决定是否触发 VLM 修复。

评估维度:
1. 标题置信度 (heading_confidence): Docling 标题识别的可靠性
2. 多栏混乱风险 (multi_column_risk): 多栏页面阅读顺序可能错乱
3. 跨页截断 (page_span_flag): 跨页表格/段落需要 VLM 判断连续性
4. 文字覆盖率 (text_coverage): 扫描件等低文本密度页

返回 0.0-1.0 质量分。分数 < 阈值 → 触发 VLM 结构修复。
"""

from dataclasses import dataclass, field
from loguru import logger

from .hierarchy_builder import SemanticNode


# ── 质量门阈值 ──────────────────────────────────────────────
DEFAULT_QUALITY_THRESHOLD = 0.70   # 低于此分触发 VLM 修复
PRECISION_THRESHOLD = 0.0          # precision 模式：全部页面触发
ECONOMY_THRESHOLD = 1.0            # economy 模式：永不触发


@dataclass
class PageQualityReport:
    """单页结构质量报告"""
    page_no: int
    score: float                          # 0.0-1.0 综合质量分
    heading_confidence: float             # 标题置信度
    multi_column_risk: float              # 多栏风险 (0=无, 1=确定多栏)
    page_span_flag: bool                  # 是否有跨页截断
    text_coverage: float                  # 文字覆盖率
    heading_count: int                    # 标题数量
    paragraph_count: int                  # 段落数量
    table_count: int                      # 表格数量
    picture_count: int                    # 图片数量
    issues: list[str] = field(default_factory=list)  # 诊断信息
    needs_vlm_repair: bool = False        # 是否需要 VLM 修复


class StructureQualityGate:
    """逐页结构质量评估器。

    用法:
        gate = StructureQualityGate(threshold=0.70)
        reports = gate.assess(hierarchy, doc)
        for r in reports:
            if r.needs_vlm_repair:
                ...  # 对该页触发 VLM 修复
    """

    def __init__(self, threshold: float = DEFAULT_QUALITY_THRESHOLD):
        self.threshold = threshold

    # ── 公开 API ──────────────────────────────────────────

    def assess(self, hierarchy: SemanticNode) -> list[PageQualityReport]:
        """评估整个文档所有页面的结构质量。

        Args:
            hierarchy: HierarchyBuilder.build() 输出的层级树

        Returns:
            按页码排序的质量报告列表
        """
        # 收集所有节点按页码分组
        pages: dict[int, list[SemanticNode]] = {}
        self._collect_by_page(hierarchy, pages)

        reports: list[PageQualityReport] = []
        for page_no in sorted(pages.keys()):
            nodes = pages[page_no]
            report = self._assess_page(page_no, nodes)
            report.needs_vlm_repair = report.score < self.threshold
            reports.append(report)

        total = len(reports)
        repair_count = sum(1 for r in reports if r.needs_vlm_repair)
        if total > 0:
            logger.info(
                f"[StructureQualityGate] {total} 页评估完成: "
                f"{repair_count}/{total} 页需要 VLM 修复 "
                f"(阈值={self.threshold}, 修复率={repair_count/total:.0%})"
            )

        return reports

    # ── 单页评估 ─────────────────────────────────────────

    def _assess_page(self, page_no: int,
                     nodes: list[SemanticNode]) -> PageQualityReport:
        """评估单页质量"""
        headings = [n for n in nodes if n.node_type == 'title']
        paragraphs = [n for n in nodes if n.node_type == 'paragraph']
        tables = [n for n in nodes if n.node_type == 'table']
        pictures = [n for n in nodes if n.node_type == 'picture']

        issues: list[str] = []

        # 1. 标题置信度
        hc = self._heading_confidence(headings, issues)

        # 2. 多栏风险
        mc = self._multi_column_risk(nodes, issues)

        # 3. 跨页截断
        ps = self._page_span_detected(nodes)

        # 4. 文字覆盖率
        tc = self._text_coverage(nodes)

        # 综合评分 (权重可调)
        # 标题: 40% | 多栏: 25% | 文字覆盖: 25% | 跨页: 10%
        score = (
            hc * 0.40 +
            (1.0 - mc) * 0.25 +  # 多栏风险越高，得分越低
            tc * 0.25 +
            (0.5 if ps else 1.0) * 0.10
        )

        return PageQualityReport(
            page_no=page_no,
            score=round(min(1.0, max(0.0, score)), 3),
            heading_confidence=round(hc, 3),
            multi_column_risk=round(mc, 3),
            page_span_flag=ps,
            text_coverage=round(tc, 3),
            heading_count=len(headings),
            paragraph_count=len(paragraphs),
            table_count=len(tables),
            picture_count=len(pictures),
            issues=issues,
        )

    # ── 子评分函数 ────────────────────────────────────────

    @staticmethod
    def _heading_confidence(headings: list[SemanticNode],
                            issues: list[str]) -> float:
        """评估标题置信度。

        Docling 的 SectionHeaderItem 没有内置置信度，用启发式判断:
        - 标题文字过短 (<2 字) → 可能是装饰性文本
        - 标题文字过长 (>60 字) → 可能是正文被误判为标题
        - 无标题 → 可能是正文页（不一定是问题，但减分）
        - 同页出现同为 level=1 的多个标题 → 可能是编号体系未识别
        """
        if not headings:
            issues.append("无标题")
            return 0.6  # 可能是纯正文页，不严重

        score = 1.0
        penalty_per_heading = 0.0

        # 检查每个标题
        for h in headings:
            text_len = len(h.text.strip())
            if text_len < 2:
                penalty_per_heading += 0.1
                issues.append(f"标题过短(len={text_len}): {h.text[:20]}")
            elif text_len > 60:
                penalty_per_heading += 0.15
                issues.append(f"标题过长(len={text_len}): {h.text[:30]}...")

        # 检查同级标题重复 (如两个 level=1)
        levels = [h.level for h in headings if h.level > 0]
        if len(levels) >= 2:
            level_counts: dict[int, int] = {}
            for lv in levels:
                level_counts[lv] = level_counts.get(lv, 0) + 1
            for lv, cnt in level_counts.items():
                if cnt >= 2:
                    penalty_per_heading += 0.2
                    issues.append(f"同页 {cnt} 个 level={lv} 标题，可能编号系统未识别")

        # 标题间没有层级递进 (全部 level=1)
        if levels and all(lv == 1 for lv in levels) and len(levels) > 1:
            penalty_per_heading += 0.15
            issues.append("多个同级别标题无层级递进")

        score -= min(0.5, penalty_per_heading)  # 最多扣 0.5
        return max(0.1, score)

    @staticmethod
    def _multi_column_risk(nodes: list[SemanticNode],
                           issues: list[str]) -> float:
        """评估多栏混乱风险。

        通过 x 坐标分布判断是否多栏布局。
        """
        items_with_bbox = [n for n in nodes if n.bbox]
        if len(items_with_bbox) < 4:
            return 0.0

        x_centers = [(n.bbox[0] + n.bbox[2]) / 2 for n in items_with_bbox]
        x_centers.sort()

        gaps = [x_centers[i + 1] - x_centers[i]
                for i in range(len(x_centers) - 1)]
        if not gaps:
            return 0.0

        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)

        # 最大间隔 > 2 倍平均 = 大概率多栏
        if max_gap > avg_gap * 2.0 and max_gap > 0.15:
            issues.append(f"检测到多栏布局 (max_gap={max_gap:.3f}, avg_gap={avg_gap:.3f})")
            # 风险系数: gap 越大越危险
            risk = min(1.0, max_gap / (avg_gap * 3.0 + 0.01))
            return round(risk, 3)

        return 0.0

    @staticmethod
    def _page_span_detected(nodes: list[SemanticNode]) -> bool:
        """检测是否有跨页截断标记"""
        return any(n.is_page_span for n in nodes)

    @staticmethod
    def _text_coverage(nodes: list[SemanticNode]) -> float:
        """评估文字覆盖率。

        纯图片/表格页 (无 paragraph) → 极低覆盖率，扫描件特征。
        """
        text_nodes = [n for n in nodes
                      if n.node_type in ('paragraph', 'title')]
        table_nodes = [n for n in nodes if n.node_type == 'table']
        picture_nodes = [n for n in nodes if n.node_type == 'picture']

        total_nodes = len(text_nodes) + len(table_nodes) + len(picture_nodes)
        if total_nodes == 0:
            return 0.0

        # text 节点占比 (表格算半覆盖，图片算未覆盖)
        coverage = (len(text_nodes) + len(table_nodes) * 0.3) / total_nodes
        return round(min(1.0, coverage), 3)

    # ── 辅助 ──────────────────────────────────────────────

    @staticmethod
    def _collect_by_page(node: SemanticNode,
                         pages: dict[int, list[SemanticNode]]):
        """递归收集所有节点按页码分组"""
        if node.node_type != 'root':
            pages.setdefault(node.page_num, []).append(node)
        for child in node.children:
            StructureQualityGate._collect_by_page(child, pages)
