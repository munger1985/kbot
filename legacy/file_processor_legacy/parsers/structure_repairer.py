"""VLM 结构修复器 — 对低质量页面进行结构修正。

与旧 PrecisionAnalyzer 的区别:
- 旧：VLM 从零输出整页 Markdown（无上下文，易丢失层级）
- 新：VLM 基于 Docling 最佳猜测 + 全局上下文进行"修正"（只改错的部分）

输入（关键改进）:
- 当前页图片
- Docling 的初步分析（标题列表、级别、bbox）
- 文档全局摘要
- 前一页的标题栈（承接层级）
- 下一页第一个标题（知道去向）
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from docling_core.types.doc.document import DoclingDocument

from platform_clients import AIModelClient


# ═══════════════════════════════════════════════════════════════
# VLM Prompt — 结构修复模式（注入全局上下文）
# ═══════════════════════════════════════════════════════════════

STRUCTURE_REPAIR_PROMPT = """你是一个文档结构质检专家。你的任务是验证并修正以下文档页面的标题层级。

### 文档上下文
- 文档摘要：{global_summary}
- 当前页：第 {page_no} 页 / 共 {total_pages} 页
- 上一页标题栈：{prev_heading_stack}
- 下一页第一个标题：{next_first_heading}

### Docling 的初步分析（需要你验证）
{current_page_headings}

### 你的任务
1. 验证每个 heading 是否为真正的标题（考虑编号体系、字体大小、位置）
2. 修正错误的 heading 级别（当前页标题级别需承接上一页的层级）
3. 如果 Docling 漏掉了某个标题，补充它
4. 如果 Docling 把正文误判为标题，标记为删除
5. 检测跨页内容（表格/段落被截断到下一页）并标记

### 输出格式
返回严格的 JSON 对象（不要加 markdown 代码块标记）:

{{
  "page": {page_no},
  "verified_headings": [
    {{
      "text": "标题文字原样",
      "level": 2,
      "action": "keep",
      "reasoning": "编号体系1.1，承接上一页第一章"
    }}
  ],
  "cross_page_sections": [],
  "page_quality_comment": "结构正常，标题层级清晰"
}}

### verified_headings 字段说明
- text: 标题原文
- level: 1-6，承接上一页标题栈
- action: "keep"(确认正确) | "correct"(修正级别) | "add"(补充漏掉的标题) | "delete"(误判，删除)

### 编号体系判断规则
- 常见的文档编号体系: "1", "1.1", "1.1.1", "第一章", "一、", "(一)", "a)", "A."
- 标题级别必须与编号体系一致：子编号比父编号多一层
- 如果当前页开头有明显的编号（如 "3.1.2"），而上一页标题栈最后一个标题有 "3.1"，则当前标题 level 应为上一页标题 level + 1

### 跨页判断
- 如果本页顶部内容明显是上一页的延续（没有新标题开头），在 cross_page_sections 中标记
- 如果本页底部内容被截断（表格不完整/段落半句话），在 cross_page_sections 中标记
- cross_page_sections 格式: [{{"text_prefix": "前20字", "type": "continues"|"truncated"|"ends"}}]
"""


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class VerifiedHeading:
    """VLM 验证后的单个标题"""
    text: str
    level: int
    action: str = "keep"       # keep | correct | add | delete
    reasoning: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "VerifiedHeading":
        return cls(
            text=d.get("text", ""),
            level=d.get("level", 1),
            action=d.get("action", "keep"),
            reasoning=d.get("reasoning", ""),
        )


@dataclass
class CrossPageSection:
    """跨页标记"""
    text_prefix: str = ""
    type: str = ""             # continues | truncated | ends

    @classmethod
    def from_dict(cls, d: dict) -> "CrossPageSection":
        return cls(
            text_prefix=d.get("text_prefix", ""),
            type=d.get("type", ""),
        )


@dataclass
class PageRepairResult:
    """VLM 对单页的结构修复结果"""
    page: int
    verified_headings: list[VerifiedHeading] = field(default_factory=list)
    cross_page_sections: list[CrossPageSection] = field(default_factory=list)
    page_quality_comment: str = ""
    raw_vlm_response: str = ""  # 调试用
    success: bool = False

    @classmethod
    def from_dict(cls, d: dict, raw: str = "") -> "PageRepairResult":
        try:
            headings = [VerifiedHeading.from_dict(h)
                        for h in d.get("verified_headings", [])]
            cross = [CrossPageSection.from_dict(c)
                     for c in d.get("cross_page_sections", [])]
            return cls(
                page=d.get("page", 0),
                verified_headings=headings,
                cross_page_sections=cross,
                page_quality_comment=d.get("page_quality_comment", ""),
                raw_vlm_response=raw,
                success=True,
            )
        except Exception as e:
            logger.warning(f"[StructureRepairer] 解析 VLM 响应失败: {e}")
            return cls(page=d.get("page", 0), success=False)


# ═══════════════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════════════

class StructureRepairer:
    """VLM 结构修复器 — 注入全局上下文，修正 Docling 的结构错误。

    用法:
        repairer = StructureRepairer(model_client)
        results = await repairer.repair_pages(
            doc=doc,
            repair_pages=[3, 7, 12],  # QualityGate 标记的低质量页
            global_summary="...",
            prev_heading_stacks={3: [...], 7: [...], 12: [...]},
            next_first_headings={3: "...", 7: "...", 12: "..."},
            docling_headings={3: [...], 7: [...], 12: [...]},
            vlm_model="gpt-4o",
        )
    """

    def __init__(self, model_client: AIModelClient | None = None):
        self.model_client = model_client or AIModelClient()
        self._page_cache: dict[str, PageRepairResult] = {}

    # ── 公开 API ──────────────────────────────────────────────

    async def repair_pages(
        self,
        doc: DoclingDocument,
        repair_pages: list[int],
        global_summary: str,
        prev_heading_stacks: dict[int, list[str]],
        next_first_headings: dict[int, str],
        docling_headings: dict[int, list[dict]],
        vlm_model: str,
    ) -> dict[int, PageRepairResult]:
        """对指定页面进行 VLM 结构修复。

        Args:
            doc: DoclingDocument（用于获取页面图片）
            repair_pages: 需要修复的页码列表
            global_summary: 文档全局摘要
            prev_heading_stacks: 每页的上一个标题栈 {page_no: [heading_text, ...]}
            next_first_headings: 每页的下一个标题 {page_no: "text"}
            docling_headings: 每页 Docling 识别出的标题 [{text, level, bbox_hint}, ...]
            vlm_model: VLM 模型名称

        Returns:
            {page_no: PageRepairResult} 修复结果
        """
        total = len(doc.pages)
        results: dict[int, PageRepairResult] = {}
        start_time = time.time()

        for i, page_no in enumerate(repair_pages):
            page_obj = doc.pages.get(page_no)
            if not page_obj or not page_obj.image or not page_obj.image.pil_image:
                logger.warning(f"[StructureRepairer] 第 {page_no} 页无图片，跳过")
                continue

            elapsed = time.time() - start_time
            eta = (elapsed / max(i + 1, 1)) * (len(repair_pages) - i - 1)
            logger.info(
                f"[VLM Repair] 修复第 {page_no}/{total} 页 | "
                f"进度 {i+1}/{len(repair_pages)} | "
                f"预计剩余 {eta:.0f}s"
            )

            try:
                img_hash = self._hash_image(page_obj.image.pil_image)
                if img_hash and img_hash in self._page_cache:
                    results[page_no] = self._page_cache[img_hash]
                    continue

                page_start = time.time()
                result = await self._repair_single_page(
                    pil_image=page_obj.image.pil_image,
                    page_no=page_no,
                    total_pages=total,
                    global_summary=global_summary,
                    prev_stack=prev_heading_stacks.get(page_no, []),
                    next_heading=next_first_headings.get(page_no, ""),
                    docling_hds=docling_headings.get(page_no, []),
                    vlm_model=vlm_model,
                )

                results[page_no] = result
                if img_hash:
                    self._page_cache[img_hash] = result

                page_elapsed = time.time() - page_start
                logger.info(
                    f"[VLM Repair] 第 {page_no} 页完成 ({page_elapsed:.1f}s) | "
                    f"headings={len(result.verified_headings)} | "
                    f"cross_page={len(result.cross_page_sections)} | "
                    f"comment: {result.page_quality_comment[:40]}"
                )

            except Exception as e:
                logger.error(f"[VLM Repair] 第 {page_no} 页修复失败: {e}")
                results[page_no] = PageRepairResult(page=page_no, success=False)

        total_elapsed = time.time() - start_time
        success_count = sum(1 for r in results.values() if r.success)
        logger.success(
            f"[StructureRepairer] 完成: {success_count}/{len(repair_pages)} 页修复成功 | "
            f"总耗时 {total_elapsed:.0f}s"
        )
        return results

    # ── 内部实现 ──────────────────────────────────────────────

    async def _repair_single_page(
        self,
        pil_image: Any,
        page_no: int,
        total_pages: int,
        global_summary: str,
        prev_stack: list[str],
        next_heading: str,
        docling_hds: list[dict],
        vlm_model: str,
    ) -> PageRepairResult:
        """对单页调 VLM 进行结构修复"""
        # 构建 Docling 初步分析的文本描述
        if docling_hds:
            heading_lines = []
            for h in docling_hds:
                heading_lines.append(
                    f"  - level={h.get('level', 1)} \"{h.get('text', '')}\" "
                    f"(位置: {h.get('bbox_hint', '未知')})"
                )
            current_headings_text = "\n".join(heading_lines)
        else:
            current_headings_text = "  (Docling 未识别出任何标题)"

        # 格式化上一页标题栈
        prev_stack_str = " > ".join(prev_stack) if prev_stack else "(文档开头，无上级标题)"

        # 格式化下一页标题
        next_heading_str = next_heading or "(文档末尾，无下一页)"

        prompt = STRUCTURE_REPAIR_PROMPT.format(
            global_summary=global_summary,
            page_no=page_no,
            total_pages=total_pages,
            prev_heading_stack=prev_stack_str,
            next_first_heading=next_heading_str,
            current_page_headings=current_headings_text,
        )

        try:
            content = await self.model_client.get_vlm_answer(
                model_name=vlm_model,
                image=pil_image,
                prompt=prompt,
                temperature=0.1,
                max_tokens=2048,
            )

            if not content:
                return PageRepairResult(page=page_no, success=False)

            # 解析 JSON
            parsed = self._parse_json_safely(content)
            if parsed:
                return PageRepairResult.from_dict(parsed, raw=content)

            logger.warning(f"[StructureRepairer] 第 {page_no} 页 VLM 返回非 JSON: {content[:200]}...")
            return PageRepairResult(page=page_no, success=False)

        except Exception as e:
            logger.error(f"[StructureRepairer] 第 {page_no} 页 VLM 调用异常: {e}")
            return PageRepairResult(page=page_no, success=False)

    @staticmethod
    def _parse_json_safely(text: str) -> dict | None:
        """安全解析 JSON，处理 markdown code fence 包裹的情况"""
        import json as _json
        import re

        text = text.strip()

        # 去掉可能的 markdown code fence
        fence_match = re.match(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # 找到第一个 { 和最后一个 }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            return None

    @staticmethod
    def _hash_image(pil_image: Any) -> str:
        try:
            return hashlib.md5(pil_image.tobytes()).hexdigest()
        except Exception:
            return ""
