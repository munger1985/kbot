"""精准引擎 — VLM 逐页 Markdown 生成模块。

每页文档图片 → VLM → Markdown（含标题层级、表格、图表描述）。
不再输出 JSON 中间格式，消除 TextMapper 对齐裂缝。
"""
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from docling_core.types.doc.document import DoclingDocument

from platform_clients import AIModelClient


# ═══════════════════════════════════════════════════════════════
# VLM Prompt — 视觉优先解析
# ═══════════════════════════════════════════════════════════════

PAGE_MARKDOWN_PROMPT = """你是一个专业的文档解析器。将这张文档页面转换为结构化 Markdown。

## 规则
1. **原文照录**：所有文字（中英文、数字、符号）原样保留，不翻译不改写
2. **英文连贯**：英文单词之间保留自然空格。禁止逐字母拆分。
   正确：Mean time between failures
   错误：M e a n t i m e b e t w e e n f a i l u r e s
3. **表格**：用 Markdown 表格还原。保留所有行列、数字、单位、符号。
   复杂表格（合并单元格）需在表头用括号标注 colspan/rowspan
4. **标题层级**：用 # ## ### #### 表示。识别编号体系（1 / 1.1 / 6.3.2 / (a) / ①）
5. **图表描述**：用 > 引用块描述图表/流程图的逻辑含义，不仅描述外观：
   > **图X 设备可靠性框图**：系统由4个串行模块组成，第3个为冗余备份...
6. **图片提取标记**：页面中值得单独提取的图片/图表处插入 [IMAGE:简短描述]
   例如：[IMAGE:表A.4 多制程集群设备时间映射]
7. **列表与公式**：保留原始编号格式，数学公式用 $$ 包裹
8. **直接输出 Markdown**：不加 json 包装、不加代码块标记、不加解释

## 页面类型判断
- **封面页**：保留完整标题（中英文），标准号，发布机构
- **目录页**：保留完整条目和页码，格式：标题（页码: 数字）
- **术语定义页**：每个术语用 **术语名**：定义内容 的格式
- **正文页**：标题层级 + 段落 + 表格 + 图表描述
- **附录页**：同上

## 禁止行为
- 禁止在英文单词字母间插入额外空格
- 禁止省略表格中的行列
- 禁止编造不存在的内容
- 禁止把中文和英文混排成乱码
- 空白页面输出空内容"""


PAGE_BLANK_CHECK_PROMPT = """快速判断：这张页面是否有有意义的文档内容？

- 如果页面是空白的、只有背景/水印/装饰 → 回复 "BLANK"
- 如果页面有任何文字、表格、图表等有意义内容 → 回复 "CONTENT"

只回复 BLANK 或 CONTENT。"""


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class PageMarkdown:
    """VLM 生成的单页 Markdown"""
    page: int
    markdown: str = ""
    meaningful: bool = True  # 是否有意义内容


# ═══════════════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════════════

class PrecisionAnalyzer:
    """VLM 逐页 Markdown 生成器"""

    def __init__(self, model_client: AIModelClient | None = None):
        self.model_client = model_client or AIModelClient()
        self._page_cache: dict[str, str] = {}  # hash → markdown

    # ── 公开 API ──────────────────────────────────────────────

    async def analyze_document(
        self,
        doc: DoclingDocument,
        vlm_model: str,
        file_id: str = "",
    ) -> list[PageMarkdown]:
        """逐页生成 Markdown。

        Returns:
            按页码排序的 PageMarkdown 列表（含 meaningful 过滤）
        """
        pages: list[PageMarkdown] = []
        page_nums = sorted(doc.pages.keys())
        total = len(page_nums)
        start_time = time.time()

        for i, page_no in enumerate(page_nums):
            page_obj = doc.pages.get(page_no)
            if not page_obj or not page_obj.image or not page_obj.image.pil_image:
                pages.append(PageMarkdown(page=page_no, markdown="", meaningful=False))
                continue

            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            logger.info(
                f"[VLM] 分析第 {page_no}/{total} 页 | "
                f"已耗时 {elapsed:.0f}s | 预计剩余 {eta:.0f}s"
            )

            try:
                page_start = time.time()

                # 先做空白页检测（低成本，快速跳过）
                img_hash = self._hash_image(page_obj.image.pil_image)
                if img_hash in self._page_cache:
                    md = self._page_cache[img_hash]
                else:
                    md = await self._generate_page_markdown(
                        page_no, page_obj.image.pil_image, vlm_model
                    )
                    self._page_cache[img_hash] = md

                meaningful = bool(md.strip())
                page_elapsed = time.time() - page_start

                pages.append(PageMarkdown(
                    page=page_no,
                    markdown=md,
                    meaningful=meaningful,
                ))
                logger.info(
                    f"[VLM] 第 {page_no} 页完成 ({page_elapsed:.1f}s) | "
                    f"markdown={len(md)} chars | "
                    f"{'有内容' if meaningful else '空白/无意义'} | "
                    f"进度 {i+1}/{total}"
                )
            except Exception as e:
                logger.error(f"[VLM] 第 {page_no} 页分析失败: {e}")
                pages.append(PageMarkdown(page=page_no, markdown="", meaningful=False))

        total_elapsed = time.time() - start_time
        content_pages = sum(1 for p in pages if p.meaningful)
        logger.success(
            f"[PrecisionAnalyzer] 完成 {total} 页 ({content_pages} 有内容) | "
            f"总耗时 {total_elapsed:.0f}s, 平均 {total_elapsed/total:.1f}s/页"
        )
        return pages

    # ── 内部实现 ──────────────────────────────────────────────

    async def _generate_page_markdown(
        self, page_no: int, pil_image: Any, vlm_model: str
    ) -> str:
        """调 VLM 生成单页 Markdown"""
        try:
            content = await self.model_client.get_vlm_answer(
                model_name=vlm_model,
                image=pil_image,
                prompt=PAGE_MARKDOWN_PROMPT,
                temperature=0.1,
                max_tokens=4096,
            )
            if content:
                return content.strip()

        except Exception as e:
            logger.error(f"[VLM] 第 {page_no} 页生成失败: {e}")

        return ""

    @staticmethod
    def _hash_image(pil_image: Any) -> str:
        try:
            return hashlib.md5(pil_image.tobytes()).hexdigest()
        except Exception:
            return ""
