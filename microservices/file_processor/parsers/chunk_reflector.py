"""Chunk 反射器 — LLM 后反思重组短 chunk。

在 SmartChunker 生成所有 chunk 后，检测连续短 chunk 组，
通过 LLM 判断是否应合并为语义更完整的大块。

触发条件（AND）:
1. 连续的 chunk 同属一个 section（section_id 相同或 hierarchy_path 相同）
2. 每个 chunk 内容长度 < SHORT_CHUNK_THRESHOLD（默认 200 字）
3. 连续数量 ≥ MIN_GROUP_SIZE（至少 2 个）

策略:
- merge: LLM 判断应合并 → 合并为一个 chunk，更新 content/search_helper/header
- enrich: LLM 判断不应合并 → 为每个 chunk 补充 cross_ref（邻居关联摘要）
"""

import json
import asyncio
from dataclasses import dataclass, field

from loguru import logger

from ..parser_schema import ChunkResult, ChunkMetadata
from utils.clients import AIModelClient


# ── 阈值 ───────────────────────────────────────────────────
SHORT_CHUNK_THRESHOLD = 200    # 短 chunk 判定阈值（字符数）
MIN_GROUP_SIZE = 2             # 最少连续短 chunk 数才触发
MAX_GROUP_SIZE = 15            # 一次 LLM 调用最多处理的 chunk 数
MAX_CONCURRENT_REFLECT = 3     # 并发的 LLM 反思调用数


# ═══════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════

MERGE_REFLECTION_PROMPT = """你是一个文档结构优化专家。以下是从文档同一章节中切出的连续段落片段：

### 文档上下文
文档摘要：{global_summary}
所属章节：{hierarchy_path}

### 待判断的片段
{chunk_list}

### 判断规则
1. 如果这些片段属于「同类项罗列」（术语定义列表、参数规格表、步骤序列、枚举条款），应**合并**为一个完整 chunk
2. 如果片段之间逻辑独立（不同主题、不同层级、无关内容），应**保持独立**，但为每条补充一条关联上下文
3. 合并后保留原有结构（编号、缩进、空行）
4. 不要改写原文，只做合并或补充 cross_ref

### 输出格式
返回严格的 JSON 对象（不要加代码块标记）:

{{
  "action": "merge",
  "merged_content": "合并后的完整内容",
  "reason": "为什么合并（1句话）"
}}

或

{{
  "action": "enrich",
  "enriched": [
    {{"index": 0, "cross_ref": "前置：无；后置：术语定义 No.22，衬底/芯片相关"}},
    {{"index": 1, "cross_ref": "前置：附属工具定义 No.21；后置：生产单元定义 No.23"}}
  ],
  "reason": "为什么不合并（1句话）"
}}
"""


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class ReflectionResult:
    """一次 LLM 反思的结果"""
    action: str = ""                       # "merge" | "enrich"
    merged_content: str = ""
    enriched: list[dict] = field(default_factory=list)
    reason: str = ""


@dataclass
class MergeGroup:
    """一组需要反思的相邻 chunk"""
    start_idx: int
    end_idx: int
    section_id: str
    hierarchy_path: list[str]


# ═══════════════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════════════

class ChunkReflector:
    """Chunk 后反思重组器。

    用法:
        reflector = ChunkReflector(global_summary="...", llm_model="gpt-4o")
        refined = await reflector.reflect(chunks)
    """

    def __init__(
        self,
        global_summary: str = "",
        llm_model: str = "",
        model_client: AIModelClient | None = None,
    ):
        self.global_summary = global_summary
        self.llm_model = llm_model
        self.model_client = model_client or AIModelClient()

    # ── 公开 API ──────────────────────────────────────────

    async def reflect(
        self,
        chunks: list[ChunkResult],
    ) -> list[ChunkResult]:
        """主入口：检测并修复碎片化 chunk。

        Args:
            chunks: SmartChunker 输出的 chunk 列表（按 chunk_num 排序）

        Returns:
            重组后的 chunk 列表（chunk_num 已重新编号）
        """
        if not self.llm_model:
            logger.info("[ChunkReflector] LLM 模型未配置，跳过反思")
            return chunks

        # 1. 找出所有候选合并组
        groups = self._find_merge_candidates(chunks)
        if not groups:
            logger.info(
                f"[ChunkReflector] {len(chunks)} chunks 无碎片化问题，跳过"
            )
            return chunks

        logger.info(
            f"[ChunkReflector] 发现 {len(groups)} 组候选碎片 "
            f"(共 {sum(g.end_idx - g.start_idx + 1 for g in groups)} chunk 需要反思)"
        )

        # 2. 并发 LLM 反思
        sem = asyncio.Semaphore(MAX_CONCURRENT_REFLECT)
        tasks = [self._reflect_group(g, chunks, sem) for g in groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. 应用结果
        refined = self._apply_reflections(chunks, groups, results)

        logger.success(
            f"[ChunkReflector] 完成: {len(chunks)} → {len(refined)} chunks "
            f"(合并 {len(groups)} 组, 减少 {len(chunks) - len(refined)} chunks)"
        )
        return refined

    # ── 候选组发现 ────────────────────────────────────────

    @staticmethod
    def _find_merge_candidates(
        chunks: list[ChunkResult],
    ) -> list[MergeGroup]:
        """扫描所有 chunk，找出连续短 chunk 组。

        使用滑动窗口：当连续 chunk 满足短且同 section 时，继续扩展。
        """
        groups: list[MergeGroup] = []
        i = 0
        while i < len(chunks):
            c = chunks[i]

            # 跳过非 text 类型（table/picture/slide 不参与合并）
            if c.chunk_type != "text":
                i += 1
                continue

            # 跳过不短的 chunk
            if len(c.content) >= SHORT_CHUNK_THRESHOLD:
                i += 1
                continue

            # 找到连续短 chunk 组的末尾
            j = i + 1
            while j < len(chunks) and j - i < MAX_GROUP_SIZE:
                nc = chunks[j]
                if nc.chunk_type != "text":
                    break
                if len(nc.content) >= SHORT_CHUNK_THRESHOLD:
                    break
                # 同组判定：
                # 1) section_id 相同（同一 section 内），或
                # 2) 同父标题（hierarchy_path 去掉最后一项后相同，即兄弟 section）
                if not ChunkReflector._same_group(c, nc):
                    break
                j += 1

            group_size = j - i
            if group_size >= MIN_GROUP_SIZE:
                section_id = c.section_id or ""
                hpath = c.hierarchy_path or []
                groups.append(MergeGroup(
                    start_idx=i,
                    end_idx=j - 1,
                    section_id=section_id,
                    hierarchy_path=hpath,
                ))

            i = j

        return groups

    @staticmethod
    def _same_group(a: ChunkResult, b: ChunkResult) -> bool:
        """判断两个 chunk 是否属于同一可合并组。

        满足任一条件即视为同组：
        1. section_id 相同（同一 section 内的多个 chunk）
        2. hierarchy_path 去掉最后一项后相同（兄弟 section，如同一个"3 术语和定义"
           下的 3.1, 3.2, 3.3... 各自是独立 section 但语义上属于同组术语）
        """
        if a.section_id and b.section_id:
            if a.section_id == b.section_id:
                return True
        if a.hierarchy_path and b.hierarchy_path:
            if a.hierarchy_path == b.hierarchy_path:
                return True
            # 兄弟 section：去掉最后一项后相同
            pa = a.hierarchy_path[:-1] if len(a.hierarchy_path) > 1 else a.hierarchy_path
            pb = b.hierarchy_path[:-1] if len(b.hierarchy_path) > 1 else b.hierarchy_path
            if pa == pb and len(pa) > 0:
                return True
        return False

    # ── LLM 反思 ──────────────────────────────────────────

    async def _reflect_group(
        self,
        group: MergeGroup,
        chunks: list[ChunkResult],
        sem: asyncio.Semaphore,
    ) -> ReflectionResult | None:
        """对一组 chunk 进行 LLM 反思"""

        async def _run():
            group_chunks = chunks[group.start_idx:group.end_idx + 1]

            # 构建 chunk 列表文本
            chunk_lines = []
            for offset, c in enumerate(group_chunks):
                chunk_lines.append(
                    f"### 片段 {offset}\n"
                    f"标题: {c.header}\n"
                    f"内容: {c.content}\n"
                )
            chunk_list_text = "\n".join(chunk_lines)

            hierarchy_str = " > ".join(group.hierarchy_path) if group.hierarchy_path else "未知章节"

            prompt = MERGE_REFLECTION_PROMPT.format(
                global_summary=self.global_summary,
                hierarchy_path=hierarchy_str,
                chunk_list=chunk_list_text,
            )

            try:
                full_response = await self.model_client.get_llm_answer(
                    model_name=self.llm_model,
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=2048,
                )

                parsed = self._parse_json_safely(full_response)
                if not parsed:
                    logger.warning(
                        f"[ChunkReflector] LLM 返回非 JSON: "
                        f"chunks[{group.start_idx}:{group.end_idx}] → {full_response[:100]}..."
                    )
                    return None

                return ReflectionResult(
                    action=parsed.get("action", ""),
                    merged_content=parsed.get("merged_content", ""),
                    enriched=parsed.get("enriched", []),
                    reason=parsed.get("reason", ""),
                )

            except Exception as e:
                logger.error(
                    f"[ChunkReflector] LLM 调用失败: "
                    f"chunks[{group.start_idx}:{group.end_idx}]: {e}"
                )
                return None

        async with sem:
            return await _run()

    @staticmethod
    def _parse_json_safely(text: str) -> dict | None:
        """安全解析 JSON（处理 markdown code fence）"""
        import re

        text = text.strip()
        fence_match = re.match(
            r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL
        )
        if fence_match:
            text = fence_match.group(1).strip()

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # ── 应用反思结果 ──────────────────────────────────────

    @staticmethod
    def _apply_reflections(
        chunks: list[ChunkResult],
        groups: list[MergeGroup],
        results: list,
    ) -> list[ChunkResult]:
        """将 LLM 反思结果应用到 chunk 列表。

        注意：需要从后往前处理，避免索引偏移。
        对每个组：
        - merge → 用一个合并后的 chunk 替换整组
        - enrich → 为每个 chunk 的 content 追加 cross_ref
        """
        # 按 start_idx 降序排列，从后往前处理
        sorted_pairs = sorted(
            zip(groups, results),
            key=lambda pair: pair[0].start_idx,
            reverse=True,
        )

        for group, result in sorted_pairs:
            if result is None or isinstance(result, BaseException):
                continue

            rr: ReflectionResult = result
            if not rr or not rr.action:
                continue

            if rr.action == "merge" and rr.merged_content:
                ChunkReflector._apply_merge(chunks, group, rr)
            elif rr.action == "enrich" and rr.enriched:
                ChunkReflector._apply_enrich(chunks, group, rr)

        # 重新编号
        for i, c in enumerate(chunks):
            c.chunk_num = i + 1

        return chunks

    @staticmethod
    def _apply_merge(
        chunks: list[ChunkResult],
        group: MergeGroup,
        rr: ReflectionResult,
    ):
        """将一组 chunk 合并为一个"""
        group_chunks = chunks[group.start_idx:group.end_idx + 1]
        if not group_chunks:
            return

        # 继承第一个 chunk 的元数据
        first = group_chunks[0]
        last = group_chunks[-1]

        merged_header = first.header
        hierarchy_path = first.hierarchy_path or group.hierarchy_path
        hierarchy_str = " > ".join(hierarchy_path) if hierarchy_path else ""

        merged_search_helper = (
            f"文档: {first.doc_summary}\n"
            f"章节: {hierarchy_str}\n"
            f"段落: {merged_header}\n"
            f"内容: {rr.merged_content[:500]}"
        )

        merged = ChunkResult.create(
            content=rr.merged_content,
            summary=first.doc_summary,
            header=merged_header,
            search_helper=merged_search_helper,
            chunk_num=first.chunk_num,
            chunk_type="text",
            metadata=ChunkMetadata(
                page_num=first.metadata.page_num,
                image_name=None,
                bbox=None,
            ),
            hierarchy_path=hierarchy_path,
            hierarchy_depth=len(hierarchy_path),
            heading_level=first.heading_level,
            section_id=group.section_id,
        )

        # 替换
        chunks[group.start_idx:group.end_idx + 1] = [merged]
        logger.debug(
            f"[ChunkReflector] merge chunks[{group.start_idx}:{group.end_idx}] "
            f"({len(group_chunks)}→1): {rr.reason}"
        )

    @staticmethod
    def _apply_enrich(
        chunks: list[ChunkResult],
        group: MergeGroup,
        rr: ReflectionResult,
    ):
        """为每个 chunk 追加 cross_ref"""
        for item in rr.enriched:
            offset = item.get("index", -1)
            cross_ref = item.get("cross_ref", "")
            if not cross_ref:
                continue

            idx = group.start_idx + offset
            if 0 <= idx < len(chunks):
                c = chunks[idx]
                c.content = f"{c.content}\n\n[关联上下文]: {cross_ref}"
                logger.debug(
                    f"[ChunkReflector] enrich chunk[{idx}]: {cross_ref[:60]}..."
                )
