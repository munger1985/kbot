import asyncio
import re
import time
import json
from typing import Any
from loguru import logger
from core.exceptions import *
from core.database.oracle import get_session
from dao.repositories import TxtChunkRepository
from .result import TxtBaseSearchResult
from utils.codec import OracleVecHandler

# ---------------------------------------------------------------------------
# Phase 3: 中文停用词表（常见高频无意义词）
# ---------------------------------------------------------------------------
_STOP_WORDS: set[str] = {
    "的", "是", "在", "了", "和", "就", "都", "也", "还", "要", "有", "被", "把",
    "对", "与", "及", "等", "或", "但", "而", "且", "所", "为", "以", "从", "到",
    "该", "这", "那", "其", "中", "上", "下", "不", "之", "能", "会", "可以",
    "一个", "这个", "那个", "哪些", "什么", "怎么", "如何", "为什么", "哪",
    "请", "问", "说", "看", "让", "用", "做", "作", "没", "好", "很", "太",
    "着", "过", "将", "已", "正", "再", "又", "向", "各", "每", "只", "去",
    "来", "出", "里", "后", "前", "时", "年", "月", "日", "之", "一", "多",
    "少", "几", "些", "更", "最", "非常", "比较", "特别", "主要", "其他",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "nor", "just", "because", "if", "then", "that",
    "this", "these", "those", "which", "who", "whom",
}


def _try_jieba_tokenize(text: str) -> list[str]:
    """尝试使用 jieba 分词；不可用时回退到简单空格分割。"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        logger.debug("[TxtBaseSearch] jieba 不可用，使用简单 split 分词")
        return text.split()


def _process_keywords(raw_keywords: str) -> str:
    """
    Phase 3: 改进的关键词处理流水线。

    步骤：
    1. 清洗特殊符号，保留中英文字符、数字和空格
    2. 中文分词（jieba 可用时）
    3. 停用词过滤
    4. 长度过滤（单字中文过滤，单字母英文过滤）
    5. 生成 Oracle Text ACCUM 查询串，长词赋予更高权重

    Returns:
        格式化后的 Oracle Text 查询串（空格分隔的词袋，不含大括号），
        供 TxtChunkRepo.full_text_search() 进一步包装 ACCUM 语法。
    """
    raw = raw_keywords.strip() if raw_keywords else ""
    if not raw:
        return ""

    # 1. 清洗：特殊符号 → 空格
    clean_text = re.sub(r'[^\w一-龥]', ' ', raw)

    # 2. 分词
    tokens = _try_jieba_tokenize(clean_text)

    # 3. 停用词 + 长度过滤
    filtered: list[str] = []
    for w in tokens:
        w = w.strip().lower()
        if not w:
            continue
        # 停用词
        if w in _STOP_WORDS:
            continue
        # 长度过滤：单字中文无意义，单字母英文也无意义
        if len(w) <= 1 and (ord(w[0]) > 127 or w.isalpha()):
            continue
        filtered.append(w)

    if not filtered:
        # 极端情况全部被过滤，回退到原始清洗结果
        fallback = [w.strip() for w in clean_text.split() if len(w.strip()) > 1]
        filtered = fallback if fallback else [clean_text.strip()]

    # 4. 去重（保留顺序）并拼接
    seen: set[str] = set()
    unique_words: list[str] = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)

    result = " ".join(unique_words)
    logger.debug(f"[TxtBaseSearch] 关键词处理: '{raw_keywords[:80]}...' → '{result[:120]}'")
    return result


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def _rrf_fusion(
    vector_results: list[dict[str, Any]],
    text_results: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """
    Phase 1: 两路独立检索结果通过 RRF 融合。

    公式：RRF_score(d) = Σ 1 / (k + rank_i(d))

    Args:
        vector_results: 向量检索结果列表
        text_results: 全文检索结果列表
        k: RRF 平滑参数，默认 60
        top_k: 融合后返回的最大结果数

    Returns:
        融合后的结果列表，每个 dict 的 score 已更新为 RRF 分数
    """
    if not vector_results and not text_results:
        return []
    if not vector_results:
        return text_results[:top_k]
    if not text_results:
        return vector_results[:top_k]

    rrf_scores: dict[str, float] = {}

    # 向量路贡献
    for rank, item in enumerate(vector_results, start=1):
        cid = item.get("chunk_id", "")
        if cid:
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

    # 全文路贡献
    for rank, item in enumerate(text_results, start=1):
        cid = item.get("chunk_id", "")
        if cid:
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

    # 按 RRF 分降序排列
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # 构建结果映射（保留两路中分数更高的那一条记录）
    result_map: dict[str, dict[str, Any]] = {}
    for item in vector_results + text_results:
        cid = item.get("chunk_id", "")
        if cid not in result_map:
            result_map[cid] = dict(item)
        else:
            existing_score = float(result_map[cid].get("score", 0.0))
            new_score = float(item.get("score", 0.0))
            if new_score > existing_score:
                result_map[cid] = dict(item)

    # 组装最终结果，用 RRF 分覆盖原始分
    fused: list[dict[str, Any]] = []
    for cid in sorted_ids[:top_k]:
        if cid in result_map:
            entry = result_map[cid]
            entry["score"] = rrf_scores[cid]
            entry["_rrf_score"] = rrf_scores[cid]
            fused.append(entry)

    logger.debug(
        f"[RRF] 融合完成: vector={len(vector_results)}, text={len(text_results)} → fused={len(fused)}"
    )
    return fused


# ---------------------------------------------------------------------------
# TxtBaseSearch
# ---------------------------------------------------------------------------

class TxtBaseSearch:
    """基于 Oracle 原生混合检索优化后的知识库检索服务。

    Phase 1 改造：两路独立检索（向量 + 全文）+ RRF 融合。
    """

    @property
    def oracle_session(self):
        """Returns a database session context manager."""
        return get_session()

    async def search(
        self,
        kb_id: int,
        keywords: str,
        search_top_k: int,
        threshold: float,
        weight: float,
        security: int,
        query_vec: list[float] | None = None,
        tags: list[str] = [],
    ) -> list[TxtBaseSearchResult]:
        """
        单一知识库混合检索入口（两路独立检索 + RRF 融合）。

        Args:
            kb_id: 知识库 ID
            keywords: LLM 提取的原始关键词
            search_top_k: 每路检索返回数量（全文/向量各取 search_top_k，RRF 合并后返回）
            threshold: 向量相似度阈值
            weight: 知识库业务权重乘数
            security: 安全等级上限
            query_vec: 查询嵌入向量（None 则只做全文检索）
            tags: 标签硬过滤
        """
        start_time = time.time()

        # ------------------------------------------------------------------
        # Phase 3: 改进的关键词清洗流水线
        # ------------------------------------------------------------------
        pure_keywords = _process_keywords(keywords)
        logger.debug(
            f"[TxtBaseSearch] KB {kb_id} 关键词: raw='{keywords[:80]}...' → processed='{pure_keywords[:120]}'"
        )

        # ------------------------------------------------------------------
        # 向量预处理
        # ------------------------------------------------------------------
        has_vec = query_vec is not None and len(query_vec) > 0
        vec_array: list = []
        if has_vec:
            vec_handler = OracleVecHandler()
            vec_array = vec_handler.convert(vec=query_vec, to_string=False)  # type: ignore[arg-type]

        # 每路召回量：各取 search_top_k，RRF 合并后直接返回
        per_route_k = search_top_k

        # ------------------------------------------------------------------
        # Phase 1: 两路独立并行检索（各自使用独立 session 避免并发冲突）
        # ------------------------------------------------------------------

        async def _do_vector_search() -> list[dict[str, Any]]:
            """独立 session 的向量检索"""
            try:
                async with self.oracle_session as session:
                    repo = TxtChunkRepository(session)
                    return await repo.vector_search(
                        kb_id=kb_id,
                        query_vec=vec_array,  # type: ignore[arg-type]
                        security=security,
                        similarity_threshold=threshold,
                        search_top_k=per_route_k,
                        tags=tags,
                    )
            except Exception as e:
                logger.error(f"[TxtBaseSearch] KB {kb_id} 向量检索异常: {e}")
                return []

        async def _do_text_search() -> list[dict[str, Any]]:
            """独立 session 的全文检索"""
            try:
                async with self.oracle_session as session:
                    repo = TxtChunkRepository(session)
                    return await repo.full_text_search(
                        kb_id=kb_id,
                        keywords=pure_keywords,
                        security=security,
                        search_top_k=per_route_k,
                        tags=tags,
                    )
            except Exception as e:
                logger.error(f"[TxtBaseSearch] KB {kb_id} 全文检索异常: {e}")
                return []

        vec_results: list[dict[str, Any]] = []
        text_results: list[dict[str, Any]] = []

        if has_vec and pure_keywords:
            vec_results, text_results = await asyncio.gather(
                _do_vector_search(), _do_text_search()
            )
        elif has_vec:
            vec_results = await _do_vector_search()
        elif pure_keywords:
            text_results = await _do_text_search()

        # ------------------------------------------------------------------
        # Phase 1: RRF 融合
        # ------------------------------------------------------------------
        fused_dataset = _rrf_fusion(
            vector_results=vec_results,
            text_results=text_results,
            k=60,
            top_k=search_top_k * 2,  # 两路各 search_top_k，最多 2*search_top_k 条不重复结果
        )

        if not fused_dataset:
            logger.warning(f"[TxtBaseSearch] KB {kb_id} 检索结果为空")
            return []

        # ------------------------------------------------------------------
        # 转换并注入业务层级加权
        # ------------------------------------------------------------------
        raw_results = self._construct_search_result(fused_dataset, weight=weight)

        # ------------------------------------------------------------------
        # Phase 4: 批量滑窗增强上下文 + 滑窗去重
        # ------------------------------------------------------------------
        enhanced_results = await self._enhance_context_with_window_batch(raw_results)
        final_results = self._merge_adjacent_chunks(enhanced_results, window_size=1)

        elapsed = time.time() - start_time
        logger.debug(
            f"[TxtBaseSearch] KB {kb_id} 检索完成: "
            f"vec={len(vec_results)}, text={len(text_results)}, "
            f"fused={len(fused_dataset)}, final={len(final_results)}, "
            f"elapsed={elapsed:.3f}s"
        )
        return final_results

    # ------------------------------------------------------------------
    # Phase 4: 批量滑窗查询
    # ------------------------------------------------------------------

    async def _enhance_context_with_window_batch(
        self, initial_results: list[TxtBaseSearchResult], window_size: int = 1
    ) -> list[TxtBaseSearchResult]:
        """
        批量滑窗增强：一次批量查询所有需要邻居的 chunk，避免 N+1 问题。

        对于 chunk_type == "text" 的结果，批量获取其前后 window_size 个邻居 chunk，
        拼接为更完整的上下文。
        """
        if not initial_results:
            return []

        # 收集所有需要查邻居的 chunk
        text_results = [r for r in initial_results if r.chunk_type == "text"]
        if not text_results:
            return initial_results

        # 构建批量查询参数
        queries = [(r.file_id, r.chunk_num) for r in text_results]

        try:
            async with self.oracle_session as session:
                repo = TxtChunkRepository(session)
                neighbors_map = await repo.get_chunks_by_ranges_batch(
                    queries=queries, window_size=window_size
                )
        except Exception as e:
            logger.error(
                f"[EnhanceWindow] 批量滑窗查询失败: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return initial_results

        # 将邻居内容写回结果
        for res in text_results:
            key = (res.file_id, res.chunk_num)
            neighbors = neighbors_map.get(key, [])
            if neighbors:
                # 按 chunk_num 排序后拼接
                neighbors_sorted = sorted(neighbors, key=lambda n: n.get("chunk_num", 0))
                joined = "\n---\n".join(
                    [n.get("content", "") for n in neighbors_sorted]
                )
                if joined:
                    res.content = joined

        return initial_results

    # ------------------------------------------------------------------
    # 原有方法（保留）
    # ------------------------------------------------------------------

    async def _enhance_context_with_window(
        self, initial_results: list[TxtBaseSearchResult], window_size: int = 1
    ) -> list[TxtBaseSearchResult]:
        """
        滑窗增强（逐条查询版本，保留兼容）。
        新代码请使用 _enhance_context_with_window_batch。
        """
        if not initial_results:
            return []

        async def expand_single_chunk(res: TxtBaseSearchResult):
            try:
                async with self.oracle_session as session:
                    repo = TxtChunkRepository(session)
                    if res.chunk_type == "text":
                        try:
                            neighbors = await repo.get_chunks_by_range(
                                file_id=res.file_id,
                                center_chunk_num=res.chunk_num,
                                window_size=window_size,
                            )
                            if neighbors:
                                res.content = "\n---\n".join(
                                    [c.get("content", "") for c in neighbors]
                                )
                        except Exception as e:
                            logger.error(
                                f"[EnhanceWindow] 获取邻居失败 chunk {res.chunk_id!r}: "
                                f"{type(e).__name__}: {e}",
                                exc_info=True,
                            )
            except Exception as outer_err:
                logger.error(
                    f"[EnhanceWindow] expand_single_chunk 外层异常 chunk {res.chunk_id!r}: "
                    f"{type(outer_err).__name__}: {outer_err}",
                    exc_info=True,
                )
                raise
            return res

        tasks = [expand_single_chunk(res) for res in initial_results]
        return list(await asyncio.gather(*tasks))

    def _construct_search_result(
        self, dataset: list[dict[str, Any]], weight: float
    ) -> list[TxtBaseSearchResult]:
        """解析数据库结果，应用业务加权"""
        results = []
        for item in dataset:
            try:
                if not isinstance(item, dict):
                    continue

                meta = item.get("metadata") or {}
                base_score = float(item.get("score") or 0.0)

                # 对表格和图片类型给予小幅 Boost
                chunk_type = item.get("chunk_type", "text")
                type_boost = 1.1 if chunk_type in ["table", "picture"] else 1.0

                final_score = base_score * type_boost * weight

                result = TxtBaseSearchResult(
                    chunk_id=item.get("chunk_id", ""),
                    chunk_num=item.get("chunk_num", 0),
                    chunk_type=chunk_type,
                    file_id=item.get("file_id", ""),
                    kb_id=int(item.get("kb_id", 0)),
                    content=item.get("content", ""),
                    header=item.get("header", ""),
                    doc_summary=item.get("doc_summary", ""),
                    search_helper=item.get("search_helper", ""),
                    page_num=int(meta.get("page_num") or 0),
                    image_name=meta.get("image_name") or "",
                    bbox=meta.get("bbox") or [],
                    hierarchy_path=json.loads(item.get("hierarchy_path", "[]")) if isinstance(item.get("hierarchy_path"), str) else (item.get("hierarchy_path") or []),
                    heading_level=int(item.get("heading_level") or 0),
                    section_id=item.get("section_id") or None,
                    score=final_score,
                    biz_metadata=item.get("biz_metadata") or {},
                    weight=weight,
                    rerank_score=0.0,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to construct result: {e}")
                continue

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _merge_adjacent_chunks(
        self, results: list[TxtBaseSearchResult], window_size: int = 1
    ) -> list[TxtBaseSearchResult]:
        """滑窗去重：防止相邻切片同时挤进召回池"""
        if not results:
            return []

        results.sort(key=lambda x: x.score, reverse=True)
        final_results: list[TxtBaseSearchResult] = []
        file_coverage: dict[str, set[int]] = {}
        MIN_KEEP_COUNT = 10

        for res in results:
            fid = res.file_id
            cnum = res.chunk_num

            if fid not in file_coverage:
                file_coverage[fid] = set()

            if len(final_results) < MIN_KEEP_COUNT:
                final_results.append(res)
                file_coverage[fid].add(cnum)
                continue

            if not any(
                abs(existing_num - cnum) <= window_size
                for existing_num in file_coverage[fid]
            ):
                final_results.append(res)
                file_coverage[fid].add(cnum)

        return final_results
