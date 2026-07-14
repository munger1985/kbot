"""ParadeDB 知识库混合检索服务 — BM25 + pgvector + RRF"""

import time
import asyncio
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from dao.repositories import TxtChunkRepository
from .result import TxtBaseSearchResult


class TxtBaseSearch:
    """基于 ParadeDB 的知识库检索服务。

    使用 pg_bm25 + pgvector + RRF 替代 ES 混合检索。
    """

    async def search(
        self,
        session: AsyncSession,
        kb_id: str,
        keywords: str,
        search_top_k: int,
        threshold: float,
        weight: float,
        security: int,
        query_vec: list[float] | None = None,
        tags: list[str] = [],
        window_size: int = 1
    ) -> list[TxtBaseSearchResult]:
        """执行混合检索并返回增强后的结果。"""
        start_time = time.time()
        logger.info(f"开始知识库检索: kb_id={kb_id}, keywords='{keywords}', mode={'混合' if query_vec else '纯文本'}")

        repo = TxtChunkRepository(session, kb_id)

        try:
            dataset = await repo.hybrid_search(
                keywords=keywords,
                security=security,
                query_vec=query_vec,
                search_top_k=search_top_k * 2,
                tags=tags
            )
        except Exception as e:
            logger.error(f"混合检索阶段发生系统异常: {str(e)}")
            return []

        if not dataset:
            logger.info(f"检索结束：未找到匹配内容 (kb_id={kb_id})")
            return []

        # 映射为业务对象
        initial_results = self._construct_search_result(dataset, weight=weight)
        logger.debug(f"原始召回数量: {len(initial_results)}")

        # Section 级上下文扩展
        if getattr(self, '_enable_section_context', True):
            enhanced_results = await self._enhance_context_by_section(session, kb_id, initial_results)
        else:
            enhanced_results = await self._enhance_context_with_window(session, kb_id, initial_results, window_size)

        # 基于覆盖范围的智能合并与冗余剔除
        final_result = self._merge_adjacent_chunks(enhanced_results, window_size=window_size)

        duration = time.time() - start_time
        logger.info(f"检索完成: 耗时={duration:.4f}s, 最终返回={len(final_result[:search_top_k])}条")
        return final_result[:search_top_k]

    async def _enhance_context_by_section(
        self,
        session: AsyncSession,
        kb_id: str,
        results: list[TxtBaseSearchResult],
    ) -> list[TxtBaseSearchResult]:
        """Section 级上下文扩展"""
        repo = TxtChunkRepository(session, kb_id)
        section_ids: set[str] = set()
        for r in results:
            sid = getattr(r, 'section_id', None)
            if sid:
                section_ids.add(sid)

        if not section_ids:
            return await self._enhance_context_with_window(session, kb_id, results, 1)

        section_contents = await repo.get_chunks_by_section_ids(section_ids=list(section_ids))

        for r in results:
            sid = getattr(r, 'section_id', None)
            if sid and sid in section_contents:
                sorted_chunks = sorted(
                    section_contents[sid], key=lambda c: c.get("chunk_num", 0)
                )
                r.content = "\n\n".join([c.get("content", "") for c in sorted_chunks])
        return results

    async def _enhance_context_with_window(
        self,
        session: AsyncSession,
        kb_id: str,
        initial_results: list[TxtBaseSearchResult],
        window_size: int = 1
    ) -> list[TxtBaseSearchResult]:
        """并行扩展所有检索结果的上下文窗口"""
        if not initial_results:
            return []
        repo = TxtChunkRepository(session, kb_id)
        tasks = [self._expand_single_chunk(repo, res, window_size) for res in initial_results]
        return list(await asyncio.gather(*tasks))

    async def _expand_single_chunk(
        self, repo: TxtChunkRepository, res: TxtBaseSearchResult, window_size: int = 1
    ) -> TxtBaseSearchResult:
        """对单个切片进行前后文拉取与内容拼接"""
        if res.chunk_type == "text":
            try:
                neighbors = await repo.get_chunks_by_range(
                    file_id=res.file_id,
                    center_chunk_num=res.chunk_num,
                    window_size=window_size
                )
                if neighbors:
                    res.content = "\n---\n".join([c.get('content', "") for c in neighbors])
            except Exception as e:
                logger.error(f"切片 {res.chunk_id} 扩展上下文失败: {str(e)}")
        return res

    def _construct_search_result(self, dataset: list, weight: float) -> list[TxtBaseSearchResult]:
        """将 ParadeDB 返回的原始数据映射为 TxtBaseSearchResult 业务对象"""
        results = []
        for item in dataset:
            try:
                meta = item.get("chunk_metadata") or {}
                result = TxtBaseSearchResult(
                    chunk_id=item.get("chunk_id", ""),
                    chunk_num=item.get("chunk_num", 0),
                    chunk_type=item.get("chunk_type", "text"),
                    file_id=item.get("file_id", ""),
                    kb_id=item.get("kb_id", ""),
                    content=item.get("content", ""),
                    header=item.get("header", ""),
                    doc_summary=item.get("doc_summary", ""),
                    search_helper=item.get("search_helper", ""),
                    page_num=int(meta.get("page_num") or 0),
                    image_name=meta.get("image_name") or "",
                    bbox=meta.get("bbox") or [],
                    hierarchy_path=item.get("hierarchy_path", []),
                    heading_level=item.get("heading_level", 0),
                    section_id=item.get("section_id"),
                    score=float(item.get("score") or 0.0),
                    weight=weight,
                    rerank_score=0.0,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"映射检索结果失败 (ID: {item.get('chunk_id')}): {str(e)}")
        return results

    def _merge_adjacent_chunks(self, results: list[TxtBaseSearchResult], window_size: int = 1) -> list[TxtBaseSearchResult]:
        """改进的去重策略"""
        if not results:
            return []
        results.sort(key=lambda x: x.score, reverse=True)
        final_results = []
        file_coverage = {}
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

            is_redundant = False
            for existing_num in file_coverage[fid]:
                if abs(existing_num - cnum) <= window_size:
                    is_redundant = True
                    break

            if not is_redundant:
                final_results.append(res)
                file_coverage[fid].add(cnum)

        return final_results
