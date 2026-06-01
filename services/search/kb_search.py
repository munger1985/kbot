import asyncio
import time
import re
from typing import Any
from loguru import logger
from sqlalchemy import text
from core.exceptions import *
from core.database.oracle import get_session
from dao.repositories import TxtChunkRepository
from .result import TxtBaseSearchResult
from utils.oracle_vec_handler import OracleVecHandler


class TxtBaseSearch:
    """基于 Oracle 26ai 原生混合检索优化后的知识库检索服务"""

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
        tags: list[str] = []
    ) -> list[TxtBaseSearchResult]:  # ─── 统一契约：直接返回干净的 List ───
        """
        单一知识库混合检索入口（直接返回融合、去重、滑窗后的清洗结果列表）
        """
        start_time = time.time()

        # 1. 文本预处理（彻底规避特殊符号带来的语法解析崩溃）
        if not keywords or not keywords.strip():
            logger.warning(f"[TxtBaseSearch] 收到空的 keywords!")
            
        # ─── 核心修正：把所有非中英文字符、数字替换为空格，避免特殊符号（如 / 或 -）混入 ───
        clean_keyword = re.sub(r'[^\w\u4e00-\u9fa5]', ' ', keywords.strip() if keywords else "")
        
        # 过滤掉空字符串，拿到纯粹的干净词块
        words = [w.strip() for w in clean_keyword.split() if w.strip()]
        
        # 每一个词块用大括号包裹，词与词之间用 ACCUM 融合
        # 例如: "{在} ACCUM {GBT} ACCUM {446872024标准文档中}"
        formatted_key = " ACCUM ".join([f"{{{w}}}" for w in words]) if words else ""

        # 2. 向量状态检查
        if not query_vec:
            has_vec = 0
            vec_array = []
        else:
            has_vec = 1
            vec_handler = OracleVecHandler()
            vec_array = vec_handler.convert(vec=query_vec, to_string=False)

        # 3. 三倍池超配（为滑窗去重留出空间）
        over_fetch_k = search_top_k * 3

        # 4. 调用底层 Oracle 26ai 原生并行混合检索
        async with self.oracle_session as session:
            repo = TxtChunkRepository(session)
            try:
                dataset = await repo.native_hybrid_search(
                    kb_id=kb_id, keywords=formatted_key, query_vec=vec_array, security=security, # type: ignore
                    has_vec=has_vec, similarity_threshold=threshold, search_top_k=over_fetch_k, tags=tags
                )
            except DataNotFoundException:
                dataset = []
            except Exception as e:
                handle_exception(e, f"Hybrid search database error: {str(e)}")
                dataset = []

        # 5. 转换并注入业务层级加权（无 structure_level）
        raw_results = self._construct_search_result(dataset, weight=weight)

        # 6. 异步并行滑窗增强上下文 + 滑窗去重
        enhanced_results = await self._enhance_context_with_window(raw_results)
        final_results = self._merge_adjacent_chunks(enhanced_results, window_size=1)

        # 7. ─── 核心修改 ───
        # 丢弃原先的字典桶。直接裁剪出该知识库的 Top-K 结果列表返回
        final_pool = final_results[:search_top_k]
        
        logger.debug(f"Database hybrid retrieval completed for KB {kb_id}. Found {len(final_pool)} items.")
        return final_pool

    async def _enhance_context_with_window(self, initial_results: list[TxtBaseSearchResult], window_size: int = 1) -> list[TxtBaseSearchResult]:
        """滑窗增强：异步多会话并行获取上下文章节"""
        if not initial_results:
            return []

        async def expand_single_chunk(res: TxtBaseSearchResult):
            # 🔍 诊断日志：记录进入时的 chunk 关键信息
            logger.debug(
                f"[EnhanceWindow] 开始处理 chunk: chunk_id={res.chunk_id!r}, "
                f"file_id={res.file_id!r}, chunk_type={res.chunk_type!r}, "
                f"chunk_num={res.chunk_num}, kb_id={getattr(res, 'kb_id', 'N/A')!r}"
            )
            try:
                async with self.oracle_session as session:
                    repo = TxtChunkRepository(session)
                    if res.chunk_type == "text":
                        try:
                            neighbors = await repo.get_chunks_by_range(
                                file_id=res.file_id, 
                                center_chunk_num=res.chunk_num, 
                                window_size=window_size
                            )
                            if neighbors:
                                # 🔍 诊断日志：记录 neighbors 的键名
                                neighbor_keys = [list(n.keys()) for n in neighbors[:3]]
                                logger.debug(
                                    f"[EnhanceWindow] chunk {res.chunk_id!r} 获取到 {len(neighbors)} 个邻居, "
                                    f"前3个的 keys: {neighbor_keys!r}"
                                )
                                res.content = "\n---\n".join([c.get('content', "") for c in neighbors])
                        except Exception as e:
                            logger.error(
                                f"[EnhanceWindow] 获取邻居失败 chunk {res.chunk_id!r}: "
                                f"错误类型: {type(e).__name__}, 错误: {e}",
                                exc_info=True
                            )
            except Exception as outer_err:
                logger.error(
                    f"[EnhanceWindow] expand_single_chunk 外层异常 chunk {res.chunk_id!r}: "
                    f"错误类型: {type(outer_err).__name__}, 错误: {outer_err}",
                    exc_info=True
                )
                raise
            return res

        tasks = [expand_single_chunk(res) for res in initial_results]
        return list(await asyncio.gather(*tasks))

    def _construct_search_result(self, dataset: list, weight: float) -> list[TxtBaseSearchResult]:
        """解析数据库结果，移除 structure_level 加权逻辑"""
        results = []
        for item in dataset:
            try:
                if not isinstance(item, dict): 
                    continue
                
                meta = item.get("metadata") or {}
                base_score = float(item.get("score") or 0.0)
                
                # 只保留对图表的特殊类型 Boost 加权
                chunk_type = item.get("chunk_type", "text")
                type_boost = 1.1 if chunk_type in ["table", "picture"] else 1.0
                
                # 最终算分
                final_score = base_score * type_boost * weight

                result = TxtBaseSearchResult(
                    chunk_id=item.get("chunk_id", ""),
                    chunk_num=item.get("chunk_num", 0),
                    chunk_type=chunk_type,
                    file_id=item.get("file_id", ""),
                    kb_id=item.get("kb_id", ""),
                    content=item.get("content", ""),
                    header=item.get("header", ""),
                    doc_summary=item.get("doc_summary", ""),
                    search_helper=item.get("search_helper", ""),
                    page_num=int(meta.get("page_num") or 0),
                    image_name=meta.get("image_name") or "",
                    bbox=meta.get("bbox") or [],
                    score=final_score,
                    biz_metadata=item.get("biz_metadata") or {},
                    weight=weight,
                    rerank_score=0.0
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to construct result: {e}")
                continue
                
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _merge_adjacent_chunks(self, results: list[TxtBaseSearchResult], window_size: int = 1) -> list[TxtBaseSearchResult]:
        """滑窗去重：防止相邻切片同时挤进召回池"""
        if not results:
            return []
        
        results.sort(key=lambda x: x.score, reverse=True)
        final_results = []
        file_coverage = {}  # dict[file_id, set[chunk_num]]
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

            if not any(abs(existing_num - cnum) <= window_size for existing_num in file_coverage[fid]):
                final_results.append(res)
                file_coverage[fid].add(cnum)
                
        return final_results