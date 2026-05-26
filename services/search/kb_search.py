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
        do_rerank: bool,
        weight: float,
        security: int, 
        query_vec: list[float] | None = None,
        tags: list[str] = []
    ) -> dict[str, list[TxtBaseSearchResult]]:
        """执行 Oracle 26ai 原生内核级混合检索 (Vector + FullText)"""
        start_time = time.time()
        logger.debug("Starting Oracle 26ai native hybrid search...")

        # 1. 文本关键词清洗与多词 ACCUM 语法转换
        clean_keyword = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', keywords.strip())
        words = [w for w in clean_keyword.split() if w]
        formatted_key = " ACCUM ".join([f"{{{w}}}" for w in words]) if words else ""

        # 2. 向量参数降级防御检查
        if not query_vec:
            logger.warning("Query vector is missing; searching via full-text paths only.")
            has_vec = 0
            vec_array = []
        else:
            has_vec = 1
            vec_handler = OracleVecHandler()
            vec_array = vec_handler.convert(vec=query_vec, to_string=False)

        # 3. 提取 3 倍超配池以确保上下文合并与重排时的质量
        over_fetch_k = search_top_k * 3

        # 4. 执行多路并发融合检索
        async with self.oracle_session as session:
            repo = TxtChunkRepository(session)
            try:
                # 统一调用仓库重构后的单路原生混合查询
                dataset = await repo.native_hybrid_search(
                    kb_id=kb_id,
                    keywords=formatted_key,
                    query_vec=vec_array, # type: ignore
                    security=security,
                    has_vec=has_vec,
                    similarity_threshold=threshold,
                    search_top_k=over_fetch_k,
                    tags=tags
                )
                logger.debug(f"Database hybrid retrieval completed. Found {len(dataset)} items.")
            except DataNotFoundException:
                dataset = []
            except Exception as e:
                handle_exception(e, f"Oracle 26ai hybrid search failed for KB {kb_id}: {str(e)}")
                dataset = []

        # 5. 对象映射与权重调整（保留你原本对层级和特殊类型的 Boost 机制）
        raw_results = self._construct_search_result(dataset, weight=weight)

        # 6. 邻近块滑动窗口增强与滑动窗去重
        enhanced_results = await self._enhance_context_with_window(raw_results)
        final_results = self._merge_adjacent_chunks(enhanced_results, window_size=1)

        # 7. 截断到最终需要的 Top-K
        final_pool = final_results[:search_top_k]

        duration = time.time() - start_time
        logger.info(f"Hybrid search processing done in {duration:.2f}s. Pool size: {len(final_pool)}")

        # 8. 归流到 rerank / norerank 桶中
        bucket_key = "rerank_result" if do_rerank else "norerank_result"
        return {
            bucket_key: final_pool,
            "norerank_result" if do_rerank else "rerank_result": []
        }

    async def _enhance_context_with_window(self, initial_results: list[TxtBaseSearchResult], window_size: int = 1) -> list[TxtBaseSearchResult]:
        """滑窗增强：异步多会话并行获取上下文章节"""
        if not initial_results:
            return []

        async def expand_single_chunk(res: TxtBaseSearchResult):
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
                            res.content = "\n---\n".join([c.get('content', "") for c in neighbors])
                    except Exception as e:
                        logger.error(f"Failed to fetch neighbors for chunk {res.chunk_id}: {e}")
                return res

        tasks = [expand_single_chunk(res) for res in initial_results]
        return list(await asyncio.gather(*tasks))

    def _construct_search_result(self, dataset: list, weight: float) -> list[TxtBaseSearchResult]:
        """解析数据库结果，并揉入原本的业务 Boost 算分规则"""
        results = []
        for item in dataset:
            try:
                if not isinstance(item, dict): 
                    continue
                
                meta = item.get("metadata") or {}
                base_score = float(item.get("score") or 0.0)
                
                # 迁移原本 RRF 中的业务 Boost 算分规则
                struct_lvl = int(item.get("structure_level") or 0)
                level_boost = 1.5 if (0 < struct_lvl < 3) else 1.0
                
                chunk_type = item.get("chunk_type", "text")
                type_boost = 1.1 if chunk_type in ["table", "picture"] else 1.0
                
                final_score = base_score * level_boost * type_boost * weight

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