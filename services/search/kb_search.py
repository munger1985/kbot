import asyncio
import time
from loguru import logger
from core.exceptions import *
from core.database.oracle import get_session
from dao.repositories import TxtChunkRepository
from .result import TxtBaseSearchResult


class TxtBaseSearch:
    """Text-based knowledge base search service.
    
    Provides hybrid retrieval capabilities including vector similarity,
    full-text keyword matching, and structural context enhancement.
    """

    @property
    def oracle_session(self):
        """Returns a database session context manager."""
        return get_session()

    def rrf_merge(self, search_top_k: int, weight: float, 
              results_list: list[list[TxtBaseSearchResult]]) -> list[TxtBaseSearchResult]:
        k = 60
        rrf_score_map = {} 
        chunk_map = {}
        
        # 统计一下收到的原始数据总量
        total_raw_count = sum(len(rs) for rs in results_list)
        if total_raw_count == 0:
            return []

        for results in results_list:
            logger.debug(f"Processing result set with {len(results)} items")
            # rank 从 1 开始
            for rank, r in enumerate(results, 1):
                # 这里的 cid 必须是全局唯一的，否则会发生覆盖
                cid = r.chunk_id
                current_rrf = 1.0 / (k + rank)
                
                rrf_score_map[cid] = rrf_score_map.get(cid, 0.0) + current_rrf
                # 只有第一次遇到这个 chunk 时才存入 map，或者保留分值更高的那个属性
                if cid not in chunk_map:
                    chunk_map[cid] = r

        merged = []
        for cid, base_rrf_score in rrf_score_map.items():
            res = chunk_map[cid]
            
            # 权重计算
            level_boost = 1.5 if (0 < getattr(res, 'structure_level', 0) < 3) else 1.0
            type_boost = 1.1 if getattr(res, 'chunk_type', '') in ["table", "picture"] else 1.0
            
            # 赋值最终分数
            res.score = base_rrf_score * level_boost * type_boost * weight
            merged.append(res)
        
        # 排序
        merged.sort(key=lambda x: x.score, reverse=True)
        
        # 日志监控：看看合并后的去重数量
        logger.debug(f"RRF Merged {total_raw_count} inputs into {len(merged)} unique chunks, returning top {search_top_k}")
        
        return merged[:search_top_k]
    
    async def search(
        self,
        kb_id: int,
        question: str,
        keywords: str,
        search_top_k: int,
        threshold: float,
        do_rerank: bool,
        weight: float,
        security: int, 
        query_vec: list[float] | None = None,
        tags: list[str] = []
    ) -> dict[str, list[TxtBaseSearchResult]]:
        """Executes hybrid tiered search.
        
        Args:
            kb_id: Knowledge base identifier.
            question: Natural language query.
            keywords: Keywords for searching.
            search_top_k: Target number of results per group.
            threshold: Similarity threshold for vector search.
            do_rerank: Whether to categorize results for subsequent reranking.
            weight: Weighting factor for score calculation.
            security: Security level filter.
            query_vec: Pre-computed query embedding.
            tags: Metadata tags for filtering.

        Returns:
            Dictionary containing 'rerank_result' and 'norerank_result'.
        """
        start_time = time.time()
        logger.debug(f"Starting hybrid search for query: {question}")
        # Execute concurrent retrieval tasks
        if not query_vec:
            logger.warning("Query vector is missing; falling back to full-text search only.")
            fulltext_raw = await self.serch_by_full_text(kb_id, security, keywords, search_top_k * 3, do_rerank, weight, tags)
            vector_raw = {"rerank_result": [], "norerank_result": []}
        else:
            # Over-fetch by factor of 3 to ensure high-quality fusion pool
            vector_raw, fulltext_raw = await asyncio.gather(
                self.search_by_vector(kb_id, security, keywords, query_vec, threshold, search_top_k * 3, do_rerank, weight, tags),
                self.serch_by_full_text(kb_id, security, keywords, search_top_k * 3, do_rerank, weight, tags)
            )

        # Merge results into designated rerank/non-rerank buckets
        final_rerank = self.rrf_merge(search_top_k=search_top_k, weight=weight, results_list=[
            vector_raw.get("rerank_result", []),
            fulltext_raw.get("rerank_result", [])
        ])
        
        final_norerank = self.rrf_merge(search_top_k=search_top_k, weight=weight, results_list=[
            vector_raw.get("norerank_result", []),
            fulltext_raw.get("norerank_result", [])
        ])

        final_rerank = self._merge_adjacent_chunks(final_rerank)
        final_norerank = self._merge_adjacent_chunks(final_norerank)

        duration = time.time() - start_time
        logger.debug(f"Hybrid search completed in {duration:.2f}s. Rerank pool: {len(final_rerank)}, Non-rerank pool: {len(final_norerank)}")

        return {
            "rerank_result": final_rerank,
            "norerank_result": final_norerank
        }
    
    async def search_by_vector(self, kb_id: int, security: int, keywords: str, query_vec: list[float],
                               threshold: float, search_top_k: int, do_rerank: bool,
                               weight: float, tags: list[str] = []) -> dict[str, list[TxtBaseSearchResult]]:
        """Performs vector similarity search."""
        logger.debug("Executing vector similarity search...")
        async with self.oracle_session as session:
            repo = TxtChunkRepository(session)
            try:
                dataset = await repo.vector_search(
                    kb_id=kb_id, query_vec=query_vec, security=security, keywords=keywords,
                    similarity_threshold=threshold, search_top_k=search_top_k, tags=tags
                )
                logger.debug(f"Vector search completed. Found {len(dataset)} results")
                results = self._construct_search_result(dataset, weight=weight, search_type="semantic")
                search_result = await self._enhance_context_with_window(results)
                
                return {"rerank_result" if do_rerank else "norerank_result": search_result}
            except DataNotFoundException:
                return {"rerank_result": [], "norerank_result": []}
            except Exception as e:
                handle_exception(e, f"Vector search failed for KB {kb_id}: {str(e)}")
        
    async def serch_by_full_text(self, kb_id: int, security: int, keywords: str,
                                search_top_k: int, do_rerank: bool, weight: float,
                                tags: list[str] = []) -> dict[str, list[TxtBaseSearchResult]]:
        """Performs keyword-based full-text search."""
        async with self.oracle_session as session:
            repo = TxtChunkRepository(session)
            try:
                dataset = await repo.full_text_search(
                    kb_id=kb_id, keywords=keywords, security=security,
                    search_top_k=search_top_k, tags=tags
                )
                logger.debug(f"Full-text search completed. Found {len(dataset)} results")
                results = self._construct_search_result(dataset, weight=weight, search_type="fulltext")
                search_result = await self._enhance_context_with_window(results)
                
                return {"rerank_result" if do_rerank else "norerank_result": search_result}
            except DataNotFoundException:
                return {"rerank_result": [], "norerank_result": []}
            except Exception as e:
                handle_exception(e, f"Full-text search failed for KB {kb_id}: {str(e)}")
    
    async def _enhance_context_with_window(self, 
                                          initial_results: list[TxtBaseSearchResult], 
                                          window_size: int = 1) -> list[TxtBaseSearchResult]:
        """Expands chunk content with neighboring chunks.
        
        Fixes IllegalStateChangeError by ensuring each parallel task 
        manages its own session lifecycle.
        """
        if not initial_results:
            return []

        async def expand_single_chunk(res: TxtBaseSearchResult):
            # CRITICAL: Open a NEW session inside the task for true parallelism
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

                # # Path Gene Injection
                # path_prefix = f"[Location: {res.path_names}]\n"
                # if path_prefix not in res.content:
                #     res.content = path_prefix + res.content
                return res

        # Parallelize the tasks - each now has its own session
        tasks = [expand_single_chunk(res) for res in initial_results]
        return list(await asyncio.gather(*tasks))

    def _construct_search_result(self, dataset: list, weight: float, search_type: str) -> list[TxtBaseSearchResult]:
        """Maps raw database records to TxtBaseSearchResult objects."""
        results = []
        for item in dataset:
            try:
                if not isinstance(item, dict): continue
                    
                meta = item.get("metadata") or {}

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
                    page_num=int(meta.get("page_num", 0)),
                    image_name=meta.get("image_name", ""),
                    bbox=meta.get("bbox", []),
                    score=float(item.get("score") or 0.0),
                    weight=weight,
                    rerank_score=0.0
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to construct result for chunk {item.get('chunk_id')}: {e}")
                continue
                
        return results

    def _merge_adjacent_chunks(self, results: list[TxtBaseSearchResult], window_size: int = 1) -> list[TxtBaseSearchResult]:
        """
        改进的去重策略：
        不再预先填充 range，而是只记录已录入的中心点。
        """
        if not results:
            return []
        
        # 保持分数降序
        results.sort(key=lambda x: x.score, reverse=True)
        
        final_results = []
        # file_coverage 仅记录真正存入 final_results 的 chunk_num
        file_coverage = {}  # dict[file_id, set[chunk_num]]
        
        MIN_KEEP_COUNT = 10 

        for res in results:
            fid = res.file_id
            cnum = res.chunk_num
            
            # 初始化该文件的覆盖集
            if fid not in file_coverage:
                file_coverage[fid] = set()

            # 1. 强制保留 Top N 结果
            if len(final_results) < MIN_KEEP_COUNT:
                final_results.append(res)
                file_coverage[fid].add(cnum)
                continue

            # 2. 冗余检查逻辑
            # 检查当前 cnum 是否落在已保存块的 window 范围内
            is_redundant = False
            for existing_num in file_coverage[fid]:
                # 如果当前块编号与已存在的块编号距离在 window_size 之内，视为冗余
                if abs(existing_num - cnum) <= window_size:
                    is_redundant = True
                    break
            
            if not is_redundant:
                final_results.append(res)
                file_coverage[fid].add(cnum)
            else:
                logger.debug(f"跳过冗余邻近块: {fid} - Chunk {cnum} (距离现有块过近)")

        logger.info(f"冗余过滤完成: {len(results)} -> {len(final_results)} (window_size={window_size})")
        return final_results