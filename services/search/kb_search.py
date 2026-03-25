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
        """Merges multiple search results using Reciprocal Rank Fusion (RRF).
        
        Args:
            search_top_k: Number of top results to return.
            weight: Global weight multiplier for the scores.
            results_list: A list of result sets from different search engines.
            
        Returns:
            A list of merged and re-ranked TxtBaseSearchResult objects.
        """
        k = 60
        rrf_score_map = {}  # key: chunk_id, value: aggregated rrf score
        chunk_map = {}
        
        # 1. Calculate base RRF scores to normalize different rank scales
        for results in results_list:
            for rank, r in enumerate(results, 1):
                current_rrf = 1.0 / (k + rank)
                rrf_score_map[r.chunk_id] = rrf_score_map.get(r.chunk_id, 0.0) + current_rrf
                chunk_map[r.chunk_id] = r
        
        # 2. Apply business logic boosts (Hierarchy and Type)
        merged = []
        for cid, base_rrf_score in rrf_score_map.items():
            res = chunk_map[cid]
            
            # Boost L1/L2 titles as they provide high-level context
            level_boost = 1.5 if (0 < res.structure_level < 3) else 1.0
            
            # Subtle boost for non-prose elements like tables or images
            type_boost = 1.1 if res.chunk_type in ["table", "picture"] else 1.0
            
            res.score = base_rrf_score * level_boost * type_boost * weight
            merged.append(res)
        
        # 3. Sort by final score and truncate
        merged.sort(key=lambda x: x.score, reverse=True)
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
                self.serch_by_full_text(kb_id, security, question, search_top_k * 3, do_rerank, weight, tags)
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

                # Path Gene Injection
                path_prefix = f"[Location: {res.path_names}]\n"
                if path_prefix not in res.content:
                    res.content = path_prefix + res.content
                return res

        # Parallelize the tasks - each now has its own session
        tasks = [expand_single_chunk(res) for res in initial_results]
        return list(await asyncio.gather(*tasks))

    def _construct_search_result(self, dataset: list, weight: float, 
                                 search_type: str) -> list[TxtBaseSearchResult]:
        """Maps raw database records to TxtBaseSearchResult objects."""
        results = []
        for item in dataset:
            try:
                if not isinstance(item, dict): continue
                    
                meta = item.get("metadata") or {}
                path_names = item.get("path_names") or ""

                result = TxtBaseSearchResult(
                    chunk_id=str(item.get("chunk_id", "")),
                    file_id=item.get("file_id", ""),
                    content=item.get("content", ""),
                    path_names=str(path_names),
                    structure_level=int(item.get("structure_level", 0)),
                    node_path=meta.get("node_path", "") or "", 
                    page_num=int(meta.get("page_num") or 0),
                    chunk_num=int(meta.get("chunk_num") or 0),
                    sub_index=int(meta.get("sub_index") or 0),
                    chunk_type=meta.get("chunk_type", "text"),
                    score=float(item.get("score") or 0.0),
                    embedding=item.get("embedding", []) or [],
                    weight=weight,
                    search_type=search_type,
                    rerank_score=0.0
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to construct result for chunk {item.get('chunk_id')}: {e}")
                continue
                
        return results

    def _merge_adjacent_chunks(self, results: list[TxtBaseSearchResult]) -> list[TxtBaseSearchResult]:
        """合并物理位置连续的 Chunk，避免上下文重复。"""
        if not results: return []
        
        # 1. 按文件和序号排序，以便识别连续性
        # 注意：这里要保持原始评分的参考，通常只合并来自同一文件且序号相邻的
        sorted_results = sorted(results, key=lambda x: (x.file_id, x.chunk_num))
        
        merged_results = []
        if not sorted_results: return []
        
        current = sorted_results[0]
        
        for next_chunk in sorted_results[1:]:
            # 如果是同一文件，且序号连续（差值在 window 范围内）
            if (next_chunk.file_id == current.file_id and 
                next_chunk.chunk_num <= current.chunk_num + 2): # 阈值可调
                
                # 合并内容（需去重或按顺序拼接）
                # 如果已经做过 window 扩展，这里需要更精细的处理，
                # 简单做法是保留 score 更高的那个，并把内容标记为已处理
                current.content += f"\n[Next Section]\n{next_chunk.content}"
                # 取两者中较高的分数
                current.score = max(current.score, next_chunk.score)
            else:
                merged_results.append(current)
                current = next_chunk
        
        merged_results.append(current)
        # 2. 最后按分数重新排回降序
        return sorted(merged_results, key=lambda x: x.score, reverse=True)