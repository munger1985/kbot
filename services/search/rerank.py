import asyncio
from loguru import logger
from utils.clients import AIModelClient
from .result import TxtBaseSearchResult


class TxtBaseRerank:
    """Reranking service for knowledge base search results.
    
    This class leverages Cross-Encoder models to re-evaluate the relevance
    of retrieved chunks against the original user query, utilizing
    hierarchical path information for better context calibration.
    """

    def __init__(self):
        self.model_client = AIModelClient()

    async def rerank(self, 
                     model_name: str, 
                     top_k: int, 
                     question: str, 
                     kb_results: list[TxtBaseSearchResult]
                    ) -> list[TxtBaseSearchResult]:
        """Reranks search results using a cross-encoder model.
        
        Args:
            model_name: Name of the reranker model (e.g., 'bge-reranker-v2-m3').
            top_k: Maximum number of final results to return.
            question: Original user query string.
            kb_results: Initial recall set (usually around 50-100 items).
            min_rerank_score: Minimum threshold to filter out irrelevant noise.
            
        Returns:
            A list of reranked TxtBaseSearchResult objects, sorted by score.
        """
        if not kb_results:
            logger.warning("Rerank received an empty result set.")
            return []

        # 1. Deduplication (Critical for Hybrid Search efficiency)
        seen_ids = set()
        unique_results = []
        for res in kb_results:
            if res.chunk_id not in seen_ids:
                unique_results.append(res)
                seen_ids.add(res.chunk_id)

        # 2. Extract contents (already contain [Location: ...] prefixes)
        contents = [res.content for res in unique_results]
        
        # 3. Call Reranker Model
        try:
            # We score the entire unique pool to ensure the best global candidates
            response = await self.model_client.call_reranker_model(
                model_name=model_name,
                query=question,
                documents=contents,
                top_k=len(unique_results) 
            )
        except Exception as e:
            logger.error(f"Reranker model invocation failed: {e}")
            # Fallback: Return top_k from original recall set based on RRF scores
            return unique_results[:top_k]

        if not response:
            logger.warning("Reranker returned an empty response.")
            return unique_results[:top_k]

        # 4. Map results back to objects and apply score filtering
        reranked_results: list[TxtBaseSearchResult] = []
        for item in response:
            try:
                index = item.get("index")
                score = float(item.get("score", 0.0))
                
                # Validation: Ensure index is within bounds of unique_results
                if index is not None and 0 <= index < len(unique_results):
                    target_result = unique_results[index]
                    target_result.rerank_score = score
                    reranked_results.append(target_result)
                else:
                    logger.warning(f"Reranker returned an out-of-bounds index: {index}")
            except (TypeError, ValueError) as e:
                logger.error(f"Error parsing reranker output item: {item} | {e}")
                continue

        # 5. Final Sort and Truncation
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Logging for observability
        if reranked_results:
            top_hit = reranked_results[0]
            logger.debug(f"Top Rerank Match | Score: {top_hit.rerank_score:.4f} | "
                         f"ID: {top_hit.chunk_id} | Content: {top_hit.content[:50]}...")

        duration_msg = (f"Rerank complete ({model_name}): {len(unique_results)} input -> Top {top_k} selected.")
        logger.info(duration_msg)

        return reranked_results[:int(top_k)]