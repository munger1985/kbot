from loguru import logger
from .agent_params import KBResult, AgentParams
from utils.call_models import call_reranker_model


class AgentRerank:
    """Agent rerank"""
    def __init__(self, agent_params: AgentParams):
        self.agent_params = agent_params

    async def rerank_kb(self, question: str, kb_results: list[KBResult]) -> list[KBResult] | None:
        """Rerank the knowledge base results.
        
        Args:
            model_unique_name: Model unique name for reranking
            top_k: Number of top results to return
            question: Query text
            kb_results: List of KBResult objects to rerank
            
        Returns:
            Reranked KBResult list with updated rerank_scores, or None if error
        """
        if not kb_results:
            return kb_results
            
        # Extract chunk_docs from KBResult objects
        chunk_docs = [result.chunk_doc for result in kb_results]
        
        # Call reranker model
        rerankers = await call_reranker_model(
            model_unique_name=self.agent_params.reranker_model_name, # type: ignore
            query=question,
            documents=chunk_docs,
            top_k=self.agent_params.reranker_top_k
        )
        
        if rerankers is None:
            return None

        reranked_results: list[KBResult] = []    
        # Update rerank_score in original KBResult objects
        for reranker in rerankers:
            index = reranker.get("index")
            score = reranker.get("score")
            logger.debug(f"Reranker index: {index}, score: {score}")

            if index is not None and score is not None:
                reranked_result = KBResult()
                reranked_result.chunk_doc=kb_results[index].chunk_doc
                reranked_result.embed_id=kb_results[index].embed_id
                reranked_result.kb_id=kb_results[index].kb_id
                reranked_result.file_id=kb_results[index].file_id
                reranked_result.chunk_doc=kb_results[index].chunk_doc
                reranked_result.chunk_metadata=kb_results[index].chunk_metadata
                reranked_result.similarity=kb_results[index].similarity
                reranked_result.weight=kb_results[index].weight
                reranked_result.rerank_score=score
                reranked_results.append(reranked_result)

                logger.debug(f"KBResult chunk_doc: {reranked_result.chunk_doc[0:20]}")
                logger.debug(f"KBResult chunk_metadata: {reranked_result.chunk_metadata}")
                logger.debug(f"KBResult rerank_score: {reranked_result.rerank_score}")
                logger.debug(f"KBResult weight: {reranked_result.weight}")
                

        logger.debug(f"Reranked {len(reranked_results)} results with reranker model {self.agent_params.reranker_model_name}")

        return reranked_results