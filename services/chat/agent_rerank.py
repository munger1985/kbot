from loguru import logger
from .agent_params import KBResult, AgentParams
from utils.call_models import CallModel


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
            Reranked KBResult list with updated reranker_scores, or None if error
        """
        if not kb_results:
            return kb_results
            
        # Extract chunk_docs from KBResult objects
        contonts = [result.content for result in kb_results]
        
        # Call reranker model
        rerankers = await CallModel().call_reranker_model(
            model_unique_name=self.agent_params.reranker_model_name, # type: ignore
            query=question,
            documents=contonts,
            top_k=self.agent_params.reranker_top_k
        )
        
        if rerankers is None:
            return None

        reranked_results: list[KBResult] = []    
        # Update reranker_score in original KBResult objects
        for reranker in rerankers:
            index = reranker.get("index")
            score = reranker.get("score")
            logger.debug(f"Reranker index: {index}, score: {score}")

            if index is not None and score is not None:
                reranked_result = KBResult()
                reranked_result.file_id=kb_results[index].file_id
                reranked_result.chunk_type=kb_results[index].chunk_type
                reranked_result.page_num=kb_results[index].page_num
                reranked_result.content=kb_results[index].content
                reranked_result.similarity=kb_results[index].similarity
                reranked_result.weight=kb_results[index].weight
                reranked_result.reranker_score=score
                reranked_results.append(reranked_result)

                logger.debug(f"KBResult chunk_doc: {reranked_result.content[0:20]}")
                logger.debug(f"KBResult reranker_score: {reranked_result.reranker_score}")
                logger.debug(f"KBResult weight: {reranked_result.weight}")
                

        logger.debug(f"Reranked {len(reranked_results)} results with reranker model {self.agent_params.reranker_model_name}")

        return reranked_results