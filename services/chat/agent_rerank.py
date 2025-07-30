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
            
        # Update rerank_score in original KBResult objects
        for reranker in rerankers:
            index = reranker.get("index")
            score = reranker.get("score")
            if index is not None and score is not None and 0 <= index < len(kb_results):
                kb_results[index].rerank_score = score
        
        # Sort KBResult objects by rerank_score in descending order
        kb_results.sort(key=lambda x: x.rerank_score, reverse=True) # type: ignore

        return kb_results