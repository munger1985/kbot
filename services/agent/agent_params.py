from pydantic import BaseModel


class AgentParams(BaseModel):
    """智能体参数类"""
    domain_id: int | None = None
    prompt_id: int | None = None
    llm_id: int | None = None
    llm_params: dict | None = None
    feedback_similarity_flag: bool = False
    synonym_similarity_flag: bool = False
    reranker_model_id: int | None = None
    reranker_top_k: int | None = None
    reranker_score_threshold: float = 0.0

    class Config:
        arbitrary_types_allowed = True
        from_attributes = True
