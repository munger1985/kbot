from pydantic import BaseModel
from typing import Any

class ToolParams(BaseModel):
    """工具参数类"""
    conf_id: int = 0
    tool_id: int = 0
    tool_type: int = 0
    tool_weight: float = 0.0
    reranker_flag: int = 0
    search_type: int = 0
    search_top_k: int = 10
    threshold: float = 0.7
    kb_catogory: int | None = None
    img2txt_model: int | None = None
    img_embed_model: int | None = None
    txt_embed_model: int | None = None

    class Config:
        # 允许任意类型，避免序列化问题
        arbitrary_types_allowed = True
        from_attributes = True

    @classmethod
    def from_orm(cls, obj: Any) -> 'ToolParams':
        """从ORM对象创建ToolParams"""
        return cls(
            conf_id=getattr(obj, 'conf_id', 0),
            tool_id=getattr(obj, 'tool_id', 0),
            tool_type=getattr(obj, 'tool_type', 0),
            tool_weight=getattr(obj, 'tool_weight', 0.0) or 0.0,
            reranker_flag=getattr(obj, 'reranker_flag', 0) or 0,
            search_type=getattr(obj, 'search_type', 0),
            search_top_k=getattr(obj, 'search_top_k', 10) or 10,
            threshold=getattr(obj, 'search_score_threshold', 0.7) or 0.7,
            kb_catogory=None,
            img2txt_model=None,
            img_embed_model=None,
            txt_embed_model=None
        )


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


class KBResult(BaseModel):
    """知识库结果类"""
    file_id: str = ""
    chunk_type: int = 1
    page_num: int = 0
    content: str = ""
    similarity: float = 0.0
    weight: float = 0.0
    reranker_score: float = 0.0

    class Config:
        arbitrary_types_allowed = True