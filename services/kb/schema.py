from pydantic import BaseModel
from typing import Any

class ModelParams(BaseModel):
    """模型参数配置"""
    llm_model: str
    txt_embedding_model: str
    img_embedding_model: str
    vlm_model: str
    rerank_model: str
    do_rerank: bool
    llm_params: dict[str, Any] | None
    rerank_top_k: int | None

class KBModelParams(ModelParams):
    dbconf: dict[str, Any] | None