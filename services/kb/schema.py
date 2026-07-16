from pydantic import BaseModel
from typing import Any

class ModelParams(BaseModel):
    """模型参数配置"""
    llm_model: str                            # 主模型：Reasoning、Planning
    llm_model_light: str | None = None        # 轻量模型：分类、提取、Reranker 等（回退到 llm_model）
    txt_embedding_model: str
    visual_embedding_model: str
    vlm_model: str
    do_rerank: bool = False                   # LLM Reranker 开关
    llm_params: dict[str, Any] | None = None
    rerank_top_k: int | None = 10
    enable_hyde: bool = False                 # HyDE 假设答案增强检索
    enable_section_context: bool = True       # Section 级上下文扩展

class KBModelParams(ModelParams):
    dbconf: dict[str, Any] | None