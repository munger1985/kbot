from pydantic import BaseModel
from typing import Any


class ModelParams(BaseModel):
    llm_model: str
    llm_params: dict[str, Any] | None
    embedding_model: str
    rerank_model: str | None
    rerank_top_k: int