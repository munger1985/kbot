"""案例固定知识资源的显式模型角色合同。"""

from uuid import UUID

from pydantic import model_validator

from .types import AIOpsContract


CASE_MODEL_ROLES = frozenset({"parser_llm", "retrieval_llm", "embedding"})


class CaseCollectionModels(AIOpsContract):
    models: dict[str, UUID]

    @model_validator(mode="after")
    def validate_roles(self):
        if set(self.models) != CASE_MODEL_ROLES:
            raise ValueError("案例须显式配置 parser_llm、retrieval_llm 和 embedding，不使用 OCR 或视觉模型")
        return self
