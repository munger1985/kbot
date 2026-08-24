"""上下文问题改写的类型化输出。"""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ContextRewriteOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_input: str = Field(min_length=1, max_length=32000)
    standalone_query: str = Field(min_length=1, max_length=32000)
    retrieval_queries: tuple[str, ...] = Field(min_length=1, max_length=5)
    resolved_references: tuple[str, ...] = ()
    active_topic: str | None = Field(default=None, max_length=512)
    ambiguity: bool = False
    clarification_question: str | None = Field(
        default=None, max_length=1000
    )
    memory_refs: tuple[str, ...] = ()

    @field_validator("resolved_references", mode="before")
    @classmethod
    def normalize_resolved_references(cls, value):
        """将模型的等价对象表达收敛为合同规定的字符串数组。"""
        if value == {}:
            return ()
        if not isinstance(value, (list, tuple)):
            return value
        normalized: list[object] = []
        for item in value:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            reference = item.get("reference")
            resolved_to = item.get("resolved_to") or item.get("resolution")
            if not isinstance(reference, str) or not reference.strip():
                normalized.append(item)
                continue
            if not isinstance(resolved_to, str) or not resolved_to.strip():
                normalized.append(item)
                continue
            normalized.append(
                f"{reference.strip()}={resolved_to.strip()}"
            )
        return tuple(normalized)

    @field_validator("memory_refs", mode="before")
    @classmethod
    def normalize_empty_memory_refs(cls, value):
        """容忍模型将空 Memory 引用数组错误输出为空对象。"""
        if value == {}:
            return ()
        return value

    @model_validator(mode="after")
    def validate_ambiguity(self):
        if self.ambiguity and not self.clarification_question:
            raise ValueError("存在歧义时必须提供 clarification_question")
        if not self.ambiguity and self.clarification_question:
            raise ValueError("无歧义时不得提供 clarification_question")
        return self
