"""上下文问题改写的类型化输出。"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="after")
    def validate_ambiguity(self):
        if self.ambiguity and not self.clarification_question:
            raise ValueError("存在歧义时必须提供 clarification_question")
        if not self.ambiguity and self.clarification_question:
            raise ValueError("无歧义时不得提供 clarification_question")
        return self
