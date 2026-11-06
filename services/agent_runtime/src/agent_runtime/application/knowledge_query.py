"""将问题编译成受限检索合同；规划器不接触候选、重排或裁掉证据。"""

import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from knowledge_algorithms.contracts.query import QueryInput, QueryMode, Predicate, OrderBy
from knowledge_algorithms.contracts.errors import require
from platform_core.contracts.knowledge_app import KnowledgePredicate, KnowledgeOrder


class KnowledgeQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["RELEVANCE", "FILTERED_ORDER", "EXHAUSTIVE"]
    requested_count: int = Field(ge=1, le=100)
    output_types: list[Literal["TEXT", "FIGURE", "PHOTO", "TABLE", "PAGE", "REGION", "MIXED"]]
    text: str | None
    combination: Literal["ANY", "ALL", "COMPOSED"] | None
    predicates: list[KnowledgePredicate] = Field(max_length=20)
    order_by: list[KnowledgeOrder] = Field(max_length=4)

    def compile(self, handles):
        ordering = tuple(OrderBy(row.field, row.descending) for row in self.order_by)
        if ordering and ordering[-1].field != "stable_key":
            ordering += (OrderBy("stable_key", False),)
        return QueryInput(QueryMode(self.mode), self.requested_count, tuple(self.output_types),
            text=self.text, image_handles=tuple(handles), combination=self.combination,
            predicates=tuple(Predicate(row.field, row.operator, row.value) for row in self.predicates),
            order_by=ordering)


async def plan_knowledge_query(*, text, image_handles, model, prompt, client, auth_context):
    require(type(model.get("max_tokens")) is int and model["max_tokens"] > 0,
            "知识规划缺少冻结模型技术上限")
    response = await client.json("POST", "/internal/v1/chat/completions", auth_context=auth_context,
        payload={"model_id": model["model_id"], "max_tokens": model["max_tokens"], "stream": False,
            "response_format": {"type": "json_object"}, "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"question": text, "image_count": len(image_handles),
                    "schema": KnowledgeQueryPlan.model_json_schema()}, ensure_ascii=False)}]})
    raw = response["choices"][0]["message"]["content"]
    require(type(raw) is str, "知识规划模型未返回 JSON 文本")
    return KnowledgeQueryPlan.model_validate_json(raw).compile(image_handles)
