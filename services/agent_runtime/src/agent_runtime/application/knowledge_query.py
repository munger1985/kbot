"""将问题编译成受限检索合同；规划器不接触候选、重排或裁掉证据。"""

import json
from dataclasses import dataclass
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

    def compile(self, handles, *, independent_modalities=False):
        ordering = tuple(OrderBy(row.field, row.descending) for row in self.order_by)
        if ordering and ordering[-1].field != "stable_key":
            ordering += (OrderBy("stable_key", False),)
        combination = self.combination
        if independent_modalities and handles:
            if self.mode != "RELEVANCE":
                handles, combination = (), None
            elif len(handles) > 1 and combination in {None, "COMPOSED"}:
                # 联合看图由回答模型执行；检索只提供各图独立召回的候选。
                combination = "ANY"
        return QueryInput(QueryMode(self.mode), self.requested_count, tuple(self.output_types),
            text=self.text, image_handles=tuple(handles), combination=combination,
            predicates=tuple(Predicate(row.field, row.operator, row.value) for row in self.predicates),
            order_by=ordering)


class VisualKnowledgeQueryPlan(KnowledgeQueryPlan):
    transcription_requested: bool = False


@dataclass(frozen=True, slots=True)
class PlannedImageQuery:
    query: QueryInput
    transcription_requested: bool


async def plan_knowledge_query(*, text, image_handles, model, prompt, client, auth_context,
                               independent_modalities=False, classify_transcription=False, text_only=False):
    require(type(model.get("max_tokens")) is int and model["max_tokens"] > 0,
            "知识规划缺少冻结模型技术上限")
    if text_only:
        require(not image_handles, "纯文字知识规划不接受图片")
        prompt += "\n当前是纯文字提问。只检索正文、表格及已有的文字图注，不检索图片；如为相关检索，必须给出文本检索词。"
    if independent_modalities and image_handles:
        prompt += ("\n当前是带用户附件的问答。text 只填写问题中可用于知识库检索的主题词；"
                   "如果只是请解释图片且没有可检索主题，text=null，不把“图中是什么”等指代语当检索词。"
                   "联合理解图片由回答模型完成，检索只提供独立召回候选；如无需全部输入同时覆盖，使用 ANY。")
    schema = VisualKnowledgeQueryPlan if classify_transcription and image_handles else KnowledgeQueryPlan
    if schema is VisualKnowledgeQueryPlan:
        prompt += ("\n仅当用户明确只要求读取、识别或逐字转写图片文字时，transcription_requested=true；"
                   "如还要求判断方法、解释流程、解读图表或进行推理，一律填 false。")
    response = await client.json("POST", "/internal/v1/chat/completions", auth_context=auth_context,
        payload={"model_id": model["model_id"], "max_tokens": model["max_tokens"], "stream": False,
            "response_format": {"type": "json_object"}, "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"question": text, "image_count": len(image_handles),
                    "schema": schema.model_json_schema()}, ensure_ascii=False)}]})
    raw = response["choices"][0]["message"]["content"]
    require(type(raw) is str, "知识规划模型未返回 JSON 文本")
    plan = schema.model_validate_json(raw)
    if text_only:
        output_types = [kind for kind in dict.fromkeys(plan.output_types) if kind in {"TEXT", "TABLE"}]
        if not output_types or "MIXED" in plan.output_types:
            output_types = ["TEXT", "TABLE"]
        plan = plan.model_copy(update={"output_types": output_types, "combination": None,
            "text": ((plan.text or "").strip() or text) if plan.mode == "RELEVANCE" else None})
    query = plan.compile(image_handles, independent_modalities=independent_modalities)
    return PlannedImageQuery(query, plan.transcription_requested) if isinstance(plan, VisualKnowledgeQueryPlan) else query
