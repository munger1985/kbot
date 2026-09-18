"""根据冻结会话上下文生成可独立理解的问题。"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import ValidationError

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.language import language_instruction, response_language
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from platform_core.prompts import StrictPromptRenderer

from .contracts import ContextRewriteOutput

FOLLOWUP_FOCUS_RULE = (
    "消解指代时必须遵守：最近一轮助手回答若已选定唯一对象，"
    "该对象就是当前默认焦点；当前输入的对象若仍依赖指代、指示或省略，"
    "即使带了来源或分析角度，也必须补全为该焦点，不得改写成完整新问题，"
    "也不得把单数指代扩成更早出现的同类对象集合；只有当前输入自己点名了"
    "对象或范围时才不继承上一轮焦点。"
)


class ContextRewriteSkill:
    def __init__(self, *, model_client, prompt_resolver):
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, context: ExecutionContext) -> SkillResult:
        conversation = dict(
            context.config_snapshot.get("conversation") or {}
        )
        memory_context = dict(conversation.get("context") or {})
        route = dict(context.config_snapshot.get("route") or {})
        if (
            str(route.get("classifier_version") or "").startswith((
                "llm-km-asset-v1:",
                "asset-search-plan-v1:",
                "km-portal-help-v1:",
            ))
            and route.get("context_required") is False
        ):
            output = ContextRewriteOutput(
                raw_input=context.original_input,
                standalone_query=context.original_input,
                retrieval_queries=(context.original_input,),
            )
            return self._result(
                context,
                output,
                prompt_ref={"source": "KM_ROUTER_SELF_CONTAINED"},
            )
        if not (
            memory_context.get("summary")
            or memory_context.get("recent_items")
            or memory_context.get("memories")
        ):
            output = ContextRewriteOutput(
                raw_input=context.original_input,
                standalone_query=context.original_input,
                retrieval_queries=(context.original_input,),
            )
            return self._result(
                context,
                output,
                prompt_ref={"source": "DETERMINISTIC_NO_CONTEXT"},
            )

        model_name = str(
            agent_model_name(
                context.config_snapshot.get("agent", {}), "context_llm"
            )
            or ""
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 models.context_llm")
        prompt = await self._prompt_resolver.resolve(
            "agent_runtime.context_rewrite"
        )
        rendered = StrictPromptRenderer.render(
            prompt,
            {
                "raw_input": context.original_input,
                "conversation_summary": memory_context.get("summary") or {},
                "recent_items": memory_context.get("recent_items") or [],
                "recalled_memories": memory_context.get("memories") or [],
            },
        )
        language = response_language(
            context.config_snapshot, context.original_input
        )
        messages = [
            {"role": "system", "content": rendered},
            {
                "role": "system",
                "content": language_instruction(language),
            },
            {
                "role": "system",
                "content": FOLLOWUP_FOCUS_RULE,
            },
        ]
        response = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=messages,
        )
        try:
            output = self._validate_response(
                response,
                original_input=context.original_input,
            )
        except ValidationError as exc:
            logger.warning(
                "上下文改写模型输出不符合契约，准备执行一次格式修正 "
                "| model={} | prompt_version={} | shape={} | errors={}",
                model_name,
                prompt.version,
                self._response_shape(response),
                self._validation_summary(exc),
            )
            corrected_response = await self._model_client.get_llm_json(
                served_model_name=model_name,
                prompt=[
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "上一份输出未通过字段校验。请仅重新输出一个 JSON 对象："
                            "retrieval_queries 必须是包含至少一个非空字符串的数组；"
                            "resolved_references 和 memory_refs 必须是字符串数组，"
                            "数组元素不得是对象；"
                            "ambiguity 必须是 JSON 布尔值 true 或 false，不能是字符串。"
                        ),
                    },
                ],
            )
            try:
                output = self._validate_response(
                    corrected_response,
                    original_input=context.original_input,
                )
            except ValidationError as corrected_exc:
                logger.error(
                    "上下文改写模型格式修正后仍不符合契约 "
                    "| model={} | prompt_version={} | shape={} | errors={}",
                    model_name,
                    prompt.version,
                    self._response_shape(corrected_response),
                    self._validation_summary(corrected_exc),
                )
                raise
        allowed_memory_ids = {
            str(item.get("memory_id"))
            for item in memory_context.get("memories") or []
            if item.get("memory_id")
        }
        if not set(output.memory_refs).issubset(allowed_memory_ids):
            raise ValueError("上下文改写引用了未提供的 Memory")
        output = self._apply_unique_query_focus(
            output,
            original_input=context.original_input,
            route=route,
            recent_items=memory_context.get("recent_items") or [],
        )
        return self._result(context, output, prompt_ref=prompt.ref())

    @classmethod
    def _apply_unique_query_focus(
        cls,
        output: ContextRewriteOutput,
        *,
        original_input: str,
        route: dict[str, Any],
        recent_items: list[dict[str, Any]],
    ) -> ContextRewriteOutput:
        """用上一轮唯一问数行确定性约束需要上下文的追问。"""
        if route.get("context_required") is not True:
            return output
        focus = cls._latest_unique_query_row(recent_items)
        if focus is None:
            return output
        focused_query = (
            f"上一轮唯一结构化结果：{focus}。当前追问：{original_input}"
        )
        if len(focused_query) > 32000:
            return output
        reference = f"上一轮唯一结构化结果={focus}"
        return ContextRewriteOutput.model_validate(
            {
                **output.model_dump(mode="python"),
                "standalone_query": focused_query,
                "retrieval_queries": (focused_query,),
                "resolved_references": tuple(dict.fromkeys((
                    *output.resolved_references,
                    reference,
                ))),
                "ambiguity": False,
                "clarification_question": None,
            }
        )

    @staticmethod
    def _latest_unique_query_row(
        recent_items: list[dict[str, Any]],
    ) -> str | None:
        """提取最近助手回答中唯一一行、可安全重放的结构化结果。"""
        for item in reversed(recent_items):
            if str(item.get("role") or "").upper() != "ASSISTANT":
                continue
            content = item.get("content")
            if not isinstance(content, dict):
                return None
            query_results = content.get("query_results")
            if not isinstance(query_results, list) or len(query_results) != 1:
                return None
            query_result = query_results[0]
            if not isinstance(query_result, dict):
                return None
            rows = query_result.get("rows")
            if not isinstance(rows, list) or len(rows) != 1:
                return None
            row = rows[0]
            if not isinstance(row, dict) or not row:
                return None

            labels: dict[str, str] = {}
            ordered_names: list[str] = []
            columns = query_result.get("columns")
            if isinstance(columns, list):
                for column in columns:
                    if not isinstance(column, dict):
                        continue
                    name = str(column.get("name") or "").strip()
                    if not name or name in ordered_names:
                        continue
                    ordered_names.append(name)
                    labels[name] = str(
                        column.get("label")
                        or column.get("display_name")
                        or name
                    ).strip()
            ordered_names.extend(
                str(name)
                for name in row
                if str(name) not in ordered_names
            )
            values: list[str] = []
            for name in ordered_names:
                value = row.get(name)
                if value is None or isinstance(value, (dict, list, tuple)):
                    continue
                rendered = str(value).strip()
                if not rendered:
                    continue
                values.append(f"{labels.get(name, name)}={rendered}")
                if len(values) >= 8:
                    break
            if not values:
                return None
            focus = "，".join(values)
            return focus[:2000]
        return None

    @staticmethod
    def _validate_response(
        response: Any,
        *,
        original_input: str,
    ) -> ContextRewriteOutput:
        """将模型响应放入固定输入后执行完整契约校验。"""
        if not isinstance(response, dict):
            raise ValueError("上下文改写模型未返回 JSON 对象")
        return ContextRewriteOutput.model_validate(
            {**response, "raw_input": original_input}
        )

    @staticmethod
    def _response_shape(response: Any) -> dict[str, Any]:
        """仅记录字段结构，不将用户问题或模型正文写入运行日志。"""
        if not isinstance(response, dict):
            return {"response_type": type(response).__name__}
        queries = response.get("retrieval_queries")
        return {
            "retrieval_queries_type": type(queries).__name__,
            "retrieval_queries_count": (
                len(queries) if isinstance(queries, (list, tuple)) else None
            ),
            "ambiguity_type": type(response.get("ambiguity")).__name__,
            "has_clarification_question": bool(
                response.get("clarification_question")
            ),
        }

    @staticmethod
    def _validation_summary(exc: ValidationError) -> list[str]:
        """生成不含原始输入值的字段校验摘要。"""
        return [
            f"{'.'.join(str(item) for item in error['loc'])}:{error['type']}"
            for error in exc.errors(include_input=False, include_url=False)
        ]

    @staticmethod
    def _result(
        context: ExecutionContext,
        output: ContextRewriteOutput,
        *,
        prompt_ref: dict[str, Any],
    ) -> SkillResult:
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="CONTEXT_REWRITE",
                schema_version="ContextRewriteOutput.v1",
                payload=output.model_dump(mode="json"),
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                    "prompt": prompt_ref,
                },
            )
        )
