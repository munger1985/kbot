"""根据冻结会话上下文生成可独立理解的问题。"""

from __future__ import annotations

from typing import Any

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from platform_core.prompts import StrictPromptRenderer

from .contracts import ContextRewriteOutput


class ContextRewriteSkill:
    def __init__(self, *, model_client, prompt_resolver):
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, context: ExecutionContext) -> SkillResult:
        conversation = dict(
            context.config_snapshot.get("conversation") or {}
        )
        memory_context = dict(conversation.get("context") or {})
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
        response = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=rendered,
            max_tokens=2048,
        )
        normalized: dict[str, Any] = {
            **response,
            "raw_input": context.original_input,
        }
        output = ContextRewriteOutput.model_validate(normalized)
        allowed_memory_ids = {
            str(item.get("memory_id"))
            for item in memory_context.get("memories") or []
            if item.get("memory_id")
        }
        if not set(output.memory_refs).issubset(allowed_memory_ids):
            raise ValueError("上下文改写引用了未提供的 Memory")
        return self._result(context, output, prompt_ref=prompt.ref())

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
