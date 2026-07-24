"""不调用领域工具的通用对话回答。"""

from __future__ import annotations

from agent_runtime.runtime import (
    ExecutionContext,
    SkillArtifact,
    SkillProgress,
    SkillResult,
)


class ConversationResponseSkill:
    def __init__(self, *, model_client, prompt_resolver):
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, context: ExecutionContext) -> SkillResult:
        """为直接调用和契约测试聚合流式结果。"""
        result = None
        async for item in self.execute_stream(context):
            if isinstance(item, SkillResult):
                result = item
        if result is None:
            raise RuntimeError("通用对话 Skill 未返回最终结果")
        return result

    async def execute_stream(self, context: ExecutionContext):
        route = dict(context.config_snapshot.get("route") or {})
        if route.get("route_type") == "CLARIFY":
            question = str(
                route.get("clarification_question")
                or "请说明这是通用问题、文档查询还是业务数据查询。"
            )
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": 1, "delta": question},
            )
            yield SkillResult(
                artifact=SkillArtifact(
                    artifact_type="GROUNDED_ANSWER",
                    schema_version="GroundedAnswer.v1",
                    payload={
                        "answer": question,
                        "status": "CLARIFICATION_REQUIRED",
                        "used_citation_labels": [],
                        "references": [],
                        "warnings": [
                            str(route.get("reason") or "路由需要澄清")
                        ],
                    },
                    provenance={
                        "run_id": str(context.run_id),
                        "task_id": str(context.task_id),
                        "answer_mode": "ROUTE_CLARIFICATION",
                    },
                )
            )
            return
        model_name = str(
            context.config_snapshot.get("agent", {}).get(
                "composer_llm_model_name"
            )
            or ""
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 composer_llm_model_name")
        prompt = await self._prompt_resolver.resolve(
            "agent_runtime.conversation_response"
        )
        standalone = self._standalone_query(context)
        instruction = str(
            context.config_snapshot.get("agent", {}).get("instruction")
            or "请准确、简洁地回答用户。"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{prompt.content}\n\nAgent 指令：\n{instruction}"
                ),
            },
            {"role": "user", "content": standalone},
        ]
        yield SkillProgress(
            event_type="thinking.delta",
            payload={
                "delta": "正在生成通用对话回答",
                "public_summary": "正在生成通用对话回答",
            },
        )
        parts: list[str] = []
        index = 0
        async for chunk in self._model_client.stream_llm_chunks(
            served_model_name=model_name,
            prompt=messages,
            max_tokens=4096,
            temperature=0.2,
        ):
            if not chunk.content:
                continue
            parts.append(chunk.content)
            index += 1
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": index, "delta": chunk.content},
            )
        answer = "".join(parts).strip()
        if not answer:
            raise ValueError("通用对话模型返回空回答")
        yield SkillResult(
            artifact=SkillArtifact(
                artifact_type="GROUNDED_ANSWER",
                schema_version="GroundedAnswer.v1",
                payload={
                    "answer": answer,
                    "status": "READY",
                    "used_citation_labels": [],
                    "references": [],
                    "warnings": [],
                },
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                    "prompt": prompt.ref(),
                    "answer_mode": "CONVERSATION",
                },
            )
        )

    @staticmethod
    def _standalone_query(context: ExecutionContext) -> str:
        for artifact in reversed(context.input_artifacts):
            if artifact.artifact_type == "CONTEXT_REWRITE":
                value = str(
                    (artifact.payload or {}).get("standalone_query") or ""
                ).strip()
                if value:
                    return value
        return context.original_input
