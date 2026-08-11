"""仅消费已验证 Artifact 的最终回答组合器。"""

import re
from typing import Any

from loguru import logger

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.runtime import (
    ExecutionContext,
    SkillArtifact,
    SkillProgress,
    SkillResult,
)
from agent_runtime.specialists.document.contracts import (
    DocumentRetrievalResult,
)
from agent_runtime.specialists.data_query.contracts import QueryResult
from agent_runtime.specialists.visualization import EChartsResult
from platform_core.prompts import StrictPromptRenderer

from .contracts import (
    AIOpsReferenceCard,
    GroundedAnswer,
    QueryResultReferenceCard,
    ReferenceCard,
)


_CITATION_PATTERN = re.compile(r"\[([A-Z]\d+)\]")
_CITATION_VARIANT_PATTERN = re.compile(
    r"(?:【\s*([A-Z]\d+)\s*】|<sup>\s*\[?([A-Z]\d+)\]?\s*</sup>)",
    re.IGNORECASE,
)


def _normalize_citations(value: str) -> str:
    """将模型常见的引用变体统一为协议规定的 ASCII 方括号格式。"""

    def replace(match: re.Match[str]) -> str:
        label = next(
            group for group in match.groups() if group is not None
        )
        return f"[{label.upper()}]"

    return _CITATION_VARIANT_PATTERN.sub(replace, value)


class ResponseComposerSkill:
    def __init__(self, *, model_client, prompt_resolver):
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, context: ExecutionContext) -> SkillResult:
        clarification = self._blocking_clarification(context)
        if clarification is not None:
            return self._result(
                context,
                GroundedAnswer(
                    answer=clarification,
                    status="CLARIFICATION_REQUIRED",
                    warnings=("当前问题存在无法安全消解的上下文歧义",),
                ),
            )
        aiops_result = self._aiops_result(context)
        if aiops_result is not None:
            return self._compose_aiops(context, aiops_result)
        query_result = self._query_result(context)
        if query_result is not None:
            return await self._compose_query_result(context, query_result)
        retrieval = self._document_result(context)
        if retrieval is None or not retrieval.citation_pack.citations:
            retrieval_warnings = (
                retrieval.warnings if retrieval is not None else ()
            )
            answer = GroundedAnswer(
                answer="当前授权知识范围内没有找到足够的可引用证据。",
                status="INSUFFICIENT_EVIDENCE",
                warnings=(
                    *retrieval_warnings,
                    "回答未调用模型补写无来源内容",
                ),
            )
            return self._result(context, answer)

        model_name = str(
            agent_model_name(
                context.config_snapshot.get("agent", {}), "composer_llm"
            )
            or ""
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 models.composer_llm")

        allowed = {
            item.citation_label: item
            for item in retrieval.citation_pack.citations
        }
        prompt_definition = await self._prompt_resolver.resolve(
            "agent_runtime.response_compose"
        )
        prompt = self._prompt(
            context,
            retrieval,
            prompt_definition=prompt_definition,
        )
        response = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=prompt,
            max_tokens=4096,
        )
        answer_text, used_labels = self._validate_model_answer(
            response, allowed
        )
        references = tuple(
            ReferenceCard(
                citation_label=label,
                collection_id=allowed[label].collection_id,
                bundle_id=allowed[label].bundle_id,
                bundle_revision_id=allowed[label].bundle_revision_id,
                document_id=allowed[label].document_id,
                document_version_id=allowed[label].document_version_id,
                title=allowed[label].title,
                locator=allowed[label].locator,
                locator_schema_version=(
                    allowed[label].locator_schema_version
                ),
            )
            for label in used_labels
        )
        grounded = GroundedAnswer(
            answer=answer_text,
            status="READY",
            used_citation_labels=used_labels,
            references=references,
            warnings=retrieval.warnings,
        )
        return self._result(context, grounded)

    async def execute_stream(self, context: ExecutionContext):
        """流式生成回答；最终 Artifact 仍执行完整引用校验。"""
        clarification = self._blocking_clarification(context)
        if clarification is not None:
            grounded = GroundedAnswer(
                answer=clarification,
                status="CLARIFICATION_REQUIRED",
                warnings=("当前问题存在无法安全消解的上下文歧义",),
            )
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": 1, "delta": clarification},
            )
            yield self._result(context, grounded)
            return
        aiops_result = self._aiops_result(context)
        if aiops_result is not None:
            result = self._compose_aiops(context, aiops_result)
            answer = str(result.artifact.payload.get("answer") or "")
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": 1, "delta": answer},
            )
            yield result
            return
        query_result = self._query_result(context)
        if query_result is not None:
            async for item in self._stream_query_result(
                context, query_result
            ):
                yield item
            return
        retrieval = self._document_result(context)
        if retrieval is None or not retrieval.citation_pack.citations:
            retrieval_warnings = (
                retrieval.warnings if retrieval is not None else ()
            )
            grounded = GroundedAnswer(
                answer="当前授权知识范围内没有找到足够的可引用证据。",
                status="INSUFFICIENT_EVIDENCE",
                warnings=(
                    *retrieval_warnings,
                    "回答未调用模型补写无来源内容",
                ),
            )
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": 1, "delta": grounded.answer},
            )
            yield self._result(context, grounded)
            return

        model_name = str(
            agent_model_name(
                context.config_snapshot.get("agent", {}), "composer_llm"
            )
            or ""
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 models.composer_llm")
        allowed = {
            item.citation_label: item
            for item in retrieval.citation_pack.citations
        }
        prompt_definition = await self._prompt_resolver.resolve(
            "agent_runtime.response_compose"
        )
        prompt = self._prompt(
            context,
            retrieval,
            prompt_definition=prompt_definition,
        )
        prompt.append(
            {
                "role": "system",
                "content": (
                    "当前为流式回答模式。只输出最终 Markdown 回答正文，"
                    "不要输出 JSON、字段名或隐藏思维过程；"
                    "引用必须严格使用 ASCII 方括号格式，例如 [C1]。"
                ),
            }
        )
        yield SkillProgress(
            event_type="thinking.delta",
            payload={
                "delta": (
                    f"正在基于 {len(allowed)} 组已验证证据组织回答"
                ),
                "public_summary": "正在组织带引用的最终回答",
            },
        )
        validated: tuple[str, tuple[str, ...]] | None = None
        for attempt in range(1, 3):
            attempt_prompt = list(prompt)
            if attempt == 2:
                attempt_prompt.append(
                    {
                        "role": "system",
                        "content": (
                            "上一份回答未通过引用校验。请重新生成完整回答；"
                            "只能陈述证据支持的事实，并在每项事实后使用"
                            "输入中已有的 ASCII 引用标签，例如 [C1]。"
                            "如果证据不能回答问题，不得使用常识补写答案。"
                        ),
                    }
                )
            answer_parts: list[str] = []
            async for chunk in self._model_client.stream_llm_chunks(
                served_model_name=model_name,
                prompt=attempt_prompt,
                max_tokens=4096,
                temperature=0,
            ):
                if chunk.content:
                    answer_parts.append(chunk.content)
            answer_text = _normalize_citations(
                "".join(answer_parts).strip()
            )
            try:
                used_labels = self._validate_streamed_answer(
                    answer_text, allowed
                )
            except ValueError as exc:
                logger.warning(
                    "文档回答未通过引用校验 "
                    "| run_id={} | task_id={} | attempt={} | error={}",
                    context.run_id,
                    context.task_id,
                    attempt,
                    str(exc),
                )
                if attempt == 1:
                    yield SkillProgress(
                        event_type="thinking.delta",
                        payload={
                            "delta": "回答引用未通过校验，正在重新生成",
                            "public_summary": (
                                "正在修正回答中的文档引用"
                            ),
                        },
                    )
                continue
            validated = (answer_text, used_labels)
            break
        if validated is None:
            grounded = GroundedAnswer(
                answer="当前授权知识范围内没有找到足够的可引用证据。",
                status="INSUFFICIENT_EVIDENCE",
                warnings=(
                    *retrieval.warnings,
                    "回答模型连续两次未生成可验证的文档引用，"
                    "已拒绝展示无来源内容",
                ),
            )
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": 1, "delta": grounded.answer},
            )
            yield self._result(context, grounded)
            return
        answer_text, used_labels = validated
        yield SkillProgress(
            event_type="answer.delta",
            payload={"chunk_index": 1, "delta": answer_text},
        )
        references = tuple(
            ReferenceCard(
                citation_label=label,
                collection_id=allowed[label].collection_id,
                bundle_id=allowed[label].bundle_id,
                bundle_revision_id=allowed[label].bundle_revision_id,
                document_id=allowed[label].document_id,
                document_version_id=allowed[label].document_version_id,
                title=allowed[label].title,
                locator=allowed[label].locator,
                locator_schema_version=(
                    allowed[label].locator_schema_version
                ),
            )
            for label in used_labels
        )
        yield self._result(
            context,
            GroundedAnswer(
                answer=answer_text,
                status="READY",
                used_citation_labels=used_labels,
                references=references,
                warnings=retrieval.warnings,
            ),
        )

    @staticmethod
    def _validate_streamed_answer(
        answer: str, allowed: dict[str, Any]
    ) -> tuple[str, ...]:
        if not answer:
            raise ValueError("模型返回的 answer 为空")
        labels = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
        unknown = set(labels) - allowed.keys()
        if unknown:
            raise ValueError(f"模型使用了未知引用标签：{sorted(unknown)}")
        if not labels:
            raise ValueError("有文档事实的回答必须实际包含引用标签")
        return labels

    @staticmethod
    def _clarification(context: ExecutionContext) -> str | None:
        artifacts = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "CONTEXT_REWRITE"
        ]
        if not artifacts:
            return None
        payload = artifacts[-1].payload or {}
        if not bool(payload.get("ambiguity", False)):
            return None
        value = str(payload.get("clarification_question") or "").strip()
        return value or "请补充说明当前问题所指的对象。"

    @classmethod
    def _blocking_clarification(
        cls, context: ExecutionContext
    ) -> str | None:
        """文档已命中可引用证据时，不让上下文歧义阻断回答。"""
        clarification = cls._clarification(context)
        if clarification is None:
            return None
        retrieval = cls._document_result(context)
        if retrieval is not None and retrieval.citation_pack.citations:
            return None
        return clarification

    @staticmethod
    def _aiops_result(
        context: ExecutionContext,
    ) -> dict[str, Any] | None:
        artifacts = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "DELEGATED_AIOPS_RESULT"
        ]
        return dict(artifacts[-1].payload) if artifacts else None

    @staticmethod
    def _query_result(
        context: ExecutionContext,
    ) -> QueryResult | None:
        for artifact in reversed(context.input_artifacts):
            if artifact.artifact_type == "QUERY_RESULT":
                return QueryResult.model_validate(artifact.payload)
        return None

    async def _compose_query_result(
        self, context: ExecutionContext, query: QueryResult
    ) -> SkillResult:
        response = await self._query_response(context, query)
        answer = str(response.get("answer") or "").strip()
        if not answer:
            raise ValueError("问数回答模型返回空 answer")
        return self._query_result_artifact(context, query, answer)

    async def _stream_query_result(
        self, context: ExecutionContext, query: QueryResult
    ):
        model_name, messages = await self._query_prompt(context, query)
        yield SkillProgress(
            event_type="thinking.delta",
            payload={
                "delta": f"正在分析 {query.row_count} 行结构化查询结果",
                "public_summary": "正在分析结构化查询结果",
            },
        )
        parts: list[str] = []
        index = 0
        async for chunk in self._model_client.stream_llm_chunks(
            served_model_name=model_name,
            prompt=[
                *messages,
                {
                    "role": "system",
                    "content": "只输出最终回答正文，不要输出 JSON。",
                },
            ],
            max_tokens=4096,
            temperature=0,
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
            raise ValueError("问数回答模型返回空回答")
        yield self._query_result_artifact(context, query, answer)

    async def _query_response(
        self, context: ExecutionContext, query: QueryResult
    ) -> dict[str, Any]:
        model_name, messages = await self._query_prompt(context, query)
        return await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=messages,
            max_tokens=4096,
        )

    async def _query_prompt(
        self, context: ExecutionContext, query: QueryResult
    ) -> tuple[str, list[dict[str, str]]]:
        agent = context.config_snapshot.get("agent", {})
        model_name = str(
            agent_model_name(agent, "composer_llm") or ""
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 models.composer_llm")
        definition = await self._prompt_resolver.resolve(
            "agent_runtime.data_response_compose"
        )
        return model_name, [
            {
                "role": "system",
                "content": (
                    f"{definition.content}\n\nAgent 指令：\n"
                    f"{agent.get('instruction') or ''}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{context.original_input}\n"
                    f"QueryResult：{query.model_dump_json()}"
                ),
            },
        ]

    def _query_result_artifact(
        self,
        context: ExecutionContext,
        query: QueryResult,
        answer: str,
    ) -> SkillResult:
        label = "Q1"
        charts = [
            EChartsResult.model_validate(item.payload).model_dump(
                mode="json"
            )
            for item in context.input_artifacts
            if item.artifact_type == "ECHARTS_CONFIG"
        ]
        warning = (
            ("问数结果已按服务端上限截断",)
            if query.truncated
            else ()
        )
        return self._result(
            context,
            GroundedAnswer(
                answer=f"{answer} [{label}]",
                status="READY",
                used_citation_labels=(label,),
                references=(
                    QueryResultReferenceCard(
                        citation_label=label,
                        query_result_id=query.query_result_id,
                        provider=query.provider,
                        row_count=query.row_count,
                    ),
                ),
                query_results=(query.model_dump(mode="json"),),
                visualizations=tuple(charts),
                warnings=warning,
            ),
        )

    @staticmethod
    def _compose_aiops(
        context: ExecutionContext, payload: dict[str, Any]
    ) -> SkillResult:
        summary = str(payload.get("safe_summary") or "").strip()
        status = str(payload.get("status") or "FAILED")
        diagnosis = payload.get("diagnosis") or {}
        artifact = diagnosis.get("artifact") or {}
        if not summary:
            summary = "AIOps 分析已结束，但未生成可公开的诊断摘要。"
        label = "O1"
        reference = AIOpsReferenceCard(
            citation_label=label,
            ops_run_id=payload["ops_run_id"],
            delegation_id=payload["delegation_id"],
            status=status,
            resource_url=(
                f"/api/v1/apps/aiops/runs/{payload['ops_run_id']}"
            ),
            root_cause_grade=diagnosis.get("root_cause_grade"),
            artifact_id=artifact.get("artifact_id"),
            content_hash=artifact.get("content_hash"),
        )
        answer = GroundedAnswer(
            answer=f"{summary} [{label}]",
            status="READY" if status == "COMPLETED" else "PARTIAL",
            used_citation_labels=(label,),
            references=(reference,),
            warnings=(
                ()
                if status == "COMPLETED"
                else (f"AIOps 子任务以 {status} 状态结束",)
            ),
        )
        return ResponseComposerSkill._result(context, answer)

    @staticmethod
    def _document_result(
        context: ExecutionContext,
    ) -> DocumentRetrievalResult | None:
        artifacts = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "CITATION_PACK"
        ]
        if not artifacts:
            return None
        latest = artifacts[-1]
        return DocumentRetrievalResult.model_validate(latest.payload)

    @staticmethod
    def _prompt(
        context: ExecutionContext,
        retrieval: DocumentRetrievalResult,
        *,
        prompt_definition,
    ) -> list[dict[str, str]]:
        instruction = (
            context.config_snapshot.get("agent", {}).get("instruction")
            or "请基于证据准确、简洁地回答。"
        )
        evidence = [
            {
                "citation_label": item.citation_label,
                "title": item.title,
                "excerpt": item.excerpt,
                "locator": item.locator,
                "heading_path": list(item.heading_path),
            }
            for item in retrieval.citation_pack.citations
        ]
        standalone_query = context.original_input
        rewrites = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "CONTEXT_REWRITE"
        ]
        if rewrites:
            standalone_query = str(
                (rewrites[-1].payload or {}).get("standalone_query")
                or standalone_query
            )
        rendered = StrictPromptRenderer.render(
            prompt_definition,
            {
                "agent_instruction": instruction,
                "raw_input": context.original_input,
                "standalone_query": standalone_query,
                "evidence": evidence,
            },
        )
        return [{"role": "system", "content": rendered}]

    @staticmethod
    def _validate_model_answer(
        response: dict[str, Any],
        allowed: dict[str, Any],
    ) -> tuple[str, tuple[str, ...]]:
        answer = _normalize_citations(
            str(response.get("answer") or "").strip()
        )
        if not answer:
            raise ValueError("模型返回的 answer 为空")
        mentioned = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
        unknown = set(mentioned) - allowed.keys()
        if unknown:
            raise ValueError(f"模型使用了未知引用标签：{sorted(unknown)}")
        reported = tuple(
            str(value)
            for value in response.get("used_citation_labels", [])
        )
        if set(reported) != set(mentioned):
            raise ValueError("回答中的引用标签与声明的使用列表不一致")
        if not mentioned:
            raise ValueError("有文档事实的回答必须实际包含引用标签")
        return answer, mentioned

    @staticmethod
    def _result(
        context: ExecutionContext,
        answer: GroundedAnswer,
    ) -> SkillResult:
        source_ids = [
            str(item.artifact_id)
            for item in context.input_artifacts
        ]
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="GROUNDED_ANSWER",
                schema_version="GroundedAnswer.v1",
                payload=answer.model_dump(mode="json"),
                provenance={
                    "input_artifact_ids": source_ids,
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                },
                security_level=max(
                    (
                        item.security_level
                        for item in context.input_artifacts
                    ),
                    default=0,
                ),
            ),
            warnings=answer.warnings,
        )
