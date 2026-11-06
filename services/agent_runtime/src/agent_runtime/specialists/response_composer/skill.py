"""仅消费已验证 Artifact 的最终回答组合器。"""

import json
import re
from typing import Any

from loguru import logger

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.language import (
    language_instruction,
    localized_message,
    response_language,
)
from agent_runtime.runtime import (
    ExecutionContext,
    SkillArtifact,
    SkillProgress,
    SkillResult,
)
from agent_runtime.specialists.data_query.contracts import QueryResult
from agent_runtime.specialists.visualization import ChartSpec
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


def _strip_model_citations(value: str) -> str:
    """移除模型擅自生成的引用标签，由问数协议统一投影 QueryResult 引用。"""
    normalized = _normalize_citations(value)
    without_labels = _CITATION_PATTERN.sub("", normalized)
    return "\n".join(
        line.rstrip() for line in without_labels.splitlines()
    ).strip()


def _markdown_answer_deltas(value: str) -> tuple[str, ...]:
    """按 Markdown 原始行切分已校验回答，保持拼接后正文完全不变。"""
    lines = tuple(value.splitlines(keepends=True))
    return lines or (value,)


class ResponseComposerSkill:
    def __init__(self, *, model_client, prompt_resolver, app_knowledge=None):
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver
        self._app_knowledge = app_knowledge

    async def execute(self, context: ExecutionContext) -> SkillResult:
        clarification = self._blocking_clarification(context)
        if clarification:
            return self._result(context, GroundedAnswer(answer=clarification,
                status="CLARIFICATION_REQUIRED"))
        aiops = self._aiops_result(context)
        if aiops is not None:
            return self._compose_aiops(context, aiops)
        specialized = await self._compose_specialized(context)
        if specialized is not None:
            return specialized
        query = self._query_result(context)
        has_documents = any(item.artifact_type == "RETRIEVAL_EVIDENCE" for item in context.input_artifacts)
        if has_documents:
            return await self._compose_app_knowledge(context, query)
        if query is not None:
            return await self._compose_query_result(context, query)
        return self._result(context, GroundedAnswer(answer="没有可用的授权证据，未生成无来源回答。",
            status="INSUFFICIENT_EVIDENCE"))

    async def execute_stream(self, context: ExecutionContext):
        """只有通过引用闭包校验的正文才进入 SSE，避免流出后无法撤回的伪造引用。"""
        result = await self.execute(context)
        body = str(result.artifact.payload.get("answer") or "")
        for index, delta in enumerate(body.splitlines(keepends=True), 1):
            yield SkillProgress(event_type="answer.delta", payload={"chunk_index": index, "delta": delta})
        yield result

    async def _compose_app_knowledge(self, context, query):
        if self._app_knowledge is None:
            raise ValueError("App 知识 Composer 未正式装配")
        from knowledge_algorithms.contracts.runtime import KnowledgeCompositionArtifact
        from knowledge_algorithms.contracts.wire import decode, encode
        result = await self._app_knowledge.execute(context)
        composition = decode(KnowledgeCompositionArtifact, result.artifact.payload)
        grounded = composition.answer
        references = tuple(ReferenceCard(citation_label=entry.label, run_id=context.run_id,
            retrieval_ref=composition.retrieval_ref, evidence_key=entry.evidence_key.stable_key,
            document_version_id=entry.evidence_key.document_version_ref, parse_run_id=entry.evidence_key.parse_run_ref,
            owner_app_id=composition.evidence.owner_binding_ref,
            locators=tuple(encode(entry.locators)),
            role=entry.role) for entry in (grounded.references if grounded else ()))
        if grounded:
            text, status, labels = grounded.body, grounded.coverage_status, grounded.used_labels
        else:
            text = ("未命中可访问资料。" if not composition.evidence.primary_hits else
                    "已保留检索结果；本轮没有成功交付足够的合法模型载荷，未补写无来源回答。")
            status, labels = "INSUFFICIENT_EVIDENCE", ()
        query_results, visualizations, warnings = (), (), ()
        if query is not None:
            data = GroundedAnswer.model_validate((await self._compose_query_result(context, query)).artifact.payload)
            text += "\n\n" + data.answer
            references += data.references
            labels += data.used_citation_labels
            query_results, visualizations, warnings = data.query_results, data.visualizations, data.warnings
        from agent_runtime.application.knowledge_presentation import public_composition
        return self._result(context, GroundedAnswer(answer=text, status=status,
            used_citation_labels=labels, references=references, query_results=query_results,
            visualizations=visualizations, warnings=warnings,
            knowledge_composition=public_composition(result.artifact.payload)))

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
        language = response_language(
            context.config_snapshot, context.original_input
        )
        value = str(payload.get("clarification_question") or "").strip()
        if value:
            return value
        return localized_message(
            "clarify_asset_scope",
            language,
        )

    @classmethod
    def _blocking_clarification(
        cls, context: ExecutionContext
    ) -> str | None:
        """文档已命中可引用证据时，不让上下文歧义阻断回答。"""
        clarification = cls._clarification(context)
        if clarification is None:
            return None
        if any(item.artifact_type == "RETRIEVAL_EVIDENCE"
               and ((item.payload or {}).get("evidence") or {}).get("evidence_set", {}).get("primary_hits")
               for item in context.input_artifacts):
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

    async def _compose_specialized(
        self, context: ExecutionContext
    ) -> SkillResult | None:
        """供显式注册的 App 专属组合器扩展，通用实现不猜测业务。"""
        del context
        return None

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
        language = response_language(
            context.config_snapshot, context.original_input
        )
        attempt_prompt = [
            *messages,
            {
                "role": "system",
                "content": "Return only the final answer body, not JSON.",
            },
        ]
        attempt_prompt = self._answer_attempt_prompt(
            attempt_prompt,
            language=language,
            allowed_labels=(),
        )
        parts: list[str] = []
        async for chunk in self._model_client.stream_llm_chunks(
            served_model_name=model_name,
            prompt=attempt_prompt,
            temperature=0,
        ):
            if chunk.content:
                parts.append(chunk.content)
        answer = "".join(parts).strip()
        if not answer:
            raise ValueError("问数回答模型返回空回答")
        result = self._query_result_artifact(context, query, answer)
        grounded_answer = str(result.artifact.payload.get("answer") or "")
        for index, delta in enumerate(
            _markdown_answer_deltas(grounded_answer), start=1
        ):
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": index, "delta": delta},
            )
        yield result

    @staticmethod
    def _answer_attempt_prompt(messages, *, language, allowed_labels):
        return [*messages, {"role": "system", "content": language_instruction(language)
            + "\n仅允许引用标签：" + ", ".join(allowed_labels)
            + "。未提供标签时不要生成引用。"}]

    async def _query_response(
        self,
        context: ExecutionContext,
        query: QueryResult,
    ) -> dict[str, Any]:
        model_name, messages = await self._query_prompt(context, query)
        language = response_language(
            context.config_snapshot, context.original_input
        )
        return await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=self._answer_attempt_prompt(
                messages,
                language=language,
                allowed_labels=(),
            ),
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
        language = response_language(
            context.config_snapshot, context.original_input
        )
        result_scope = self._query_result_scope(query)
        return model_name, [
            {
                "role": "system",
                "content": (
                    f"{definition.content}\n\n"
                    f"{language_instruction(language)}\n"
                    "当前输入是一个受控 QueryResult，不是文档证据。"
                    "不得应用 Agent 指令中的文档引用、Bundle 或附件规则，"
                    "也不得自行生成任何引用标签；QueryResult 引用由系统添加。"
                    f"{result_scope}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"response_language={language}\n"
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
        answer_without_labels = _strip_model_citations(answer)
        if not answer_without_labels:
            raise ValueError("问数回答移除模型引用标签后为空")
        notice = self._truncation_notice(context, query)
        if notice:
            answer_without_labels = f"{answer_without_labels}\n\n{notice}"
        warning = (
            ("问数结果已按服务端上限截断",)
            if query.truncated
            else ()
        )
        return self._result(
            context,
            GroundedAnswer(
                answer=f"{answer_without_labels} [{label}]",
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
                visualizations=self._visualizations(context),
                warnings=warning,
            ),
        )

    @staticmethod
    def _query_result_scope(query: QueryResult) -> str:
        if query.truncated:
            return (
                f"结构化范围事实：当前只展示 {len(query.rows)} 行，"
                f"已观察到至少 {query.row_count} 行且仍有其他结果；"
                "未执行全量计数。必须明确说明结果已截断，"
                "不得声称这是全部结果或给出精确总数。"
            )
        return (
            f"结构化范围事实：当前返回 {len(query.rows)} 行，"
            "QueryResult 未标记截断。"
        )

    @staticmethod
    def _truncation_notice(
        context: ExecutionContext,
        query: QueryResult,
    ) -> str:
        if not query.truncated:
            return ""
        language = response_language(
            context.config_snapshot, context.original_input
        )
        if language.startswith("zh"):
            return (
                f"当前显示前 {len(query.rows)} 条，仍有其他结果；"
                "本次未执行全量计数。"
            )
        if language.startswith("ko"):
            return (
                f"현재 처음 {len(query.rows)}개 결과만 표시하며 더 많은 "
                "결과가 있습니다. 전체 건수는 계산하지 않았습니다."
            )
        if language.startswith("ja"):
            return (
                f"現在は先頭 {len(query.rows)} 件のみを表示しており、"
                "ほかにも結果があります。全件数は集計していません。"
            )
        return (
            f"Showing the first {len(query.rows)} results; more "
            "results exist, and no full count was run."
        )

    @staticmethod
    def _visualizations(
        context: ExecutionContext,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            ChartSpec.model_validate(item.payload).model_dump(
                mode="json"
            )
            for item in context.input_artifacts
            if item.artifact_type == "CHART_SPEC"
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
