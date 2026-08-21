"""仅消费已验证 Artifact 的最终回答组合器。"""

import json
import re
from typing import Any

from loguru import logger

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.language import (
    answer_matches_language,
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
from agent_runtime.specialists.document.contracts import (
    DocumentRetrievalResult,
)
from agent_runtime.specialists.data_query.contracts import QueryResult
from agent_runtime.specialists.visualization import EChartsResult
from platform_core.contracts import AssetSearchPlanV1
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
        retrieval = self._document_result(context)
        asset_search_plan = self._asset_search_plan(context)
        if (
            query_result is not None
            and retrieval is not None
            and asset_search_plan is not None
        ):
            if asset_search_plan.operation in {"COUNT", "GROUP"}:
                return await self._compose_asset_aggregate_with_evidence(
                    context, query_result, retrieval, asset_search_plan
                )
            return await self._compose_km_asset_enumeration(
                context, query_result, retrieval,
                search_plan=asset_search_plan,
            )
        if query_result is not None:
            if asset_search_plan is not None:
                return await self._compose_asset_query_result(
                    context, query_result, asset_search_plan
                )
            return await self._compose_query_result(context, query_result)
        if retrieval is None or not retrieval.citation_pack.citations:
            retrieval_warnings = (
                retrieval.warnings if retrieval is not None else ()
            )
            answer = GroundedAnswer(
                answer=localized_message(
                    "insufficient_evidence",
                    response_language(
                        context.config_snapshot, context.original_input
                    ),
                ),
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
        language = response_language(
            context.config_snapshot, context.original_input
        )
        validated: tuple[str, tuple[str, ...]] | None = None
        last_error: ValueError | None = None
        for attempt in range(1, 3):
            attempt_prompt = self._answer_attempt_prompt(
                prompt, language=language, repair=attempt == 2
            )
            response = await self._model_client.get_llm_json(
                served_model_name=model_name,
                prompt=attempt_prompt,
            )
            try:
                validated = self._validate_model_answer(
                    response,
                    allowed,
                    language=language,
                )
                break
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "文档回答未通过最终校验 "
                    "| run_id={} | task_id={} | attempt={} | error={}",
                    context.run_id,
                    context.task_id,
                    attempt,
                    str(exc),
                )
        if validated is None:
            warning = str(
                last_error or ValueError("文档回答未通过最终校验")
            )
            return self._result(
                context,
                self._verified_source_fallback(
                    retrieval,
                    language=language,
                    validation_warning=warning,
                ),
            )
        answer_text, used_labels = validated
        answer_text, used_labels = self._append_asset_supporting_list(
            answer_text,
            used_labels,
            allowed,
            search_plan=asset_search_plan,
            language=language,
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
        retrieval = self._document_result(context)
        asset_search_plan = self._asset_search_plan(context)
        if (
            query_result is not None
            and retrieval is not None
            and asset_search_plan is not None
        ):
            if asset_search_plan.operation in {"COUNT", "GROUP"}:
                result = await self._compose_asset_aggregate_with_evidence(
                    context, query_result, retrieval, asset_search_plan
                )
                answer = str(result.artifact.payload.get("answer") or "")
                yield SkillProgress(
                    event_type="answer.delta",
                    payload={"chunk_index": 1, "delta": answer},
                )
                yield result
                return
            result = await self._compose_km_asset_enumeration(
                context, query_result, retrieval,
                search_plan=asset_search_plan,
            )
            answer = str(result.artifact.payload.get("answer") or "")
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": 1, "delta": answer},
            )
            yield result
            return
        if query_result is not None:
            if asset_search_plan is not None:
                result = await self._compose_asset_query_result(
                    context, query_result, asset_search_plan
                )
                answer = str(result.artifact.payload.get("answer") or "")
                yield SkillProgress(
                    event_type="answer.delta",
                    payload={"chunk_index": 1, "delta": answer},
                )
                yield result
                return
            async for item in self._stream_query_result(
                context, query_result
            ):
                yield item
            return
        if retrieval is None or not retrieval.citation_pack.citations:
            retrieval_warnings = (
                retrieval.warnings if retrieval is not None else ()
            )
            grounded = GroundedAnswer(
                answer=localized_message(
                    "insufficient_evidence",
                    response_language(
                        context.config_snapshot, context.original_input
                    ),
                ),
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
        language = response_language(
            context.config_snapshot, context.original_input
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
            attempt_prompt = self._answer_attempt_prompt(
                prompt, language=language, repair=attempt == 2
            )
            answer_parts: list[str] = []
            async for chunk in self._model_client.stream_llm_chunks(
                served_model_name=model_name,
                prompt=attempt_prompt,
                temperature=0,
            ):
                if chunk.content:
                    answer_parts.append(chunk.content)
            answer_text = _normalize_citations(
                "".join(answer_parts).strip()
            )
            try:
                used_labels = self._validate_streamed_answer(
                    answer_text,
                    allowed,
                    language=language,
                )
            except ValueError as exc:
                logger.warning(
                    "文档回答未通过最终校验 "
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
                            "delta": "回答未通过语言或引用校验，正在重新生成",
                            "public_summary": (
                                "正在修正回答语言与文档引用"
                            ),
                        },
                    )
                continue
            validated = (answer_text, used_labels)
            break
        if validated is None:
            grounded = self._verified_source_fallback(
                retrieval,
                language=language,
                validation_warning=(
                    "回答模型连续两次未生成通过校验的文档回答"
                ),
            )
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": 1, "delta": grounded.answer},
            )
            yield self._result(context, grounded)
            return
        answer_text, used_labels = validated
        answer_text, used_labels = self._append_asset_supporting_list(
            answer_text,
            used_labels,
            allowed,
            search_plan=asset_search_plan,
            language=language,
        )
        for index, delta in enumerate(
            _markdown_answer_deltas(answer_text), start=1
        ):
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": index, "delta": delta},
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
        answer: str,
        allowed: dict[str, Any],
        *,
        language: str = "zh-CN",
    ) -> tuple[str, ...]:
        if not answer:
            raise ValueError("模型返回的 answer 为空")
        labels = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
        unknown = set(labels) - allowed.keys()
        if unknown:
            raise ValueError(f"模型使用了未知引用标签：{sorted(unknown)}")
        if not labels:
            raise ValueError("有文档事实的回答必须实际包含引用标签")
        if not answer_matches_language(
            answer,
            language,
            ignored_texts=(
                str(getattr(item, "title", "") or "")
                for item in allowed.values()
            ),
        ):
            raise ValueError(f"回答语言与 language={language} 不一致")
        return labels

    @staticmethod
    def _verified_source_fallback(
        retrieval: DocumentRetrievalResult,
        *,
        language: str,
        validation_warning: str,
    ) -> GroundedAnswer:
        """生成失败时不投影未经模型确认相关的 Bundle。"""
        return GroundedAnswer(
            answer=localized_message("citation_validation_failed", language),
            status="ANSWER_VALIDATION_FAILED",
            warnings=(
                *retrieval.warnings,
                validation_warning,
            ),
        )

    @staticmethod
    def _reference_card(item: Any) -> ReferenceCard:
        """把已验证 Citation 投影为公开引用卡片。"""
        return ReferenceCard(
            citation_label=item.citation_label,
            collection_id=item.collection_id,
            bundle_id=item.bundle_id,
            bundle_revision_id=item.bundle_revision_id,
            document_id=item.document_id,
            document_version_id=item.document_version_id,
            title=item.title,
            locator=item.locator,
            locator_schema_version=item.locator_schema_version,
        )

    @staticmethod
    def _answer_attempt_prompt(
        prompt: list[dict[str, str]],
        *,
        language: str,
        repair: bool,
    ) -> list[dict[str, str]]:
        """确保冻结语言约束始终是模型看到的最后一条系统指令。"""
        messages = list(prompt)
        if repair:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The previous answer failed final validation. Generate "
                        "the complete answer again using only supported facts "
                        "and the supplied ASCII citation labels such as [C1]."
                    ),
                }
            )
        messages.append(
            {"role": "system", "content": language_instruction(language)}
        )
        return messages

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
        if value and answer_matches_language(value, language):
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
    def _asset_search_plan(
        context: ExecutionContext,
    ) -> AssetSearchPlanV1 | None:
        """读取 Root 冻结的统一 Asset 搜索计划。"""
        route = context.config_snapshot.get("route") or {}
        raw = (
            route.get("asset_search_plan")
            if isinstance(route, dict)
            else getattr(route, "asset_search_plan", None)
        )
        return AssetSearchPlanV1.model_validate(raw) if raw else None

    @staticmethod
    def _document_scope(context: ExecutionContext) -> dict[str, Any]:
        artifact = next(
            (
                item for item in reversed(context.input_artifacts)
                if item.artifact_type == "DOCUMENT_SCOPE"
            ),
            None,
        )
        return dict(artifact.payload or {}) if artifact is not None else {}

    @staticmethod
    def _enumeration_assets_from_query(
        query: QueryResult,
    ) -> list[dict[str, Any]]:
        """当编排输入缺少范围 Artifact 时，从同一问数结果恢复清单。"""
        fields = (
            "asset_id", "title", "product", "solution",
            "bundle_id", "bundle_revision_id",
        )
        assets: list[dict[str, Any]] = []
        for row in query.rows[:10]:
            folded = {str(key).casefold(): value for key, value in row.items()}
            assets.append({name: folded.get(name.casefold()) for name in fields})
        return assets

    @staticmethod
    def _enumeration_prefix(
        *, language: str, total_count: int, shown_count: int,
        truncated: bool, source_truncated: bool,
    ) -> str:
        if language.startswith("zh"):
            if source_truncated:
                return (
                    f"问数结果至少命中 {total_count} 个相关 Asset；"
                    f"以下列出前 {shown_count} 个，其他结果已截断。[Q1]"
                )
            if truncated:
                return (
                    f"问数结果共命中 {total_count} 个相关 Asset；"
                    f"以下列出前 {shown_count} 个，其他结果已截断。[Q1]"
                )
            return (
                f"问数结果共命中 {total_count} 个相关 Asset，"
                f"以下全部列出。[Q1]"
            )
        if source_truncated:
            return (
                f"The data query found at least {total_count} related assets. "
                f"The first {shown_count} are listed; the rest are truncated. [Q1]"
            )
        if truncated:
            return (
                f"The data query found {total_count} related assets. "
                f"The first {shown_count} are listed; the rest are truncated. [Q1]"
            )
        return (
            f"The data query found {total_count} related assets; all are "
            "listed below. [Q1]"
        )

    @staticmethod
    def _validate_enumeration_body(
        answer: str,
        *,
        assets: list[dict[str, Any]],
        allowed: dict[str, Any],
        language: str,
    ) -> None:
        if not answer:
            raise ValueError("主题 Asset 清单为空")
        if re.match(r"^\s*[\[{]\s*(?:[\{\[\"']|$)", answer):
            raise ValueError("主题 Asset 清单不得输出序列化容器")
        leaked_asset_ids = [
            str(item.get("asset_id") or "")
            for item in assets
            if str(item.get("asset_id") or "")
            and str(item.get("asset_id")) in answer
        ]
        if leaked_asset_ids or re.search(
            r"(?i)['\"]?asset_id['\"]?\s*:", answer
        ):
            raise ValueError("主题 Asset 清单不得显示 asset_id")
        missing_titles = [
            str(item.get("title") or "")
            for item in assets
            if str(item.get("title") or "") not in answer
        ]
        if missing_titles:
            raise ValueError(f"主题 Asset 清单缺少标题：{missing_titles}")
        labels = set(_CITATION_PATTERN.findall(answer))
        unknown = labels - allowed.keys() - {"Q1"}
        if unknown:
            raise ValueError(f"主题 Asset 清单使用未知引用：{sorted(unknown)}")
        if allowed and not labels.intersection(allowed):
            raise ValueError("已有正文证据但清单未使用文档引用")
        labels_by_bundle = (
            ResponseComposerSkill._citation_labels_by_bundle(allowed)
        )
        title_positions = [
            answer.index(str(item.get("title") or ""))
            for item in assets
        ]
        if title_positions != sorted(title_positions):
            raise ValueError("主题 Asset 清单顺序与问数结果不一致")
        for index, item in enumerate(assets):
            segment_end = (
                title_positions[index + 1]
                if index + 1 < len(title_positions)
                else len(answer)
            )
            segment_labels = set(_CITATION_PATTERN.findall(
                answer[title_positions[index]:segment_end]
            )) & allowed.keys()
            expected_labels = set(labels_by_bundle.get(
                str(item.get("bundle_id") or "").casefold(), ()
            ))
            if expected_labels and not segment_labels.intersection(
                expected_labels
            ):
                raise ValueError(
                    "Asset 清单缺少对应 Bundle 的正文引用："
                    f"{item.get('title')}"
                )
            if segment_labels - expected_labels:
                raise ValueError(
                    "Asset 清单使用了其他 Bundle 的正文引用："
                    f"{item.get('title')}"
                )
        if not answer_matches_language(
            answer,
            language,
            ignored_texts=(
                str(item.get(field) or "")
                for item in assets
                for field in ("title", "product", "solution")
            ),
        ):
            raise ValueError(f"主题 Asset 清单语言与 language={language} 不一致")

    @staticmethod
    def _enumeration_fallback(
        assets: list[dict[str, Any]], *, language: str,
        allowed: dict[str, Any] | None = None,
        query_label: str = "Q1",
    ) -> str:
        labels_by_bundle = ResponseComposerSkill._citation_labels_by_bundle(
            allowed or {}
        )
        if not assets:
            return "未找到匹配的 Asset。" if language.startswith("zh") else (
                "No matching assets were found."
            )
        lines = []
        for index, item in enumerate(assets, start=1):
            title = str(item.get("title") or item.get("asset_id") or "Asset")
            product = str(item.get("product") or "").strip()
            solution = str(item.get("solution") or "").strip()
            if language.startswith("zh"):
                details = "；".join(
                    value for value in (
                        f"产品：{product}" if product else "",
                        f"解决方案：{solution}" if solution else "",
                    ) if value
                )
            else:
                details = "; ".join(
                    value for value in (
                        f"Product: {product}" if product else "",
                        f"Solution: {solution}" if solution else "",
                    ) if value
                )
            suffix = f" — {details}" if details else ""
            citation_labels = labels_by_bundle.get(
                str(item.get("bundle_id") or "").casefold(), ()
            )
            references = " ".join(
                (
                    f"[{query_label}]",
                    *(f"[{label}]" for label in citation_labels[:1]),
                )
            )
            lines.append(f"{index}. **{title}**{suffix} {references}")
        return "\n".join(lines)

    @staticmethod
    def _citation_labels_by_bundle(
        allowed: dict[str, Any],
    ) -> dict[str, tuple[str, ...]]:
        """按 Bundle 归并 Citation，确保每个 Asset 只使用自己的正文证据。"""
        grouped: dict[str, list[str]] = {}
        for label, citation in allowed.items():
            bundle_id = str(getattr(citation, "bundle_id", "") or "")
            if bundle_id:
                grouped.setdefault(bundle_id.casefold(), []).append(label)
        return {
            bundle_id: tuple(labels)
            for bundle_id, labels in grouped.items()
        }

    @staticmethod
    def _append_asset_supporting_list(
        answer: str,
        used_labels: tuple[str, ...],
        allowed: dict[str, Any],
        *,
        search_plan: AssetSearchPlanV1 | None,
        language: str,
    ) -> tuple[str, tuple[str, ...]]:
        """纯文档回答附带 3 到 5 个可追溯 Asset，不放宽实际命中。"""
        if search_plan is None or search_plan.operation == "LIST":
            return answer, used_labels
        target = search_plan.result_assets.target_count
        selected: list[tuple[str, Any]] = []
        seen_bundles: set[str] = set()
        for label, citation in allowed.items():
            bundle_key = str(getattr(citation, "bundle_id", "") or "")
            if not bundle_key or bundle_key in seen_bundles:
                continue
            selected.append((label, citation))
            seen_bundles.add(bundle_key)
            if len(selected) >= target:
                break
        if not selected:
            return answer, used_labels
        heading = "相关 Asset：" if language.startswith("zh") else (
            "Related assets:"
        )
        lines = []
        for index, (label, citation) in enumerate(selected, start=1):
            title = str(
                getattr(citation, "bundle_title", None)
                or getattr(citation, "title", None)
                or "Asset"
            )
            lines.append(f"{index}. **{title}** [{label}]")
        labels = tuple(dict.fromkeys((
            *used_labels,
            *(label for label, _ in selected),
        )))
        return f"{answer}\n\n{heading}\n\n" + "\n".join(lines), labels

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
        language = response_language(
            context.config_snapshot, context.original_input
        )
        answer = ""
        for attempt in range(1, 3):
            response = await self._query_response(
                context, query, repair=attempt == 2
            )
            answer = str(response.get("answer") or "").strip()
            if answer and answer_matches_language(answer, language):
                break
        if not answer:
            raise ValueError("问数回答模型返回空 answer")
        if not answer_matches_language(answer, language):
            raise ValueError(f"问数回答语言与 language={language} 不一致")
        return self._query_result_artifact(context, query, answer)

    async def _compose_km_asset_enumeration(
        self,
        context: ExecutionContext,
        query: QueryResult,
        retrieval: DocumentRetrievalResult,
        *,
        search_plan: AssetSearchPlanV1 | None = None,
        query_label: str = "Q1",
    ) -> SkillResult:
        """用问数冻结资格边界，再按同 Bundle 正文证据确定最终清单。"""
        scope = self._document_scope(context)
        candidate_assets = [
            dict(item) for item in scope.get("assets") or ()
            if isinstance(item, dict)
        ]
        if not candidate_assets:
            candidate_assets = self._enumeration_assets_from_query(query)
        citations = retrieval.citation_pack.citations
        allowed = {item.citation_label: item for item in citations}
        asset_by_bundle = {
            str(item.get("bundle_id") or "").casefold(): item
            for item in candidate_assets
            if str(item.get("bundle_id") or "")
        }
        assets: list[dict[str, Any]] = []
        seen_bundles: set[str] = set()
        result_limit = (
            search_plan.result_assets.target_count
            if search_plan is not None
            else 10
        )
        for citation in citations:
            bundle_key = str(citation.bundle_id).casefold()
            asset = asset_by_bundle.get(bundle_key)
            if asset is None or bundle_key in seen_bundles:
                continue
            assets.append(asset)
            seen_bundles.add(bundle_key)
            if len(assets) >= result_limit:
                break
        if search_plan is not None and search_plan.preferences:
            for asset in candidate_assets:
                bundle_key = str(asset.get("bundle_id") or "").casefold()
                if not bundle_key or bundle_key in seen_bundles:
                    continue
                assets.append(asset)
                seen_bundles.add(bundle_key)
                if len(assets) >= result_limit:
                    break
        if query.row_count > 0 and not candidate_assets:
            raise ValueError("KM_ASSET_ENUMERATION_LIST_EMPTY")
        language = response_language(
            context.config_snapshot, context.original_input
        )
        semantic = bool(
            search_plan is not None
            and (
                search_plan.has_semantic_eligibility
                or search_plan.preferences
            )
        )
        if semantic and not assets:
            return self._result(context, GroundedAnswer(
                answer=localized_message("insufficient_evidence", language),
                status="INSUFFICIENT_EVIDENCE",
                warnings=tuple(dict.fromkeys((
                    *retrieval.warnings,
                    "没有 Asset 同时满足资格边界与正文证据要求",
                ))),
            ))
        if semantic:
            semantic_count_fallback = bool(
                search_plan is not None
                and "SEMANTIC_TOTAL_COUNT" in search_plan.unsupported_requests
            )
            if semantic_count_fallback:
                prefix = (
                    f"语义相关性不能用于精确统计总数；以下提供 {len(assets)} 个"
                    "较新且有正文证据的相关 Asset 供参考。"
                    if language.startswith("zh")
                    else (
                        "Semantic relevance cannot produce an exact total. "
                        f"Here are {len(assets)} recent relevant assets with "
                        "content evidence."
                    )
                )
            else:
                if citations:
                    prefix = (
                        f"以下是 {len(assets)} 个满足条件的 Asset；"
                        "正文证据已用于语义条件或偏好排序。"
                        if language.startswith("zh")
                        else (
                            f"Here are {len(assets)} matching assets; content "
                            "evidence was used for semantic conditions or preferences."
                        )
                    )
                else:
                    prefix = (
                        f"以下是 {len(assets)} 个满足精确条件的 Asset；"
                        "未找到可证明软偏好的正文证据，因此保留原排序。"
                        if language.startswith("zh")
                        else (
                            f"Here are {len(assets)} assets matching the exact "
                            "criteria. No content evidence proved the soft preference, "
                            "so the original order is retained."
                        )
                    )
        else:
            total_count = int(scope.get("total_count") or query.row_count)
            prefix = self._enumeration_prefix(
                language=language,
                total_count=total_count,
                shown_count=len(assets),
                truncated=bool(scope.get("truncated") or query.truncated),
                source_truncated=not bool(
                    query.provenance.get("count_exact", True)
                ),
            )
        body = ""
        model_name = str(agent_model_name(
            context.config_snapshot.get("agent", {}), "composer_llm"
        ) or "").strip()
        if model_name and assets:
            prompt = await self._prompt_resolver.resolve(
                "agent_runtime.km_asset_enumeration_compose"
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"{prompt.content}\n\n{language_instruction(language)}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "question": context.original_input,
                        "operation": (
                            search_plan.operation if search_plan else "LIST"
                        ),
                        "answer_detail": (
                            search_plan.answer_detail if search_plan else "BRIEF"
                        ),
                        "assets": assets,
                        "citations": [
                            {
                                "citation_label": item.citation_label,
                                "bundle_id": str(item.bundle_id),
                                "title": item.title,
                                "excerpt": item.excerpt,
                            }
                            for item in citations
                        ],
                    }, ensure_ascii=False, default=str),
                },
            ]
            for attempt in range(2):
                response = await self._model_client.get_llm_json(
                    served_model_name=model_name,
                    prompt=messages,
                )
                candidate = _normalize_citations(
                    str(response.get("answer") or "").strip()
                    if isinstance(response, dict) else ""
                )
                try:
                    self._validate_enumeration_body(
                        candidate,
                        assets=assets,
                        allowed=allowed,
                        language=language,
                    )
                    body = candidate
                    break
                except ValueError as exc:
                    if attempt == 0:
                        messages.append({
                            "role": "system",
                            "content": (
                                "上一份清单未通过校验，请完整重写所有 Asset。"
                                f"错误：{exc}"
                            ),
                        })
        if not body:
            body = self._enumeration_fallback(
                assets,
                language=language,
                allowed=allowed,
                query_label=query_label,
            )
        else:
            for asset in assets:
                title = str(asset.get("title") or "")
                if title:
                    body = body.replace(title, f"{title} [{query_label}]", 1)
        answer = f"{prefix}\n\n{body}".strip()
        used_labels = tuple(dict.fromkeys(
            label for label in _CITATION_PATTERN.findall(body)
            if label in allowed
        ))
        references = (
            QueryResultReferenceCard(
                citation_label=query_label,
                query_result_id=query.query_result_id,
                provider=query.provider,
                row_count=query.row_count,
            ),
            *(
                self._reference_card(allowed[label])
                for label in used_labels
            ),
        )
        warnings = list(retrieval.warnings)
        if bool(scope.get("truncated")):
            display_limit = int(scope.get("display_limit") or len(assets) or 10)
            warnings.append(
                f"相关 Asset 超过 {display_limit} 个，回答已按请求数量截断"
            )
        return self._result(context, GroundedAnswer(
            answer=answer,
            status="READY",
            used_citation_labels=(query_label, *used_labels),
            references=references,
            query_results=(query.model_dump(mode="json"),),
            warnings=tuple(dict.fromkeys(warnings)),
        ))

    async def _compose_asset_aggregate_with_evidence(
        self,
        context: ExecutionContext,
        query: QueryResult,
        retrieval: DocumentRetrievalResult,
        search_plan: AssetSearchPlanV1,
    ) -> SkillResult:
        """组合精确聚合 Q1 与经语义偏好排序的支撑 Asset Q2/Cn。"""
        aggregate = await self._compose_query_result(context, query)
        aggregate_answer = GroundedAnswer.model_validate(
            aggregate.artifact.payload
        )
        if not query.supporting_rows:
            return aggregate
        sample = query.model_copy(update={
            "query_result_id": (
                query.supporting_query_result_id or query.query_result_id
            ),
            "columns": query.supporting_columns,
            "rows": query.supporting_rows,
            "row_count": len(query.supporting_rows),
            "truncated": False,
            "supporting_columns": (),
            "supporting_rows": (),
            "supporting_query_result_id": None,
        })
        support_result = await self._compose_km_asset_enumeration(
            context,
            sample,
            retrieval,
            search_plan=search_plan,
            query_label="Q2",
        )
        support_answer = GroundedAnswer.model_validate(
            support_result.artifact.payload
        )
        if support_answer.status != "READY":
            return self._result(context, aggregate_answer.model_copy(update={
                "warnings": tuple(dict.fromkeys((
                    *aggregate_answer.warnings,
                    *support_answer.warnings,
                ))),
            }))
        return self._result(context, GroundedAnswer(
            answer=f"{aggregate_answer.answer}\n\n{support_answer.answer}",
            status="READY",
            used_citation_labels=(
                *aggregate_answer.used_citation_labels,
                *support_answer.used_citation_labels,
            ),
            references=(
                *aggregate_answer.references,
                *support_answer.references,
            ),
            query_results=(
                query.model_copy(update={
                    "supporting_columns": (),
                    "supporting_rows": (),
                    "supporting_query_result_id": None,
                }).model_dump(mode="json"),
                sample.model_dump(mode="json"),
            ),
            warnings=tuple(dict.fromkeys((
                *aggregate_answer.warnings,
                *support_answer.warnings,
            ))),
        ))

    async def _compose_asset_query_result(
        self,
        context: ExecutionContext,
        query: QueryResult,
        search_plan: AssetSearchPlanV1,
    ) -> SkillResult:
        """确定性展示 Asset 清单，并为聚合问数附带同范围样例。"""
        if search_plan.operation == "LIST":
            assets = self._enumeration_assets_from_query(query)[
                :search_plan.result_assets.target_count
            ]
            language = response_language(
                context.config_snapshot, context.original_input
            )
            prefix = self._enumeration_prefix(
                language=language,
                total_count=query.row_count,
                shown_count=len(assets),
                truncated=query.truncated,
                source_truncated=not bool(
                    query.provenance.get("count_exact", True)
                ),
            )
            body = self._enumeration_fallback(
                assets, language=language, allowed={}
            )
            return self._result(context, GroundedAnswer(
                answer=f"{prefix}\n\n{body}",
                status="READY",
                used_citation_labels=("Q1",),
                references=(QueryResultReferenceCard(
                    citation_label="Q1",
                    query_result_id=query.query_result_id,
                    provider=query.provider,
                    row_count=query.row_count,
                ),),
                query_results=(query.model_dump(mode="json"),),
                warnings=(
                    ("问数结果已按服务端上限截断",)
                    if query.truncated else ()
                ),
            ))

        base = await self._compose_query_result(context, query)
        if not query.supporting_rows:
            return base
        payload = GroundedAnswer.model_validate(base.artifact.payload)
        sample = query.model_copy(update={
            "query_result_id": (
                query.supporting_query_result_id or query.query_result_id
            ),
            "rows": query.supporting_rows,
            "columns": query.supporting_columns,
            "row_count": len(query.supporting_rows),
            "truncated": False,
            "supporting_columns": (),
            "supporting_rows": (),
            "supporting_query_result_id": None,
            "provenance": {
                **query.provenance,
                "supporting_of": str(query.query_result_id),
            },
        })
        assets = self._enumeration_assets_from_query(sample)[
            :search_plan.result_assets.target_count
        ]
        language = response_language(
            context.config_snapshot, context.original_input
        )
        heading = "同条件下的较新 Asset：" if language.startswith("zh") else (
            "Recent assets from the same result scope:"
        )
        asset_lines = self._enumeration_fallback(
            assets, language=language, allowed={}, query_label="Q2"
        )
        answer = f"{payload.answer}\n\n{heading}\n\n{asset_lines}"
        return self._result(context, payload.model_copy(update={
            "answer": answer,
            "used_citation_labels": (*payload.used_citation_labels, "Q2"),
            "references": (
                *payload.references,
                QueryResultReferenceCard(
                    citation_label="Q2",
                    query_result_id=sample.query_result_id,
                    provider=sample.provider,
                    row_count=sample.row_count,
                ),
            ),
            "query_results": (
                query.model_copy(update={
                    "supporting_columns": (),
                    "supporting_rows": (),
                    "supporting_query_result_id": None,
                }).model_dump(mode="json"),
                sample.model_dump(mode="json"),
            ),
        }))

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
        answer = ""
        for attempt in range(1, 3):
            attempt_prompt = [
                *messages,
                {
                    "role": "system",
                    "content": "Return only the final answer body, not JSON.",
                },
            ]
            attempt_prompt = self._answer_attempt_prompt(
                attempt_prompt, language=language, repair=attempt == 2
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
            if answer and answer_matches_language(answer, language):
                break
        if not answer:
            raise ValueError("问数回答模型返回空回答")
        if not answer_matches_language(answer, language):
            raise ValueError(f"问数回答语言与 language={language} 不一致")
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

    async def _query_response(
        self,
        context: ExecutionContext,
        query: QueryResult,
        *,
        repair: bool = False,
    ) -> dict[str, Any]:
        model_name, messages = await self._query_prompt(context, query)
        language = response_language(
            context.config_snapshot, context.original_input
        )
        return await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=self._answer_attempt_prompt(
                messages, language=language, repair=repair
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
        result_scope = (
            f"结构化范围事实：当前只展示 {len(query.rows)} 行，"
            f"已观察到至少 {query.row_count} 行且仍有其他结果；"
            "未执行全量计数。必须明确说明结果已截断，"
            "不得声称这是全部结果或给出精确总数。"
            if query.truncated
            else (
                f"结构化范围事实：当前返回 {len(query.rows)} 行，"
                "QueryResult 未标记截断。"
            )
        )
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
                    f"language={language}\n"
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
        if query.truncated:
            language = response_language(
                context.config_snapshot, context.original_input
            )
            if language.startswith("zh"):
                scope_notice = (
                    f"当前显示前 {len(query.rows)} 条，仍有其他结果；"
                    "本次未执行全量计数。"
                )
            elif language.startswith("ko"):
                scope_notice = (
                    f"현재 처음 {len(query.rows)}개 결과만 표시하며 더 많은 "
                    "결과가 있습니다. 전체 건수는 계산하지 않았습니다."
                )
            elif language.startswith("ja"):
                scope_notice = (
                    f"現在は先頭 {len(query.rows)} 件のみを表示しており、"
                    "ほかにも結果があります。全件数は集計していません。"
                )
            else:
                scope_notice = (
                    f"Showing the first {len(query.rows)} results; more "
                    "results exist, and no full count was run."
                )
            answer_without_labels = (
                f"{answer_without_labels}\n\n{scope_notice}"
            )
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
                "bundle_id": str(item.bundle_id),
                "bundle_title": item.bundle_title or item.title,
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
        language = response_language(
            context.config_snapshot, context.original_input
        )
        messages = [
            {"role": "system", "content": rendered},
            {
                "role": "system",
                "content": (
                    "只回答与用户问题相关且得到证据支持的内容。不得在回答中"
                    "列出、讨论或对比不匹配的候选；只使用正文实际需要的引用，"
                    "未使用的证据不得出现在 used_citation_labels 或引用列表中。"
                ),
            },
            {"role": "system", "content": language_instruction(language)},
        ]
        return messages

    @staticmethod
    def _validate_model_answer(
        response: dict[str, Any],
        allowed: dict[str, Any],
        *,
        language: str = "zh-CN",
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
        if not answer_matches_language(
            answer,
            language,
            ignored_texts=(
                str(getattr(item, "title", "") or "")
                for item in allowed.values()
            ),
        ):
            raise ValueError(f"回答语言与 language={language} 不一致")
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
