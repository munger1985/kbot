"""KM Asset 对通用 Agent Skill 的专属实现入口。"""

from __future__ import annotations

import re

from loguru import logger

from agent_runtime.specialists.data_query import DataQuerySkill
from agent_runtime.specialists.document import KnowledgeRetrievalSkill
from agent_runtime.specialists.response_composer import ResponseComposerSkill
from agent_runtime.specialists.response_composer.contracts import GroundedAnswer

from .document_scope import KmAssetDocumentScopeExtractSkill
from .composer import KmAssetComposerMixin
from .retrieval import KmAssetRetrievalMixin


_CITATION_LABEL_PATTERN = re.compile(r"\[[A-Z]\d+\]")


class KmAssetKnowledgeRetrievalSkill(
    KmAssetRetrievalMixin, KnowledgeRetrievalSkill
):
    """执行 KM Asset Search Plan 与同 Bundle KC 取证。"""

    @staticmethod
    def _prefer_evidence_order(context):
        route = context.config_snapshot.get("route") or {}
        plan = route.get("asset_search_plan") if isinstance(route, dict) else None
        if not isinstance(plan, dict):
            return False
        unsupported = set(plan.get("unsupported_requests") or ())
        return "SEMANTIC_TOTAL_COUNT" not in unsupported

    async def _retrieve_evidence(
        self,
        *,
        context,
        query,
        candidates,
        scoped,
        retrieval_config,
        coverage_mode,
    ):
        search_plan = self._asset_search_plan(context)
        if search_plan is not None and scoped:
            return await self._retrieve_asset_plan_evidence(
                context=context,
                plan=search_plan,
                candidates=candidates,
                retrieval_config=retrieval_config,
                coverage_mode=coverage_mode,
            )
        return await super()._retrieve_evidence(
            context=context,
            query=query,
            candidates=candidates,
            scoped=scoped,
            retrieval_config=retrieval_config,
            coverage_mode=coverage_mode,
        )


class KmAssetResponseComposerSkill(
    KmAssetComposerMixin, ResponseComposerSkill
):
    """组合 KM Asset 问文、问数和引用结果。"""

    @staticmethod
    def _result(context, answer):
        """内部诊断不作为 KM Agent 的用户提示展示。"""
        if answer.warnings:
            logger.debug(
                "KM Asset 回答已隐藏内部提示 | run_id={} | task_id={} | "
                "status={} | warning_count={}",
                context.run_id,
                context.task_id,
                answer.status,
                len(answer.warnings),
            )
            answer = answer.model_copy(update={"warnings": ()})
        return ResponseComposerSkill._result(context, answer)

    async def _compose_specialized(self, context):
        query_result = self._query_result(context)
        retrieval = self._document_result(context)
        search_plan = self._asset_search_plan(context)
        if search_plan is None or query_result is None:
            return None
        if retrieval is not None:
            if search_plan.operation == "ANSWER":
                return await self._compose_answer_with_assets(
                    context, retrieval, search_plan
                )
            if search_plan.operation in {"COUNT", "GROUP"}:
                return await self._compose_asset_aggregate_with_evidence(
                    context, query_result, retrieval, search_plan
                )
            return await self._compose_km_asset_enumeration(
                context,
                query_result,
                retrieval,
                search_plan=search_plan,
            )
        return await self._compose_asset_query_result(
            context, query_result, search_plan
        )

    async def _compose_answer_with_assets(
        self, context, retrieval, search_plan
    ):
        """先回答正文问题，再附加同一证据范围内的相关 Asset。"""
        document_context = context.model_copy(update={
            "input_artifacts": tuple(
                artifact for artifact in context.input_artifacts
                if artifact.artifact_type != "QUERY_RESULT"
            ),
        })
        document_result = await ResponseComposerSkill.execute(
            self, document_context
        )
        grounded = GroundedAnswer.model_validate(
            document_result.artifact.payload
        )
        if grounded.status != "READY":
            fallback = self._exact_answer_source_fallback(
                context=context,
                retrieval=retrieval,
                search_plan=search_plan,
                grounded=grounded,
            )
            if fallback is not None:
                return self._result(context, fallback)
            return self._result(context, grounded)
        allowed = {
            item.citation_label: item
            for item in retrieval.citation_pack.citations
        }
        answer, used_labels = self._append_asset_supporting_list(
            grounded.answer,
            grounded.used_citation_labels,
            allowed,
            search_plan=search_plan,
            language=search_plan.language,
        )
        references = tuple(
            self._reference_card(allowed[label])
            for label in used_labels
            if label in allowed
        )
        return self._result(context, grounded.model_copy(update={
            "answer": answer,
            "used_citation_labels": used_labels,
            "references": references,
        }))

    @classmethod
    def _exact_answer_source_fallback(
        cls,
        *,
        context,
        retrieval,
        search_plan,
        grounded,
    ):
        """单 Asset 详情生成失败时直接展示唯一的已验证正文证据。"""
        route = context.config_snapshot.get("route") or {}
        answer_basis = (
            route.get("answer_basis") if isinstance(route, dict) else None
        )
        citations = retrieval.citation_pack.citations
        if (
            grounded.status != "ANSWER_VALIDATION_FAILED"
            or search_plan.operation != "ANSWER"
            or answer_basis != "EXACT_METADATA_ANSWER"
            or len(citations) != 1
        ):
            return None
        citation = citations[0]
        excerpt = _CITATION_LABEL_PATTERN.sub(
            "", str(citation.excerpt or "")
        ).strip()
        if not excerpt:
            return None
        title = _CITATION_LABEL_PATTERN.sub(
            "", str(citation.bundle_title or citation.title or "")
        ).strip()
        answer = (
            f"**{title}**\n\n{excerpt} [{citation.citation_label}]"
            if title
            else f"{excerpt} [{citation.citation_label}]"
        )
        logger.warning(
            "KM Asset 单项详情改用已验证正文兜底 "
            "| run_id={} | task_id={} | citation_label={}",
            context.run_id,
            context.task_id,
            citation.citation_label,
        )
        return GroundedAnswer(
            answer=answer,
            status="READY",
            used_citation_labels=(citation.citation_label,),
            references=(cls._reference_card(citation),),
            warnings=grounded.warnings,
        )


class KmAssetDataQuerySkill(DataQuerySkill):
    """执行 KM Asset 托管语义模型问数。"""
