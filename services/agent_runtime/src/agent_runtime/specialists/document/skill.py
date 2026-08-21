"""基于 Knowledge Core 两阶段检索的 Document Skill。"""

import base64
import asyncio
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID

from PIL import Image
from platform_core.contracts import (
    AssetBooleanExpression,
    AssetSearchCriterion,
    AssetSearchPlanV1,
    AuthContext,
    PrincipalKind,
)

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.language import detect_unicode_language
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult

from .contracts import (
    Citation,
    CitationPack,
    DocumentRetrievalResult,
    RetrievalCoverage,
)


_TOPIC_EXPANSION_LANGUAGES = frozenset({"zh-CN", "ja-JP", "ko-KR"})


class KnowledgeRetrievalSkill:
    """只调用 KC API，不访问 KC Entity、Repository 或向量表。"""

    def __init__(
        self,
        *,
        knowledge_core_client,
        service_name: str,
        model_client=None,
        prompt_resolver=None,
    ):
        self._client = knowledge_core_client
        self._service_name = service_name
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, context: ExecutionContext) -> SkillResult:
        query = self._standalone_query(context)
        collection_ids = await self._resolve_collection_ids(context)
        if not collection_ids:
            return self._empty_result(
                context,
                status="INSUFFICIENT_EVIDENCE",
                warning="当前 Agent 没有可用的 Collection 绑定",
            )

        retrieval_config = self._retrieval_config(context)
        coverage_mode = str(
            (context.config_snapshot.get("route") or {}).get(
                "coverage_mode", "BALANCED"
            )
        ).upper()
        image_payloads = await self._load_query_images(context)
        image_processing: dict[str, Any] = {
            "image_count": len(image_payloads),
            "visual_search": "NOT_REQUESTED",
            "vlm_text_search": "NOT_REQUESTED",
        }
        warnings: list[str] = []
        vlm_descriptions, vlm_prompt_ref = await self._describe_images(
            context, image_payloads, image_processing, warnings
        )
        retrieval_query = self._multimodal_query(
            query, vlm_descriptions
        )
        visual_outcome = await self._visual_hits(
            context,
            collection_ids,
            retrieval_config,
            image_payloads,
            image_processing,
            warnings,
        )
        visual_hits = list(visual_outcome.get("results") or [])
        self._finalize_image_warnings(
            image_processing=image_processing,
            warnings=warnings,
        )
        scoped_candidates = await self._scoped_candidates(
            context=context,
            allowed_collection_ids=collection_ids,
            warnings=warnings,
        )
        if scoped_candidates is None:
            discovery = await self._client.discover(
                query=retrieval_query,
                collection_ids=collection_ids,
                domain_id=context.domain_id,
                agent_id=str(context.agent_id),
                auth_context=self._auth_context(context),
                max_security_level=self._security_level(context),
                per_collection_limit=retrieval_config["max_bundles"],
                coverage_mode=coverage_mode,
                run_id=context.run_id,
                task_id=context.task_id,
            )
            warnings.extend(discovery.get("warnings") or [])
            discovery_diagnostics = dict(
                discovery.get("diagnostics") or {}
            )
            candidates = self._merge_candidates(
                visual_hits,
                list(discovery.get("candidates") or []),
                limit=retrieval_config["max_bundles"],
            )
        else:
            candidates = scoped_candidates
            discovery_diagnostics = {
                "strategy": "QUERY_RESULT_BUNDLE_SCOPE",
                "target_count": len(candidates),
            }
        if not candidates:
            return self._empty_result(
                context,
                status="INSUFFICIENT_EVIDENCE",
                warning="Knowledge Core 未发现相关 Bundle",
                warnings=tuple(warnings),
                query_plan={
                    "image_processing": image_processing,
                    "diagnostics": {
                        "discovery": discovery_diagnostics,
                    },
                },
            )

        asset_search_plan = self._asset_search_plan(context)
        if asset_search_plan is not None and scoped_candidates is not None:
            evidence, candidates = await self._retrieve_asset_plan_evidence(
                context=context,
                plan=asset_search_plan,
                candidates=candidates,
                retrieval_config=retrieval_config,
                coverage_mode=coverage_mode,
            )
        else:
            evidence = await self._client.retrieve_evidence(
                query=retrieval_query,
                candidates=self._evidence_candidates(candidates),
                domain_id=context.domain_id,
                agent_id=str(context.agent_id),
                auth_context=self._auth_context(context),
                max_security_level=self._security_level(context),
                max_evidence=retrieval_config["max_citations"],
                context_limit=retrieval_config["context_limit"],
                coverage_mode=coverage_mode,
                run_id=context.run_id,
                task_id=context.task_id,
            )
        warnings.extend(evidence.get("warnings") or [])
        evidence_diagnostics = dict(
            evidence.get("diagnostics") or {}
        )
        raw_citations = list(evidence.get("citations") or [])
        citations = self._map_citations(
            raw_citations,
            candidates=candidates,
            prefer_evidence_order=self._prefer_evidence_order(context),
        )
        status = "READY" if citations else "INSUFFICIENT_EVIDENCE"
        gaps = () if citations else ("未找到可引用的正文证据",)
        result = DocumentRetrievalResult(
            status=status,
            citation_pack=CitationPack(
                question=query,
                query_plan={
                    "strategy": "KC_TWO_STAGE",
                    "visual_query_count": len(
                        self._query_image_descriptors(context)
                    ),
                    "image_processing": image_processing,
                    "vlm_prompt": vlm_prompt_ref,
                    "target_level": "AUTO",
                    "collection_ids": [
                        str(value) for value in collection_ids
                    ],
                    "max_bundles": retrieval_config["max_bundles"],
                    "max_citations": retrieval_config["max_citations"],
                    "coverage_mode": coverage_mode,
                    "diagnostics": {
                        "discovery": discovery_diagnostics,
                        "evidence": evidence_diagnostics,
                    },
                },
                bundle_candidates=tuple(candidates),
                citations=tuple(citations),
                coverage=RetrievalCoverage(
                    candidate_bundle_count=len(candidates),
                    selected_document_count=len(
                        {item.document_id for item in citations}
                    ),
                    selected_evidence_count=sum(
                        len(item.evidence_ids) for item in citations
                    ),
                    uncovered_aspects=gaps,
                ),
            ),
            retrieval_report={
                "strategy_version": "kc-two-stage-v1",
                "discovery_candidate_count": len(candidates),
                "citation_count": len(citations),
                "selector": "bundle-evidence-aggregation-v1",
                "diagnostics": {
                    "discovery": discovery_diagnostics,
                    "evidence": evidence_diagnostics,
                },
                "visual_hit_count": len(visual_hits),
                "vlm_description_count": len(vlm_descriptions),
            },
            coverage_gaps=gaps,
            warnings=tuple(warnings),
        )
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="CITATION_PACK",
                schema_version="DocumentRetrievalResult.v1",
                payload=result.model_dump(mode="json"),
                provenance={
                    "knowledge_core_api": "internal/v1",
                    "strategy": "KC_TWO_STAGE",
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                },
                security_level=self._security_level(context),
            )
        )

    @staticmethod
    def _asset_search_plan(
        context: ExecutionContext,
    ) -> AssetSearchPlanV1 | None:
        route = context.config_snapshot.get("route") or {}
        raw = route.get("asset_search_plan") if isinstance(route, dict) else None
        return AssetSearchPlanV1.model_validate(raw) if raw else None

    @staticmethod
    def _evidence_candidates(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """只向 KC 发送 Evidence API 声明的候选字段。"""
        return [{
            "collection_id": item["collection_id"],
            "bundle_id": item["bundle_id"],
            "bundle_revision_id": item["bundle_revision_id"],
            "document_version_ids": (
                [item["document_version_id"]]
                if item.get("document_version_id") else []
            ),
        } for item in candidates]

    async def _retrieve_asset_plan_evidence(
        self,
        *,
        context: ExecutionContext,
        plan: AssetSearchPlanV1,
        candidates: list[dict[str, Any]],
        retrieval_config: dict[str, int],
        coverage_mode: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """逐个语义条件取证，再按布尔表达式生成条件命中矩阵。"""
        hard_content = [
            item for item in plan.criteria
            if item.kind not in {"METADATA", "IDENTIFIER"}
        ]
        preference_content = [
            item for item in plan.preferences
            if item.criterion.kind not in {"METADATA", "IDENTIFIER"}
        ]
        criterion_requests: list[tuple[str, AssetSearchCriterion | None]] = []
        for criterion in hard_content:
            criterion_requests.append((criterion.criterion_id, criterion))
        for preference in preference_content:
            criterion_requests.append((
                preference.preference_id, preference.criterion
            ))
        content_support_required = bool(
            not hard_content
            and plan.target == "CONTENT"
            and plan.operation in {"ANSWER", "COMPARE"}
        )
        if content_support_required:
            criterion_requests.append(("__content__", None))

        expansions = await asyncio.gather(*(
            self._criterion_queries(
                context=context,
                plan=plan,
                criterion=criterion,
            )
            if criterion is not None
            else self._content_queries(plan.query_text)
            for _, criterion in criterion_requests
        ))
        requests: list[tuple[str, AssetSearchCriterion | None, str]] = []
        expansion_warnings: list[str] = []
        for (key, criterion), (queries, query_warnings) in zip(
            criterion_requests, expansions, strict=True
        ):
            expansion_warnings.extend(query_warnings)
            requests.extend(
                (key, criterion, query) for query in queries
            )

        semaphore = asyncio.Semaphore(4)

        async def retrieve_one(key, criterion, query):
            async with semaphore:
                response = await self._client.retrieve_evidence(
                    query=query,
                    candidates=self._evidence_candidates(candidates),
                    domain_id=context.domain_id,
                    agent_id=str(context.agent_id),
                    auth_context=self._auth_context(context),
                    max_security_level=self._security_level(context),
                    max_evidence=retrieval_config["max_citations"],
                    context_limit=retrieval_config["context_limit"],
                    coverage_mode=coverage_mode,
                    run_id=context.run_id,
                    task_id=context.task_id,
                )
            groups = list(response.get("citations") or [])
            if criterion is not None and criterion.kind == "EXACT_PHRASE":
                groups = self._exact_phrase_groups(groups, criterion)
            return key, response, groups

        retrieved = await asyncio.gather(*(
            retrieve_one(*request) for request in requests
        ))
        groups_by_key = self._merge_groups_by_criterion(retrieved)
        hit_sets, support_judgments = await self._judge_asset_support(
            context=context,
            plan=plan,
            groups_by_key=groups_by_key,
        )
        ranks = {
            key: {
                str(group.get("bundle_id") or ""): rank
                for rank, group in enumerate(groups, start=1)
            }
            for key, groups in groups_by_key.items()
        }
        assets = {
            str(item.get("bundle_id") or ""): item
            for item in self._document_scope_assets(context)
        }
        criterion_by_id = {item.criterion_id: item for item in plan.criteria}
        eligible: list[dict[str, Any]] = []
        matrix: list[dict[str, Any]] = []
        candidate_order = {
            str(item.get("bundle_id") or ""): index
            for index, item in enumerate(candidates)
        }
        for candidate in candidates:
            bundle_id = str(candidate.get("bundle_id") or "")
            asset = assets.get(bundle_id, {})
            statuses = {
                criterion_id: self._criterion_matches(
                    criterion,
                    asset=asset,
                    bundle_id=bundle_id,
                    hit_sets=hit_sets,
                )
                for criterion_id, criterion in criterion_by_id.items()
            }
            is_eligible = self._expression_matches(
                plan.eligibility_expression, statuses
            )
            if content_support_required:
                is_eligible = is_eligible and bundle_id in hit_sets.get(
                    "__content__", set()
                )
            preference_hits = [
                preference.preference_id
                for preference in plan.preferences
                if self._criterion_matches(
                    preference.criterion,
                    asset=asset,
                    bundle_id=bundle_id,
                    hit_sets={
                        preference.criterion.criterion_id: hit_sets.get(
                            preference.preference_id, set()
                        )
                    },
                )
            ]
            matrix.append({
                "asset_id": asset.get("asset_id"),
                "bundle_id": bundle_id,
                "eligible": is_eligible,
                "requirements": statuses,
                "matched_preferences": preference_hits,
            })
            if is_eligible:
                candidate = dict(candidate)
                candidate["_preference_hits"] = len(preference_hits)
                hard_ranks = [
                    ranks.get(item.criterion_id, {}).get(bundle_id, 10**6)
                    for item in hard_content
                    if statuses.get(item.criterion_id)
                ]
                candidate["_weakest_hard_rank"] = max(hard_ranks, default=0)
                eligible.append(candidate)
        eligible.sort(key=lambda item: (
            -int(item.get("_preference_hits") or 0),
            int(item.get("_weakest_hard_rank") or 0),
            candidate_order.get(str(item.get("bundle_id") or ""), 10**6),
        ))

        merged_groups: list[dict[str, Any]] = []
        for candidate in eligible:
            bundle_id = str(candidate.get("bundle_id") or "")
            merged_items: list[dict[str, Any]] = []
            first_group: dict[str, Any] | None = None
            for key, groups in groups_by_key.items():
                for group in groups:
                    if str(group.get("bundle_id") or "") != bundle_id:
                        continue
                    if first_group is None:
                        first_group = dict(group)
                    merged_items.extend(group.get("items") or [])
            if first_group is not None and merged_items:
                first_group["items"] = merged_items
                merged_groups.append(first_group)
            candidate.pop("_preference_hits", None)
            candidate.pop("_weakest_hard_rank", None)
        warnings = expansion_warnings + [
            warning
            for _, response, _ in retrieved
            for warning in response.get("warnings") or []
        ]
        return {
            "citations": merged_groups,
            "warnings": warnings,
            "diagnostics": {
                "strategy": "ASSET_CRITERION_MATRIX.v1",
                "criteria_count": len(hard_content),
                "preference_count": len(preference_content),
                "eligible_count": len(eligible),
                "requirements": matrix,
                "support_judgments": support_judgments,
            },
        }, eligible

    async def _judge_asset_support(
        self,
        *,
        context: ExecutionContext,
        plan: AssetSearchPlanV1,
        groups_by_key: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
        """只让直接正文支持通过语义硬条件，缺失判定一律视为不支持。"""
        entries = []
        for key, groups in groups_by_key.items():
            for group in groups:
                excerpts = []
                for item in group.get("items") or []:
                    evidence = item.get("evidence") or {}
                    content = str(evidence.get("content_text") or "").strip()
                    if content:
                        excerpts.append(content[:1200])
                if excerpts:
                    entries.append({
                        "criterion_key": key,
                        "bundle_id": str(group.get("bundle_id") or ""),
                        "excerpts": excerpts,
                    })
        if not entries:
            return {key: set() for key in groups_by_key}, []
        model_name = str(agent_model_name(
            context.config_snapshot.get("agent", {}), "composer_llm"
        ) or "").strip()
        if not model_name or self._model_client is None or self._prompt_resolver is None:
            raise ValueError("Agent 未配置 Asset 条件支持判断模型")
        prompt = await self._prompt_resolver.resolve(
            "agent_runtime.km_asset_criterion_support"
        )
        criteria = {
            item.criterion_id: {
                "kind": item.kind,
                "values": list(item.values),
                "field_scope": list(item.field_scope),
            }
            for item in plan.criteria
            if item.kind not in {"METADATA", "IDENTIFIER"}
        }
        criteria.update({
            item.preference_id: {
                "kind": item.criterion.kind,
                "values": list(item.criterion.values),
                "field_scope": list(item.criterion.field_scope),
            }
            for item in plan.preferences
            if item.criterion.kind not in {"METADATA", "IDENTIFIER"}
        })
        criteria["__content__"] = {
            "kind": "QUESTION_SUPPORT",
            "values": [plan.query_text],
            "field_scope": ["CONTENT"],
        }
        known = {
            (item["criterion_key"], item["bundle_id"]) for item in entries
        }
        last_error = ""
        for attempt in range(2):
            response = await self._model_client.get_llm_json(
                served_model_name=model_name,
                prompt=[
                    {"role": "system", "content": prompt.content},
                    {"role": "user", "content": json.dumps({
                        "question": plan.query_text,
                        "criteria": criteria,
                        "evidence": entries,
                    }, ensure_ascii=False)},
                ],
            )
            try:
                judgments = response.get("judgments")
                if not isinstance(judgments, list):
                    raise ValueError("judgments 必须是数组")
                normalized = []
                seen = set()
                for item in judgments:
                    if not isinstance(item, dict):
                        raise ValueError("judgment 必须是对象")
                    pair = (
                        str(item.get("criterion_key") or ""),
                        str(item.get("bundle_id") or ""),
                    )
                    status = str(item.get("status") or "")
                    if pair not in known or pair in seen:
                        raise ValueError("judgment 引用了未知或重复的条件证据")
                    if status not in {
                        "DIRECT_SUPPORT", "PARTIAL_SUPPORT", "CONTEXT_ONLY",
                        "CONTRADICTS", "NO_SUPPORT",
                    }:
                        raise ValueError("judgment status 不符合支持判断协议")
                    seen.add(pair)
                    normalized.append({
                        "criterion_key": pair[0],
                        "bundle_id": pair[1],
                        "status": status,
                    })
                hit_sets = {key: set() for key in groups_by_key}
                for item in normalized:
                    if item["status"] == "DIRECT_SUPPORT":
                        hit_sets[item["criterion_key"]].add(item["bundle_id"])
                return hit_sets, normalized
            except (AttributeError, TypeError, ValueError) as exc:
                last_error = str(exc)
                if attempt == 0:
                    continue
        raise ValueError(f"Asset 条件支持判断输出不符合协议：{last_error}")

    @staticmethod
    def _document_scope_assets(context: ExecutionContext) -> list[dict[str, Any]]:
        for artifact in reversed(context.input_artifacts):
            if artifact.artifact_type == "DOCUMENT_SCOPE":
                return [
                    dict(item) for item in (artifact.payload or {}).get("assets") or []
                    if isinstance(item, dict)
                ]
        return []

    @staticmethod
    def _criterion_query(criterion: AssetSearchCriterion) -> str:
        return " ".join(str(value) for value in criterion.values).strip()

    @staticmethod
    async def _content_queries(
        query: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """把普通正文问题包装为与条件扩展一致的异步结果。"""
        return (query,), ()

    async def _criterion_queries(
        self,
        *,
        context: ExecutionContext,
        plan: AssetSearchPlanV1,
        criterion: AssetSearchCriterion,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """为中日韩语义条件补充英文检索词，原文与译词保持独立查询。"""
        original = self._criterion_query(criterion)
        if not original:
            return (), ()
        equivalents = (
            tuple(criterion.resolved_concept.equivalents)
            if criterion.resolved_concept is not None
            else ()
        )
        if equivalents:
            return tuple(dict.fromkeys((original, *equivalents))), ()
        if criterion.kind != "SEMANTIC_CONCEPT" or len(criterion.values) != 1:
            return (original,), ()
        source_language = detect_unicode_language(original)
        if source_language not in _TOPIC_EXPANSION_LANGUAGES:
            return (original,), ()
        agent = context.config_snapshot.get("agent", {})
        model_name = str(
            agent_model_name(agent, "data_planner_llm")
            or agent_model_name(agent, "composer_llm")
            or ""
        ).strip()
        if (
            not model_name
            or self._model_client is None
            or self._prompt_resolver is None
        ):
            return (original,), (
                "主题英文补充检索不可用，已仅使用原语言检索",
            )
        prompt = await self._prompt_resolver.resolve(
            "agent_runtime.km_topic_english_expand"
        )
        messages = [
            {"role": "system", "content": prompt.content},
            {"role": "user", "content": json.dumps({
                "source_language": source_language,
                "question": plan.query_text,
                "original_topic": original,
            }, ensure_ascii=False)},
        ]
        for attempt in range(2):
            try:
                response = await self._model_client.get_llm_json(
                    served_model_name=model_name,
                    prompt=messages,
                )
                english_topics = self._validate_topic_expansion(
                    response,
                    source_language=source_language,
                    original_topic=original,
                )
                return tuple(dict.fromkeys((original, *english_topics))), ()
            except (AttributeError, TypeError, ValueError) as exc:
                if attempt == 0:
                    messages.append({
                        "role": "system",
                        "content": (
                            "上一个结果不符合 KMTopicExpansion.v1，"
                            f"请只输出修正后的 JSON。错误：{exc}"
                        ),
                    })
        return (original,), (
            "主题英文补充检索失败，已仅使用原语言检索",
        )

    @staticmethod
    def _validate_topic_expansion(
        response: Any,
        *,
        source_language: str,
        original_topic: str,
    ) -> tuple[str, ...]:
        """校验主题扩展回显和英文词数量，拒绝模型擅自改写主题。"""
        if not isinstance(response, dict):
            raise TypeError("主题扩展输出必须是 JSON Object")
        if str(response.get("source_language") or "") != source_language:
            raise ValueError("source_language 与输入不一致")
        if str(response.get("original_topic") or "").strip() != original_topic:
            raise ValueError("original_topic 不得改写")
        raw_topics = response.get("english_topics")
        if not isinstance(raw_topics, list) or not 2 <= len(raw_topics) <= 3:
            raise ValueError("english_topics 必须包含 2 到 3 项")
        topics = tuple(str(item).strip() for item in raw_topics)
        if any(
            not value or re.search(r"[A-Za-z]", value) is None
            for value in topics
        ):
            raise ValueError("english_topics 每项必须包含英文字符")
        if len({value.casefold() for value in topics}) != len(topics):
            raise ValueError("english_topics 不能包含重复词")
        return topics

    @staticmethod
    def _merge_groups_by_criterion(
        retrieved: list[tuple[str, dict[str, Any], list[dict[str, Any]]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """按条件和 Bundle 合并多语言证据，避免同一条件重复判定。"""
        merged: dict[str, dict[str, dict[str, Any]]] = {}
        seen_items: dict[tuple[str, str], set[str]] = {}
        for key, _, groups in retrieved:
            by_bundle = merged.setdefault(key, {})
            for group in groups:
                bundle_id = str(group.get("bundle_id") or "")
                if not bundle_id:
                    continue
                if bundle_id not in by_bundle:
                    target = dict(group)
                    target["items"] = []
                    by_bundle[bundle_id] = target
                else:
                    target = by_bundle[bundle_id]
                item_keys = seen_items.setdefault((key, bundle_id), set())
                for item in group.get("items") or []:
                    item_key = json.dumps(
                        item, ensure_ascii=False, sort_keys=True, default=str
                    )
                    if item_key in item_keys:
                        continue
                    item_keys.add(item_key)
                    target["items"].append(item)
        return {
            key: list(groups.values())
            for key, groups in merged.items()
        }

    @staticmethod
    def _exact_phrase_groups(
        groups: list[dict[str, Any]], criterion: AssetSearchCriterion
    ) -> list[dict[str, Any]]:
        phrases = [str(value).casefold() for value in criterion.values]
        result = []
        for group in groups:
            items = [
                item for item in group.get("items") or []
                if all(
                    phrase in str(
                        (item.get("evidence") or {}).get("content_text") or ""
                    ).casefold()
                    for phrase in phrases
                )
            ]
            if items:
                copy = dict(group)
                copy["items"] = items
                result.append(copy)
        return result

    @classmethod
    def _criterion_matches(
        cls,
        criterion: AssetSearchCriterion,
        *,
        asset: dict[str, Any],
        bundle_id: str,
        hit_sets: dict[str, set[str]],
    ) -> bool:
        if criterion.kind not in {"METADATA", "IDENTIFIER"}:
            if bundle_id in hit_sets.get(criterion.criterion_id, set()):
                return True
            return cls._semantic_metadata_matches(criterion, asset=asset)
        values = [asset.get(field.casefold()) for field in criterion.field_scope]
        expected = list(criterion.values)
        value = values[0] if values else None
        operator = criterion.operator
        if operator == "IS_NULL":
            return value is None
        if operator == "IS_NOT_NULL":
            return value is not None
        normalized = str(value or "").strip().casefold()
        expected_text = [str(item).strip().casefold() for item in expected]
        if operator == "EQ":
            return normalized == expected_text[0]
        if operator == "NE":
            return normalized != expected_text[0]
        if operator == "IN":
            return normalized in expected_text
        if operator == "NOT_IN":
            return normalized not in expected_text
        if operator == "CONTAINS":
            return all(item in normalized for item in expected_text)
        if operator == "STARTS_WITH":
            return normalized.startswith(expected_text[0])
        if operator == "BETWEEN":
            return expected_text[0] <= normalized <= expected_text[1]
        comparisons = {
            "GT": normalized > expected_text[0],
            "GTE": normalized >= expected_text[0],
            "LT": normalized < expected_text[0],
            "LTE": normalized <= expected_text[0],
        }
        return comparisons.get(operator, False)

    @staticmethod
    def _semantic_metadata_matches(
        criterion: AssetSearchCriterion,
        *,
        asset: dict[str, Any],
    ) -> bool:
        """仅用合同声明的可搜索元数据字段提供确定性直接支持。"""
        if criterion.evidence_requirement != "METADATA_OR_CONTENT":
            return False
        searchable_fields = [
            field.casefold()
            for field in criterion.field_scope
            if field.upper() in {"TITLE", "PRODUCT", "SOLUTION"}
        ]
        if not searchable_fields:
            return False
        searchable_text = "\n".join(
            str(asset.get(field) or "").strip().casefold()
            for field in searchable_fields
        )
        expected = [
            str(value).strip().casefold()
            for value in criterion.values
            if str(value).strip()
        ]
        return bool(expected) and all(
            value in searchable_text for value in expected
        )

    @classmethod
    def _expression_matches(
        cls,
        expression: AssetBooleanExpression | None,
        statuses: dict[str, bool],
    ) -> bool:
        if expression is None:
            return True
        if expression.node_type == "REF":
            return statuses.get(str(expression.criterion_id), False)
        if expression.node_type == "NOT":
            return not cls._expression_matches(expression.child, statuses)
        values = [
            cls._expression_matches(item, statuses)
            for item in expression.children
        ]
        return all(values) if expression.node_type == "ALL" else any(values)

    async def _scoped_candidates(
        self,
        *,
        context: ExecutionContext,
        allowed_collection_ids: tuple[UUID, ...],
        warnings: list[str],
    ) -> list[dict[str, Any]] | None:
        """把问数选定的 Bundle 转成 KC Evidence 的确定性候选。"""
        scope = next(
            (
                item.payload or {}
                for item in reversed(context.input_artifacts)
                if item.artifact_type == "DOCUMENT_SCOPE"
            ),
            None,
        )
        if not isinstance(scope, dict) or "bundle_targets" not in scope:
            return None
        allowed = {str(value) for value in allowed_collection_ids}
        candidates: list[dict[str, Any]] = []
        retrieval_config = self._retrieval_config(context)
        raw_targets = list(scope.get("bundle_targets") or [])
        scoped_targets = raw_targets[:retrieval_config["candidate_scope_limit"]]
        semaphore = asyncio.Semaphore(16)

        async def resolve_target(target):
            if not isinstance(target, dict):
                return None, None
            try:
                bundle_id = UUID(str(target["bundle_id"]))
                revision_id = UUID(str(target["bundle_revision_id"]))
                async with semaphore:
                    status = await self._client.get_bundle_status(
                        domain_id=context.domain_id,
                        bundle_id=bundle_id,
                        auth_context=self._auth_context(context),
                    )
                availability = str(
                    status.get("availability_status") or ""
                ).upper()
                current_revision_id = str(
                    status.get("current_revision_id") or ""
                )
                if availability not in {"READY", "PARTIAL"}:
                    return None, "问数命中的 Asset 当前 Bundle 不可检索"
                if current_revision_id != str(revision_id):
                    return None, "问数命中的 Asset 已切换到其他 Bundle Revision"
                collection_id = str(status.get("collection_id") or "")
                if collection_id not in allowed:
                    return None, "问数命中的 Asset 不属于当前 Agent Collection"
                return {
                    "collection_id": collection_id,
                    "bundle_id": str(bundle_id),
                    "bundle_revision_id": str(revision_id),
                    "document_version_ids": [],
                    "display_title": str(target.get("title") or ""),
                }, None
            except (KeyError, TypeError, ValueError):
                return None, "问数命中的 Asset 缺少有效 Bundle 定位信息"
            except Exception:
                return None, "部分问数命中的 Asset 无法解析对应 Bundle"

        resolved = await asyncio.gather(*(
            resolve_target(target) for target in scoped_targets
        ))
        for candidate, warning in resolved:
            if candidate is not None:
                candidates.append(candidate)
            if warning:
                warnings.append(warning)
        return candidates

    async def _visual_hits(
        self,
        context: ExecutionContext,
        collection_ids: tuple[UUID, ...],
        retrieval_config: dict[str, int | bool],
        image_payloads: list[bytes],
        image_processing: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        if not image_payloads:
            return {"results": []}
        try:
            response = await self._client.search_visual(
                images_base64=[
                    base64.b64encode(payload).decode("ascii")
                    for payload in image_payloads
                ],
                collection_ids=collection_ids,
                domain_id=context.domain_id,
                agent_id=context.agent_id,
                auth_context=self._auth_context(context),
                per_image_limit=retrieval_config["max_bundles"],
                result_limit=retrieval_config["max_bundles"],
            )
        except Exception:
            image_processing["visual_search"] = "FAILED"
            warnings.append("图片相似检索暂时不可用，已继续其他检索路径")
            return {"results": []}
        searched = list(response.get("searched_collection_ids") or [])
        skipped = list(response.get("skipped_collection_ids") or [])
        if searched and skipped:
            image_processing["visual_search"] = "PARTIAL"
            warnings.append(
                "部分 Collection 未配置 Visual Embedding，已跳过其图搜图路径"
            )
        elif searched:
            image_processing["visual_search"] = "EXECUTED"
        else:
            image_processing["visual_search"] = "SKIPPED_NOT_CONFIGURED"
        image_processing["visual_searched_collection_ids"] = searched
        image_processing["visual_skipped_collection_ids"] = skipped
        return response

    async def _load_query_images(
        self, context: ExecutionContext
    ) -> list[bytes]:
        payloads: list[bytes] = []
        for descriptor in self._query_image_descriptors(context):
            uri = str(descriptor.get("storage_uri") or "")
            if uri:
                payloads.append(
                    await asyncio.to_thread(Path(uri).read_bytes)
                )
        return payloads

    async def _describe_images(
        self,
        context: ExecutionContext,
        image_payloads: list[bytes],
        image_processing: dict[str, Any],
        warnings: list[str],
    ) -> tuple[list[str], dict[str, Any] | None]:
        if not image_payloads:
            return [], None
        model_name = str(
            agent_model_name(
                context.config_snapshot.get("agent", {}), "query_vlm"
            )
            or ""
        ).strip()
        if not model_name:
            image_processing["vlm_text_search"] = (
                "SKIPPED_NOT_CONFIGURED"
            )
            return [], None
        if self._model_client is None or self._prompt_resolver is None:
            image_processing["vlm_text_search"] = "FAILED"
            warnings.append("图片理解服务未初始化，已跳过图片转文字检索")
            return [], None
        try:
            prompt = await self._prompt_resolver.resolve(
                "agent_runtime.query_image_description"
            )
            descriptions = await asyncio.gather(
                *(
                    self._model_client.get_vlm_answer(
                        model_name,
                        Image.open(BytesIO(payload)).convert("RGB"),
                        prompt=prompt.content,
                        temperature=0.1,
                        max_tokens=1024,
                    )
                    for payload in image_payloads
                )
            )
        except Exception:
            image_processing["vlm_text_search"] = "FAILED"
            warnings.append("图片理解暂时不可用，已继续其他检索路径")
            return [], (
                prompt.ref() if "prompt" in locals() else None
            )
        normalized = [
            str(value).strip()[:3000]
            for value in descriptions
            if str(value).strip()
        ]
        image_processing["vlm_text_search"] = (
            "EXECUTED" if normalized else "FAILED"
        )
        if not normalized:
            warnings.append("图片理解未产生可检索文本")
        return normalized, prompt.ref()

    @staticmethod
    def _multimodal_query(
        query: str, descriptions: list[str]
    ) -> str:
        if not descriptions:
            return query
        supplement = "\n".join(
            f"查询图片{i + 1}：{value}"
            for i, value in enumerate(descriptions)
        )
        return f"{query}\n\n{supplement}"[:8000]

    @staticmethod
    def _finalize_image_warnings(
        *,
        image_processing: dict[str, Any],
        warnings: list[str],
    ) -> None:
        if not image_processing.get("image_count"):
            return
        visual = image_processing["visual_search"]
        vlm = image_processing["vlm_text_search"]
        if (
            visual == "SKIPPED_NOT_CONFIGURED"
            and vlm == "SKIPPED_NOT_CONFIGURED"
        ):
            warnings.append(
                "未配置 Visual Embedding 或查询 VLM，已忽略上传图片并仅处理文字"
            )
        elif visual == "SKIPPED_NOT_CONFIGURED":
            warnings.append(
                "未配置 Visual Embedding，已仅使用 VLM 图片转文字检索"
            )
        elif vlm == "SKIPPED_NOT_CONFIGURED":
            warnings.append(
                "未配置查询 VLM，已仅执行图片相似检索"
            )

    @staticmethod
    def _query_image_descriptors(
        context: ExecutionContext,
    ) -> list[dict[str, Any]]:
        return list(
            context.config_snapshot.get("client_metadata", {}).get(
                "query_images", []
            )
            or []
        )

    @staticmethod
    def _merge_candidates(
        visual_hits: list[dict[str, Any]],
        text_candidates: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """视觉候选优先保留，再补充文本 Discovery 候选。"""
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*visual_hits, *text_candidates]:
            key = str(item.get("bundle_revision_id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(item)
            if len(output) >= limit:
                break
        return output

    async def _resolve_collection_ids(
        self, context: ExecutionContext
    ) -> tuple[UUID, ...]:
        configured = (
            context.config_snapshot.get("retrieval", {})
            .get("collection_ids", [])
        )
        if configured:
            return tuple(UUID(str(value)) for value in configured)
        response = await self._client.list_agent_bindings(
            domain_id=context.domain_id,
            agent_id=context.agent_id,
            auth_context=self._auth_context(context),
        )
        return tuple(
            UUID(str(item["collection_id"]))
            for item in response.get("bindings", [])
            if item.get("status") == "ACTIVE"
        )

    @staticmethod
    def _standalone_query(context: ExecutionContext) -> str:
        document_scopes = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "DOCUMENT_SCOPE"
        ]
        if document_scopes:
            scoped_query = str(
                (document_scopes[-1].payload or {}).get("query") or ""
            ).strip()
            if scoped_query:
                return scoped_query
        artifacts = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "CONTEXT_REWRITE"
        ]
        if not artifacts:
            return context.original_input
        query = str(
            (artifacts[-1].payload or {}).get("standalone_query") or ""
        ).strip()
        return query or context.original_input

    def _auth_context(self, context: ExecutionContext) -> AuthContext:
        return AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id=self._service_name,
            calling_service=self._service_name,
            request_id=context.request_id,
            trace_id=context.trace_id,
            domain_id=str(context.domain_id),
            asserted_user_id=context.actor_id,
        )

    @staticmethod
    def _security_level(context: ExecutionContext) -> int:
        value = int(
            context.config_snapshot.get("retrieval", {})
            .get("security_level", 0)
        )
        return max(0, min(value, 3))

    @staticmethod
    def _retrieval_config(
        context: ExecutionContext,
    ) -> dict[str, int]:
        agent_snapshot = context.config_snapshot.get("agent", {})
        agent_config = agent_snapshot.get("config", {})
        retrieval = agent_config.get("retrieval", {})
        return {
            "max_bundles": max(
                1, min(int(retrieval.get("max_bundles", 10)), 50)
            ),
            "max_citations": max(
                1, min(int(retrieval.get("max_citations", 12)), 50)
            ),
            "context_limit": max(
                0, min(int(retrieval.get("context_limit", 4)), 20)
            ),
            "candidate_scope_limit": max(
                1,
                min(int(retrieval.get("candidate_scope_limit", 1000)), 1000),
            ),
        }

    @staticmethod
    def _map_citations(
        raw_citations: list[dict[str, Any]],
        *,
        candidates: list[dict[str, Any]],
        prefer_evidence_order: bool = False,
    ) -> list[Citation]:
        titles = {
            str(item["bundle_id"]): str(item.get("display_title") or "")
            for item in candidates
        }
        groups_by_bundle: dict[str, list[dict[str, Any]]] = {}
        for group in raw_citations:
            groups_by_bundle.setdefault(str(group["bundle_id"]), []).append(
                group
            )
        candidate_order = [str(item["bundle_id"]) for item in candidates]
        ordered_bundle_ids = list(dict.fromkeys(
            (*groups_by_bundle.keys(), *candidate_order)
            if prefer_evidence_order
            else (*candidate_order, *groups_by_bundle.keys())
        ))
        result: list[Citation] = []
        for bundle_key in ordered_bundle_ids:
            groups = groups_by_bundle.get(bundle_key, [])
            if not groups:
                continue
            selected: list[dict[str, Any]] = []
            for group in groups:
                items = list(group.get("items") or [])
                primary = [
                    item
                    for item in items
                    if item.get("final_role") == "PRIMARY"
                ]
                selected.extend(primary or items)
            if not selected:
                continue
            first_group = groups[0]
            first = selected[0].get("evidence") or {}
            if not first.get("document_id"):
                continue
            excerpt_parts: list[str] = []
            evidence_ids: list[UUID] = []
            for item in selected:
                evidence = item.get("evidence") or {}
                content = str(evidence.get("content_text") or "").strip()
                if content:
                    document_name = str(
                        evidence.get("document_name")
                        or evidence.get("external_document_id")
                        or "Bundle 正文"
                    )
                    heading = " > ".join(evidence.get("heading_path") or [])
                    location = f" · {heading}" if heading else ""
                    excerpt_parts.append(
                        f"文档：{document_name}{location}\n{content}"
                    )
                evidence_id = evidence.get("evidence_id")
                if evidence_id:
                    evidence_ids.append(UUID(str(evidence_id)))
            excerpt = "\n\n".join(dict.fromkeys(excerpt_parts))[:4000]
            provenance = first.get("provenance") or {}
            bundle_id = UUID(bundle_key)
            bundle_title = (
                str(first.get("bundle_title") or "").strip()
                or titles.get(str(bundle_id))
                or "未命名 Bundle"
            )
            result.append(
                Citation(
                    citation_label=f"C{len(result) + 1}",
                    collection_id=UUID(str(first_group["collection_id"])),
                    bundle_id=bundle_id,
                    bundle_revision_id=UUID(
                        str(first_group["bundle_revision_id"])
                    ),
                    document_id=UUID(str(first["document_id"])),
                    document_version_id=UUID(
                        str(first_group["document_version_id"])
                    ),
                    evidence_ids=tuple(dict.fromkeys(evidence_ids)),
                    title=bundle_title,
                    bundle_title=bundle_title,
                    external_document_id=first.get(
                        "external_document_id"
                    ),
                    document_role=first.get("document_role"),
                    excerpt=excerpt,
                    locator=dict(first.get("locator") or {}),
                    locator_schema_version=str(
                        first["locator_schema_version"]
                    ),
                    heading_path=tuple(first.get("heading_path") or ()),
                    relevance_reason="混合检索候选 Bundle 的正文证据",
                    source_hash=(
                        first.get("content_hash")
                        or provenance.get("source_hash")
                        or provenance.get("content_hash")
                    ),
                )
            )
        return result

    @staticmethod
    def _prefer_evidence_order(context: ExecutionContext) -> bool:
        route = context.config_snapshot.get("route") or {}
        plan = route.get("asset_search_plan") if isinstance(route, dict) else None
        if not isinstance(plan, dict):
            return False
        unsupported = set(plan.get("unsupported_requests") or ())
        return "SEMANTIC_TOTAL_COUNT" not in unsupported

    @staticmethod
    def _empty_result(
        context: ExecutionContext,
        *,
        status: str,
        warning: str,
        warnings: tuple[str, ...] = (),
        query_plan: dict[str, Any] | None = None,
    ) -> SkillResult:
        result = DocumentRetrievalResult(
            status=status,
            citation_pack=CitationPack(
                question=context.original_input,
                query_plan={
                    "strategy": "KC_TWO_STAGE",
                    **(query_plan or {}),
                },
                bundle_candidates=(),
                citations=(),
                coverage=RetrievalCoverage(
                    candidate_bundle_count=0,
                    selected_document_count=0,
                    selected_evidence_count=0,
                    uncovered_aspects=(warning,),
                ),
            ),
            retrieval_report={
                "strategy_version": "kc-two-stage-v1",
                "discovery_candidate_count": 0,
                "citation_count": 0,
            },
            coverage_gaps=(warning,),
            warnings=tuple(dict.fromkeys((*warnings, warning))),
        )
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="CITATION_PACK",
                schema_version="DocumentRetrievalResult.v1",
                payload=result.model_dump(mode="json"),
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                },
                security_level=(
                    KnowledgeRetrievalSkill._security_level(context)
                ),
            ),
            warnings=tuple(dict.fromkeys((*warnings, warning))),
        )
