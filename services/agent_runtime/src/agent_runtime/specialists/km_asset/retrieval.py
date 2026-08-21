"""KM Asset 基于统一搜索计划执行 KC 正文取证。"""

import asyncio
import json
import re
from typing import Any

from loguru import logger
from platform_core.contracts import (
    AssetBooleanExpression,
    AssetSearchCriterion,
    AssetSearchPlanV1,
)

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.language import detect_unicode_language
from agent_runtime.runtime import ExecutionContext


_TOPIC_EXPANSION_LANGUAGES = frozenset({"zh-CN", "ja-JP", "ko-KR"})


class KmAssetRetrievalMixin:
    """只为 KM Asset Skill 提供搜索计划取证实现。"""

    @staticmethod
    def _asset_search_plan(
        context: ExecutionContext,
    ) -> AssetSearchPlanV1 | None:
        route = context.config_snapshot.get("route") or {}
        raw = route.get("asset_search_plan") if isinstance(route, dict) else None
        return AssetSearchPlanV1.model_validate(raw) if raw else None

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
        queries_by_key: dict[str, tuple[str, ...]] = {}
        expansion_warnings: list[str] = []
        for (key, criterion), (queries, query_warnings) in zip(
            criterion_requests, expansions, strict=True
        ):
            queries_by_key[key] = queries
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
        hit_sets = self._evidence_hit_sets(groups_by_key)
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
            asset = dict(candidate)
            asset.update(assets.get(bundle_id, {}))
            if not asset.get("title") and asset.get("display_title"):
                asset["title"] = asset["display_title"]
            statuses = {
                criterion_id: self._criterion_matches(
                    criterion,
                    asset=asset,
                    bundle_id=bundle_id,
                    hit_sets=hit_sets,
                    semantic_terms=queries_by_key.get(criterion_id, ()),
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
                    semantic_terms=queries_by_key.get(
                        preference.preference_id, ()
                    ),
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
                candidate["_asset_title"] = str(
                    asset.get("title")
                    or candidate.get("display_title")
                    or ""
                )
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

        identity_warnings: list[str] = []
        identity_targets = self._missing_identity_candidates(
            eligible,
            groups_by_key,
            limit=plan.result_assets.target_count,
        )

        async def retrieve_asset_identity(candidate):
            """用 Asset 标题从同 Bundle manifest 取得最低可引用证据。"""
            query = str(
                candidate.get("_asset_title") or plan.query_text
            ).strip()
            async with semaphore:
                response = await self._client.retrieve_evidence(
                    query=query,
                    candidates=self._evidence_candidates([candidate]),
                    domain_id=context.domain_id,
                    agent_id=str(context.agent_id),
                    auth_context=self._auth_context(context),
                    max_security_level=self._security_level(context),
                    max_evidence=1,
                    context_limit=0,
                    coverage_mode=coverage_mode,
                    run_id=context.run_id,
                    task_id=context.task_id,
                )
            bundle_id = str(candidate.get("bundle_id") or "")
            groups = [
                group
                for group in response.get("citations") or ()
                if str(group.get("bundle_id") or "") == bundle_id
                and group.get("items")
            ]
            return groups, list(response.get("warnings") or ())

        if identity_targets:
            identity_results = await asyncio.gather(*(
                retrieve_asset_identity(candidate)
                for candidate in identity_targets
            ))
            groups_by_key["__asset_identity__"] = [
                group
                for groups, _warnings in identity_results
                for group in groups
            ]
            identity_warnings.extend(
                warning
                for _groups, warnings in identity_results
                for warning in warnings
            )
        logger.info(
            "Asset 条件证据收敛完成 | run_id={} | task_id={} | "
            "candidate_count={} | evidence_bundle_count={} | "
            "evidence_hit_count={} | eligible_count={}",
            context.run_id,
            context.task_id,
            len(candidates),
            len({
                str(group.get("bundle_id") or "")
                for groups in groups_by_key.values()
                for group in groups
                if group.get("bundle_id")
            }),
            sum(
                len(group.get("items") or ())
                for groups in groups_by_key.values()
                for group in groups
            ),
            len(eligible),
        )

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
            candidate.pop("_asset_title", None)
        warnings = expansion_warnings + [
            warning
            for _, response, _ in retrieved
            for warning in response.get("warnings") or []
        ] + identity_warnings
        return {
            "citations": merged_groups,
            "warnings": warnings,
            "diagnostics": {
                "strategy": "ASSET_CRITERION_MATRIX.v1",
                "criteria_count": len(hard_content),
                "preference_count": len(preference_content),
                "eligible_count": len(eligible),
                "requirements": matrix,
            },
        }, eligible

    @staticmethod
    def _missing_identity_candidates(
        eligible: list[dict[str, Any]],
        groups_by_key: dict[str, list[dict[str, Any]]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """找出还没有正文 C 的展示候选，用 manifest 补齐 Asset 自身引用。"""
        covered = {
            str(group.get("bundle_id") or "")
            for groups in groups_by_key.values()
            for group in groups
            if any(
                (item.get("evidence") or {}).get("document_id")
                for item in group.get("items") or ()
                if isinstance(item, dict)
            )
        }
        return [
            candidate
            for candidate in eligible[:limit]
            if str(candidate.get("bundle_id") or "") not in covered
        ]

    @staticmethod
    def _evidence_hit_sets(
        groups_by_key: dict[str, list[dict[str, Any]]],
    ) -> dict[str, set[str]]:
        """KC 已选中的同 Bundle 引用就是语义条件证据，不再交给 LLM 否决。"""
        return {
            key: {
                str(group.get("bundle_id") or "")
                for group in groups
                if group.get("bundle_id") and group.get("items")
            }
            for key, groups in groups_by_key.items()
        }

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
        semantic_terms: tuple[str, ...] = (),
    ) -> bool:
        if criterion.kind not in {"METADATA", "IDENTIFIER"}:
            if bundle_id in hit_sets.get(criterion.criterion_id, set()):
                return True
            return cls._semantic_metadata_matches(
                criterion,
                asset=asset,
                semantic_terms=semantic_terms,
            )
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
        semantic_terms: tuple[str, ...] = (),
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
        if expected and all(value in searchable_text for value in expected):
            return True
        alternatives = [
            str(value).strip().casefold()
            for value in semantic_terms
            if str(value).strip()
        ]
        return any(
            value in searchable_text for value in alternatives
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
