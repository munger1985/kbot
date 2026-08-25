"""KM Asset 基于统一搜索计划执行 KC 正文取证。"""

import asyncio
import json
import re
from time import perf_counter
from typing import Any
from uuid import UUID

from loguru import logger
from platform_core.contracts import (
    AssetBooleanExpression,
    AssetSearchCriterion,
    AssetSearchPlanV1,
    KNOWLEDGE_EVIDENCE_CANDIDATE_LIMIT,
)

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.language import detect_unicode_language, response_language
from agent_runtime.runtime import ExecutionContext


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
            combined_query = self._combined_retrieval_query(queries)
            if combined_query:
                requests.append((key, criterion, combined_query))

        semaphore = asyncio.Semaphore(4)

        target_count = max(1, plan.result_assets.target_count)
        qualification_limit = min(
            len(candidates),
            max(retrieval_config["max_citations"], target_count * 3),
        )
        retrieval_started_at = perf_counter()

        async def retrieve_one(key, criterion, query):
            async with semaphore:
                response = await self._client.retrieve_evidence(
                    query=query,
                    candidates=self._evidence_candidates(candidates),
                    domain_id=context.domain_id,
                    agent_id=str(context.agent_id),
                    auth_context=self._auth_context(context),
                    max_security_level=self._security_level(context),
                    max_evidence=qualification_limit,
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
        logger.info(
            "Asset 条件取证请求完成 | run_id={} | task_id={} | "
            "criterion_count={} | request_count={} | candidate_count={} | "
            "qualification_limit={} | duration_ms={:.2f}",
            context.run_id,
            context.task_id,
            len(criterion_requests),
            len(requests),
            len(candidates),
            qualification_limit,
            (perf_counter() - retrieval_started_at) * 1000,
        )
        groups_by_key = self._merge_groups_by_criterion(retrieved)
        criteria_by_key = {
            item.criterion_id: item for item in hard_content
        }
        criteria_by_key.update({
            item.preference_id: item.criterion
            for item in preference_content
        })
        hit_sets = self._qualified_evidence_hit_sets(
            groups_by_key,
            criteria_by_key=criteria_by_key,
            queries_by_key=queries_by_key,
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
            "evidence_hit_count={} | qualified_bundle_count={} | "
            "eligible_count={}",
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
            len({
                bundle_id
                for bundle_ids in hit_sets.values()
                for bundle_id in bundle_ids
            }),
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

    async def _prepare_scoped_targets(
        self,
        *,
        context: ExecutionContext,
        allowed_collection_ids: tuple[UUID, ...],
        raw_targets: list[dict[str, Any]],
        retrieval_config: dict[str, int],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """在解析 Bundle 状态前，把大范围问数资格集收敛为语义候选。"""
        scoped_targets = raw_targets[:retrieval_config["candidate_scope_limit"]]
        plan = self._asset_search_plan(context)
        if (
            plan is None
            or len(scoped_targets) <= KNOWLEDGE_EVIDENCE_CANDIDATE_LIMIT
        ):
            return scoped_targets

        hard_content = [
            item for item in plan.criteria
            if item.kind not in {"METADATA", "IDENTIFIER"}
        ]
        preference_content = [
            item for item in plan.preferences
            if item.criterion.kind not in {"METADATA", "IDENTIFIER"}
        ]
        semantic_criteria: list[AssetSearchCriterion | None] = [
            *hard_content,
            *(item.criterion for item in preference_content),
        ]
        content_support_required = bool(
            not hard_content
            and plan.target == "CONTENT"
            and plan.operation in {"ANSWER", "COMPARE"}
        )
        if content_support_required:
            semantic_criteria.append(None)
        if not semantic_criteria:
            return scoped_targets

        queries: list[str] = []
        for criterion in semantic_criteria:
            if criterion is None:
                topics = (plan.query_text,)
            else:
                original = self._criterion_query(criterion)
                equivalents = (
                    tuple(criterion.resolved_concept.equivalents)
                    if criterion.resolved_concept is not None
                    else ()
                )
                topics = self._unique_queries((original, *equivalents))
            query = self._combined_retrieval_query(topics)
            if query:
                queries.append(query)
        if not queries:
            return scoped_targets

        shortlist_limit = min(
            KNOWLEDGE_EVIDENCE_CANDIDATE_LIMIT,
            max(
                plan.result_assets.target_count * 4,
                int(retrieval_config.get("max_bundles", 10)) * 2,
                int(retrieval_config.get("max_citations", 12)) * 2,
            ),
        )
        revision_ids: list[UUID] = []
        seen_revision_ids: set[UUID] = set()
        for item in scoped_targets:
            try:
                revision_id = UUID(str(item["bundle_revision_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            if revision_id not in seen_revision_ids:
                seen_revision_ids.add(revision_id)
                revision_ids.append(revision_id)
        if not revision_ids:
            return []
        semaphore = asyncio.Semaphore(4)

        async def discover(query):
            async with semaphore:
                return await self._client.discover(
                    query=query,
                    collection_ids=allowed_collection_ids,
                    bundle_revision_ids=tuple(revision_ids),
                    domain_id=context.domain_id,
                    agent_id=str(context.agent_id),
                    auth_context=self._auth_context(context),
                    max_security_level=self._security_level(context),
                    per_collection_limit=shortlist_limit,
                    coverage_mode=str(
                        (context.config_snapshot.get("route") or {}).get(
                            "coverage_mode", "BALANCED"
                        )
                    ).upper(),
                    run_id=context.run_id,
                    task_id=context.task_id,
                )

        discovered = await asyncio.gather(*(discover(query) for query in queries))
        by_revision = {
            str(item.get("bundle_revision_id") or ""): item
            for item in scoped_targets
        }
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for response in discovered:
            response_candidates = list(response.get("candidates") or ())
            warnings.extend(response.get("warnings") or ())
            for item in response_candidates:
                revision_id = str(item.get("bundle_revision_id") or "")
                candidate = by_revision.get(revision_id)
                if candidate is None or revision_id in seen:
                    continue
                seen.add(revision_id)
                selected.append(candidate)
                if len(selected) >= shortlist_limit:
                    break
            if len(selected) >= shortlist_limit:
                break
        required = bool(hard_content or content_support_required)
        if not required and len(selected) < shortlist_limit:
            for candidate in scoped_targets:
                revision_id = str(candidate.get("bundle_revision_id") or "")
                if revision_id in seen:
                    continue
                seen.add(revision_id)
                selected.append(candidate)
                if len(selected) >= shortlist_limit:
                    break
        logger.info(
            "Asset 语义候选已收敛 | run_id={} | task_id={} | "
            "scope_count={} | candidate_count={} | candidate_limit={} | "
            "request_count={}",
            context.run_id,
            context.task_id,
            len(scoped_targets),
            len(selected),
            shortlist_limit,
            len(queries),
        )
        return selected

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
    def _qualified_evidence_hit_sets(
        groups_by_key: dict[str, list[dict[str, Any]]],
        *,
        criteria_by_key: dict[str, AssetSearchCriterion],
        queries_by_key: dict[str, tuple[str, ...]],
    ) -> dict[str, set[str]]:
        """只把明确包含条件主题的同 Bundle KC 证据视为资格支持。"""
        result: dict[str, set[str]] = {}
        for key, groups in groups_by_key.items():
            criterion = criteria_by_key.get(key)
            if criterion is None:
                result[key] = {
                    str(group.get("bundle_id") or "")
                    for group in groups
                    if group.get("bundle_id") and group.get("items")
                }
                continue
            terms = tuple(dict.fromkeys((
                *(
                    str(value).strip()
                    for value in criterion.values
                    if str(value).strip()
                ),
                *queries_by_key.get(key, ()),
            )))
            result[key] = {
                str(group.get("bundle_id") or "")
                for group in groups
                if group.get("bundle_id")
                and KmAssetRetrievalMixin._group_supports_terms(
                    group,
                    terms=terms,
                )
            }
        return result

    @classmethod
    def _group_supports_terms(
        cls, group: dict[str, Any], *, terms: tuple[str, ...]
    ) -> bool:
        """在 KC 返回的 Asset 正文中确定性核对原主题或扩展主题。"""
        searchable = "\n".join(
            str(value or "")
            for item in group.get("items") or ()
            if isinstance(item, dict)
            for evidence in (item.get("evidence") or {},)
            for value in (
                evidence.get("content_text"),
                evidence.get("retrieval_text"),
                evidence.get("bundle_title"),
                evidence.get("document_name"),
            )
        ).casefold()
        if not searchable:
            return False
        return any(
            cls._text_supports_semantic_term(searchable, term=term)
            for term in terms
        )

    @classmethod
    def _text_supports_semantic_term(
        cls, searchable: str, *, term: str
    ) -> bool:
        """核对完整短语，或核对同一 Asset 内全部拉丁概念词。"""
        normalized = " ".join(str(term).casefold().split())
        if not normalized:
            return False
        if not re.search(r"[a-z0-9]", normalized):
            return normalized in searchable

        phrase_pattern = r"(?<![a-z0-9])" + re.escape(normalized)
        phrase_pattern += r"(?![a-z0-9])"
        if re.search(phrase_pattern, searchable):
            return True

        concept_tokens = re.findall(r"[a-z0-9]+", normalized)
        if len(concept_tokens) < 2:
            return False
        searchable_tokens = re.findall(r"[a-z0-9]+", searchable)
        searchable_stems = {
            cls._semantic_token_stem(token) for token in searchable_tokens
        }
        concept_stems = {
            cls._semantic_token_stem(token) for token in concept_tokens
        }
        return concept_stems.issubset(searchable_stems)

    @staticmethod
    def _semantic_token_stem(token: str) -> str:
        """仅归一化常见英文词尾，避免把语义条件退化为子串匹配。"""
        if len(token) <= 4 or token.isdigit():
            return token
        for suffix in ("ing", "ers", "er", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                return token[:-len(suffix)]
        return token

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
    def _combined_retrieval_query(queries: tuple[str, ...]) -> str:
        """把同一条件的多语言等价词合并为一次 KC 混合检索。"""
        return "\n".join(dict.fromkeys(
            query.strip() for query in queries if query.strip()
        ))

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
        """为多语言语义条件补充英文词形，原文与扩展词合并检索。"""
        original = self._criterion_query(criterion)
        if not original:
            return (), ()
        equivalents = (
            tuple(criterion.resolved_concept.equivalents)
            if criterion.resolved_concept is not None
            else ()
        )
        if equivalents:
            return self._unique_queries((original, *equivalents)), ()
        if criterion.kind != "SEMANTIC_CONCEPT" or len(criterion.values) != 1:
            return (original,), ()
        source_language = detect_unicode_language(original)
        response_language_value = response_language(
            context.config_snapshot, context.original_input
        )
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
                "response_language": response_language_value,
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
                return self._unique_queries((original, *english_topics)), ()
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
    def _unique_queries(values: tuple[str, ...]) -> tuple[str, ...]:
        """按大小写不敏感语义去重，同时保留第一个检索词的原始写法。"""
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(str(value).split())
            key = normalized.casefold()
            if not normalized or key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return tuple(result)

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

    @classmethod
    def _semantic_metadata_matches(
        cls,
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
            str(value).strip()
            for value in criterion.values
            if str(value).strip()
        ]
        if expected and all(
            cls._text_supports_semantic_term(searchable_text, term=value)
            for value in expected
        ):
            return True
        alternatives = [
            str(value).strip()
            for value in semantic_terms
            if str(value).strip()
        ]
        return any(
            cls._text_supports_semantic_term(searchable_text, term=value)
            for value in alternatives
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
