"""KM Asset 查询结果、正文证据与 Asset 引用组合。"""

import re
from typing import Any

from loguru import logger
from platform_core.contracts import AssetSearchPlanV1
from platform_core.prompts import StrictPromptRenderer

from agent_runtime.language import (
    localized_message,
    response_language,
)
from agent_runtime.runtime import ExecutionContext, SkillResult
from agent_runtime.specialists.data_query.contracts import QueryResult
from agent_runtime.specialists.document.contracts import DocumentRetrievalResult
from agent_runtime.specialists.response_composer.contracts import (
    GroundedAnswer,
    QueryResultReferenceCard,
)
from agent_runtime.specialists.response_composer.skill import (
    _CITATION_PATTERN,
    _normalize_citations,
)


_KM_TEXT = {
    "zh-CN": {
        "complete": (
            "问数结果共命中 {total_count} 个相关 Asset，以下全部列出。"
        ),
        "truncated": (
            "问数结果共命中 {total_count} 个相关 Asset；以下列出前 "
            "{shown_count} 个，其他结果已截断。"
        ),
        "source_truncated": (
            "问数结果至少命中 {total_count} 个相关 Asset；以下列出前 "
            "{shown_count} 个，其他结果已截断。"
        ),
        "not_found": "未找到匹配的 Asset。",
        "product": "产品",
        "solution": "解决方案",
        "preference": (
            "其中，{references} 还命中“{preference_text}”偏好，可优先参考。"
        ),
        "related_assets": "相关 Asset：",
        "count": "当前共有 {asset_count} 个可用 Asset。",
        "group_intro": "按条件分组的 Asset 统计如下：",
        "semantic_count": (
            "语义相关性不能用于精确统计总数；以下提供 {asset_count} 个"
            "较新且有正文证据的相关 Asset 供参考。"
        ),
        "semantic_matches": (
            "以下是 {asset_count} 个满足条件的 Asset；"
            "正文证据已用于语义条件或偏好排序。"
        ),
    },
    "ja-JP": {
        "complete": (
            "データクエリで関連する Asset が {total_count} 件見つかりました。"
            "以下にすべて表示します。"
        ),
        "truncated": (
            "データクエリで関連する Asset が {total_count} 件見つかりました。"
            "先頭 {shown_count} 件を表示し、残りは省略しています。"
        ),
        "source_truncated": (
            "データクエリで関連する Asset が少なくとも {total_count} 件"
            "見つかりました。先頭 {shown_count} 件を表示し、残りは"
            "省略しています。"
        ),
        "not_found": "一致する Asset は見つかりませんでした。",
        "product": "製品",
        "solution": "ソリューション",
        "preference": (
            "このうち、{references} は「{preference_text}」という希望条件にも"
            "一致するため、優先的に参照できます。"
        ),
        "related_assets": "関連する Asset：",
        "count": "現在利用可能な Asset は {asset_count} 件です。",
        "group_intro": "条件別の Asset 集計結果は次のとおりです。",
        "semantic_count": (
            "意味的な関連性から正確な総数を算出することはできません。"
            "参考として、本文の根拠がある新しい関連 Asset を "
            "{asset_count} 件示します。"
        ),
        "semantic_matches": (
            "条件に一致する Asset は {asset_count} 件です。"
            "意味条件または希望条件による順位付けには本文の根拠を"
            "使用しています。"
        ),
    },
    "ko-KR": {
        "complete": (
            "데이터 쿼리에서 관련 Asset {total_count}개를 찾았습니다. "
            "아래에 모두 표시합니다."
        ),
        "truncated": (
            "데이터 쿼리에서 관련 Asset {total_count}개를 찾았습니다. "
            "처음 {shown_count}개만 표시하고 나머지는 생략했습니다."
        ),
        "source_truncated": (
            "데이터 쿼리에서 관련 Asset을 최소 {total_count}개 찾았습니다. "
            "처음 {shown_count}개만 표시하고 나머지는 생략했습니다."
        ),
        "not_found": "일치하는 Asset을 찾지 못했습니다.",
        "product": "제품",
        "solution": "솔루션",
        "preference": (
            "이 중 {references}은(는) ‘{preference_text}’ 선호 조건도 "
            "충족하므로 우선 참고할 수 있습니다."
        ),
        "related_assets": "관련 Asset:",
        "count": "현재 사용 가능한 Asset은 {asset_count}개입니다.",
        "group_intro": "조건별 Asset 집계 결과는 다음과 같습니다.",
        "semantic_count": (
            "의미적 관련성으로는 정확한 전체 개수를 집계할 수 없습니다. "
            "참고할 수 있도록 본문 근거가 있는 최신 관련 Asset "
            "{asset_count}개를 제공합니다."
        ),
        "semantic_matches": (
            "조건을 충족하는 Asset은 {asset_count}개입니다. 의미 조건 또는 "
            "선호 조건의 순위를 정할 때 본문 근거를 사용했습니다."
        ),
    },
    "en-US": {
        "complete": (
            "The data query found {total_count} related assets; all are "
            "listed below."
        ),
        "truncated": (
            "The data query found {total_count} related assets. The first "
            "{shown_count} are listed; the rest are truncated."
        ),
        "source_truncated": (
            "The data query found at least {total_count} related assets. "
            "The first {shown_count} are listed; the rest are truncated."
        ),
        "not_found": "No matching assets were found.",
        "product": "Product",
        "solution": "Solution",
        "preference": (
            "Among these, {references} also match the “{preference_text}” "
            "preference and may be prioritized."
        ),
        "related_assets": "Related assets:",
        "count": "There are {asset_count} available assets.",
        "group_intro": "The grouped asset counts are as follows:",
        "semantic_count": (
            "Semantic relevance cannot produce an exact total. Here are "
            "{asset_count} recent relevant assets with content evidence."
        ),
        "semantic_matches": (
            "Here are {asset_count} matching assets; content evidence was "
            "used for semantic conditions or preferences."
        ),
    },
}


def _km_text(language: str, key: str, **values: Any) -> str:
    """集中生成 KM 用户可见文本，未知语言使用英文安全兜底。"""
    templates = _KM_TEXT.get(language) or _KM_TEXT["en-US"]
    return templates[key].format(**values)


class KmAssetComposerMixin:
    """只为 KM Asset Composer 提供清单和引用组合。"""

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
        if source_truncated:
            key = "source_truncated"
        elif truncated:
            key = "truncated"
        else:
            key = "complete"
        return _km_text(
            language,
            key,
            total_count=total_count,
            shown_count=shown_count,
        )

    @staticmethod
    def _validate_enumeration_body(
        answer: str,
        *,
        assets: list[dict[str, Any]],
        allowed: dict[str, Any],
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
        unknown = labels - allowed.keys()
        if unknown:
            raise ValueError(f"主题 Asset 清单使用未知引用：{sorted(unknown)}")
        if allowed and not labels.intersection(allowed):
            raise ValueError("已有正文证据但清单未使用文档引用")
        labels_by_bundle = (
            KmAssetComposerMixin._citation_labels_by_bundle(allowed)
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
            if not expected_labels:
                raise ValueError(
                    "Asset 清单缺少可投影的 Bundle 正文引用："
                    f"{item.get('title')}"
                )
            if not segment_labels.intersection(expected_labels):
                raise ValueError(
                    "Asset 清单缺少对应 Bundle 的正文引用："
                    f"{item.get('title')}"
                )
            if segment_labels - expected_labels:
                raise ValueError(
                    "Asset 清单使用了其他 Bundle 的正文引用："
                    f"{item.get('title')}"
                )

    @staticmethod
    def _enumeration_fallback(
        assets: list[dict[str, Any]], *, language: str,
        allowed: dict[str, Any] | None = None,
    ) -> str:
        labels_by_bundle = KmAssetComposerMixin._citation_labels_by_bundle(
            allowed or {}
        )
        if not assets:
            return _km_text(language, "not_found")
        lines = []
        for index, item in enumerate(assets, start=1):
            title = str(item.get("title") or item.get("asset_id") or "Asset")
            product = str(item.get("product") or "").strip()
            solution = str(item.get("solution") or "").strip()
            details = "; ".join(
                value
                for value in (
                    (
                        f"{_km_text(language, 'product')}: {product}"
                        if product
                        else ""
                    ),
                    (
                        f"{_km_text(language, 'solution')}: {solution}"
                        if solution
                        else ""
                    ),
                )
                if value
            )
            suffix = f" — {details}" if details else ""
            citation_labels = labels_by_bundle.get(
                str(item.get("bundle_id") or "").casefold(), ()
            )
            if not citation_labels:
                raise ValueError(
                    "KM_ASSET_CITATION_MISSING: "
                    f"{item.get('title') or item.get('asset_id') or 'Asset'}"
                )
            references = f"[{citation_labels[0]}]"
            lines.append(f"{index}. **{title}**{suffix} {references}")
        return "\n".join(lines)

    @staticmethod
    def _without_query_references(answer: GroundedAnswer) -> GroundedAnswer:
        """KM 结果保留问数数据，但不向用户投影 Q 引用。"""
        body = re.sub(r"[ \t]*\[Q\d+\]", "", answer.answer)
        body = "\n".join(line.rstrip() for line in body.splitlines()).strip()
        payload = answer.model_dump(mode="json")
        payload.update({
            "answer": body,
            "used_citation_labels": [
                label
                for label in answer.used_citation_labels
                if not label.startswith("Q")
            ],
            "references": [
                item.model_dump(mode="json")
                for item in answer.references
                if not isinstance(item, QueryResultReferenceCard)
            ],
        })
        return GroundedAnswer.model_validate(payload)

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

    @classmethod
    def _preference_summary(
        cls,
        *,
        search_plan: AssetSearchPlanV1 | None,
        retrieval: DocumentRetrievalResult,
        assets: list[dict[str, Any]],
        allowed: dict[str, Any],
        language: str,
    ) -> str:
        """按检索条件矩阵说明展示结果实际命中的软偏好。"""
        if search_plan is None or not search_plan.preferences:
            return ""
        query_diagnostics = retrieval.citation_pack.query_plan.get(
            "diagnostics", {}
        )
        evidence_diagnostics = (
            query_diagnostics.get("evidence", {})
            if isinstance(query_diagnostics, dict)
            else {}
        )
        matrix = (
            evidence_diagnostics.get("requirements", ())
            if isinstance(evidence_diagnostics, dict)
            else ()
        )
        matched_by_bundle = {
            str(item.get("bundle_id") or "").casefold(): {
                str(value) for value in item.get("matched_preferences") or ()
            }
            for item in matrix
            if isinstance(item, dict) and item.get("bundle_id")
        }
        labels_by_bundle = cls._citation_labels_by_bundle(allowed)
        lines: list[str] = []
        for preference in sorted(
            search_plan.preferences, key=lambda item: item.priority
        ):
            labels: list[str] = []
            for asset in assets:
                bundle_id = str(asset.get("bundle_id") or "").casefold()
                if preference.preference_id not in matched_by_bundle.get(
                    bundle_id, set()
                ):
                    continue
                citation_labels = labels_by_bundle.get(bundle_id, ())
                if citation_labels:
                    labels.append(citation_labels[0])
            labels = list(dict.fromkeys(labels))
            if not labels:
                continue
            preference_text = " / ".join(
                str(value) for value in preference.criterion.values
            )
            references = "、".join(f"[{label}]" for label in labels)
            if language != "zh-CN":
                references = ", ".join(f"[{label}]" for label in labels)
            lines.append(
                _km_text(
                    language,
                    "preference",
                    references=references,
                    preference_text=preference_text,
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _select_result_assets(
        candidate_assets: list[dict[str, Any]],
        citations: tuple[Any, ...],
        *,
        semantic: bool,
        result_limit: int,
    ) -> list[dict[str, Any]]:
        """纯元数据保留问数清单；语义清单只保留有同 Bundle C 的 Asset。"""
        asset_by_bundle = {
            str(item.get("bundle_id") or "").casefold(): item
            for item in candidate_assets
            if str(item.get("bundle_id") or "")
        }
        cited_bundles = {
            str(citation.bundle_id).casefold()
            for citation in citations
        }
        ordered = (
            (
                asset_by_bundle.get(str(citation.bundle_id).casefold())
                for citation in citations
            )
            if semantic
            else (
                asset
                for asset in candidate_assets
                if str(asset.get("bundle_id") or "").casefold()
                in cited_bundles
            )
        )
        selected: list[dict[str, Any]] = []
        seen_bundles: set[str] = set()
        for asset in ordered:
            if asset is None:
                continue
            bundle_key = str(asset.get("bundle_id") or "").casefold()
            if not bundle_key or bundle_key in seen_bundles:
                continue
            selected.append(asset)
            seen_bundles.add(bundle_key)
            if len(selected) >= result_limit:
                break
        return selected

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
        heading = _km_text(language, "related_assets")
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

    async def _compose_km_asset_enumeration(
        self,
        context: ExecutionContext,
        query: QueryResult,
        retrieval: DocumentRetrievalResult,
        *,
        search_plan: AssetSearchPlanV1 | None = None,
    ) -> SkillResult:
        """用问数冻结资格边界，再按同 Bundle 正文证据确定最终清单。"""
        scope = self._document_scope(context)
        candidate_assets = [
            dict(item) for item in scope.get("assets") or ()
            if isinstance(item, dict)
        ]
        if not candidate_assets:
            candidate_assets = self._enumeration_assets_from_query(query)
        if query.row_count == 0 and not candidate_assets:
            language = response_language(
                context.config_snapshot, context.original_input
            )
            return self._result(context, GroundedAnswer(
                answer=_km_text(language, "not_found"),
                status="READY",
            ))
        citations = retrieval.citation_pack.citations
        allowed = {item.citation_label: item for item in citations}
        semantic = bool(
            search_plan is not None
            and (
                search_plan.has_semantic_eligibility
                or search_plan.preferences
            )
        )
        result_limit = (
            search_plan.result_assets.target_count
            if search_plan is not None
            else 10
        )
        assets = self._select_result_assets(
            candidate_assets,
            citations,
            semantic=semantic,
            result_limit=result_limit,
        )
        if query.row_count > 0 and not candidate_assets:
            raise ValueError("KM_ASSET_ENUMERATION_LIST_EMPTY")
        language = response_language(
            context.config_snapshot, context.original_input
        )
        if candidate_assets and not assets:
            return self._result(context, GroundedAnswer(
                answer=localized_message("insufficient_evidence", language),
                status="INSUFFICIENT_EVIDENCE",
                warnings=tuple(dict.fromkeys((
                    *retrieval.warnings,
                    "没有 Asset 同时具备查询资格与 Asset 正文引用",
                ))),
            ))
        if semantic:
            semantic_count_fallback = bool(
                search_plan is not None
                and "SEMANTIC_TOTAL_COUNT" in search_plan.unsupported_requests
            )
            if semantic_count_fallback:
                prefix = _km_text(
                    language,
                    "semantic_count",
                    asset_count=len(assets),
                )
            else:
                prefix = _km_text(
                    language,
                    "semantic_matches",
                    asset_count=len(assets),
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
        body = self._enumeration_fallback(
            assets,
            language=language,
            allowed=allowed,
        )
        preference_summary = self._preference_summary(
            search_plan=search_plan,
            retrieval=retrieval,
            assets=assets,
            allowed=allowed,
            language=language,
        )
        if preference_summary:
            body = f"{body}\n\n{preference_summary}"
        answer = f"{prefix}\n\n{body}".strip()
        used_labels = tuple(dict.fromkeys(
            label for label in _CITATION_PATTERN.findall(body)
            if label in allowed
        ))
        references = tuple(
            self._reference_card(allowed[label])
            for label in used_labels
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
            used_citation_labels=used_labels,
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
        """组合精确聚合结果与只包含 Asset C 的支撑清单。"""
        aggregate_answer = self._asset_aggregate_answer(
            context,
            query,
            search_plan,
        )
        if not query.supporting_rows:
            return self._result(context, aggregate_answer)
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
            language = response_language(
                context.config_snapshot, context.original_input
            )
            return self._result(context, GroundedAnswer(
                answer=localized_message("insufficient_evidence", language),
                status="INSUFFICIENT_EVIDENCE",
                warnings=("Asset 清单缺少必需的 Asset 正文引用",),
            ))

        payload = self._asset_aggregate_answer(
            context,
            query,
            search_plan,
        )
        if not query.supporting_rows:
            return self._result(context, payload)
        return self._result(context, payload.model_copy(update={
            "warnings": tuple(dict.fromkeys((
                *payload.warnings,
                "支撑 Asset 缺少必需的 Asset 正文引用，未在回答中展示",
            ))),
        }))

    @classmethod
    def _asset_aggregate_answer(
        cls,
        context: ExecutionContext,
        query: QueryResult,
        search_plan: AssetSearchPlanV1,
    ) -> GroundedAnswer:
        """确定性组合 KM COUNT/GROUP，避免结构化结果再次调用 LLM。"""
        language = response_language(
            context.config_snapshot, context.original_input
        )
        if search_plan.operation == "COUNT":
            answer = _km_text(
                language,
                "count",
                asset_count=cls._query_asset_count(query),
            )
        elif search_plan.operation == "GROUP":
            answer = cls._grouped_asset_table(
                query,
                language=language,
            )
        else:
            raise ValueError("KM_ASSET_AGGREGATE_OPERATION_INVALID")
        return GroundedAnswer(
            answer=answer,
            status="READY",
            query_results=(query.model_dump(mode="json"),),
            warnings=query.warnings,
        )

    @staticmethod
    def _query_asset_count(query: QueryResult) -> int:
        """从冻结 QueryResult 读取精确 Asset 数量。"""
        if len(query.rows) != 1:
            raise ValueError("KM_ASSET_COUNT_RESULT_INVALID")
        value = next((
            item
            for key, item in query.rows[0].items()
            if str(key).casefold() == "asset_count"
        ), None)
        if isinstance(value, bool):
            raise ValueError("KM_ASSET_COUNT_RESULT_INVALID")
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("KM_ASSET_COUNT_RESULT_INVALID") from exc
        if count < 0:
            raise ValueError("KM_ASSET_COUNT_RESULT_INVALID")
        return count

    @staticmethod
    def _grouped_asset_table(
        query: QueryResult,
        *,
        language: str,
    ) -> str:
        """把冻结的分组结果渲染为稳定 Markdown 表格。"""
        if not query.rows:
            return _km_text(language, "not_found")
        fields = list(dict.fromkeys(
            str(key)
            for row in query.rows
            for key in row
        ))
        labels = {
            "asset_count": {
                "zh-CN": "Asset 数量",
                "ja-JP": "Asset 件数",
                "ko-KR": "Asset 수",
                "en-US": "Asset Count",
            },
            "product": {
                "zh-CN": "产品（领域）",
                "ja-JP": "製品（ドメイン）",
                "ko-KR": "제품(도메인)",
                "en-US": "Product (Domain)",
            },
        }

        def label(field: str) -> str:
            localized = labels.get(field.casefold())
            if localized is None:
                return field
            return localized.get(language, localized["en-US"])

        def cell(value: Any) -> str:
            return str(value if value is not None else "-").replace(
                "|", "\\|"
            ).replace("\n", " ")

        header = "| " + " | ".join(label(field) for field in fields) + " |"
        divider = "| " + " | ".join("---" for _ in fields) + " |"
        rows = [
            "| " + " | ".join(cell(row.get(field)) for field in fields) + " |"
            for row in query.rows
        ]
        return "\n\n".join((
            _km_text(language, "group_intro"),
            "\n".join((header, divider, *rows)),
        ))
