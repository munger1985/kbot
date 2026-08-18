"""在授权候选内使用 Collection Retrieval LLM 做对象和证据组重排。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from typing import Any, Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from knowledge_core.application.evidence_retrieval import (
    CitationGroup,
    EvidenceGroupItem,
)
from knowledge_core.application.retrieval import BundleCandidate
from knowledge_core.domain.model_bindings import collection_model_id
from platform_core.dictionary import ModelCategory


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _CandidateDecision(_StrictModel):
    candidate_label: str
    relevance: Literal[
        "DIRECT",
        "STRONG",
        "POSSIBLE",
        "SEMANTIC_ONLY",
        "IRRELEVANT",
    ]
    matched_requirements: tuple[str, ...] = ()
    reason_refs: tuple[str, ...] = ()


class _CandidateDecisionBatch(_StrictModel):
    decisions: tuple[_CandidateDecision, ...]


class _EvidenceDecision(_StrictModel):
    group_label: str
    support: Literal[
        "DIRECT_SUPPORT",
        "PARTIAL_SUPPORT",
        "CONTEXT_ONLY",
        "CONTRADICTS",
        "NO_SUPPORT",
    ]
    primary_item_labels: tuple[str, ...] = ()
    structural_context_labels: tuple[str, ...] = ()
    answerable_aspects: tuple[str, ...] = ()
    unsupported_aspects: tuple[str, ...] = ()


class _EvidenceDecisionBatch(_StrictModel):
    decisions: tuple[_EvidenceDecision, ...]


class CollectionRetrievalModelResolver:
    """从 KC 自有 Collection 配置解析可调用的 LLM 技术名称。"""

    def __init__(self, *, uow_factory, model_config_client):
        self._uow_factory = uow_factory
        self._model_config_client = model_config_client

    async def resolve(self, collection_id: UUID) -> tuple[UUID, str]:
        async with self._uow_factory() as uow:
            if uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work 未初始化")
            collection = await uow.collections.get_by_id(
                collection_id=collection_id
            )
            if collection is None or collection.status != "ACTIVE":
                raise ValueError("Collection 不存在或未启用")
            model_id = collection_model_id(collection, "retrieval_llm")
            if model_id is None:
                raise ValueError("Collection 未配置 models.retrieval_llm")
        model = await self._model_config_client.get_model(model_id)
        if int(model.get("category") or 0) != int(ModelCategory.LLM):
            raise ValueError("Collection Retrieval 模型不是 LLM")
        if model.get("status") != "ACTIVE":
            raise ValueError("Collection Retrieval LLM 未启用")
        served_name = str(model.get("served_model_name") or "").strip()
        if not served_name:
            raise ValueError("Collection Retrieval LLM 缺少 served_model_name")
        return model_id, served_name


class KnowledgeCoreLlmReranker:
    """LLM 只做受约束的类别判断，不产生裸 Chunk 数值分数。"""

    _CANDIDATE_ORDER = {
        "DIRECT": 0,
        "STRONG": 1,
        "POSSIBLE": 2,
        "SEMANTIC_ONLY": 3,
        "IRRELEVANT": 4,
    }
    _EVIDENCE_ORDER = {
        "DIRECT_SUPPORT": 0,
        "PARTIAL_SUPPORT": 1,
        "CONTRADICTS": 2,
        "CONTEXT_ONLY": 3,
        "NO_SUPPORT": 4,
    }
    def __init__(
        self,
        *,
        model_resolver: CollectionRetrievalModelResolver,
        model_client,
        prompt_resolver,
    ):
        self._model_resolver = model_resolver
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def rerank_candidates(
        self,
        *,
        query: str,
        candidates: Sequence[BundleCandidate],
        coverage_mode: Literal["BREADTH", "BALANCED"] = "BALANCED",
    ) -> tuple[list[BundleCandidate], dict[str, Any], list[str]]:
        grouped: dict[UUID, list[BundleCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.collection_id].append(candidate)
        ranked: dict[UUID, list[BundleCandidate]] = {}
        details: list[dict[str, Any]] = []
        warnings: list[str] = []
        successful = 0
        for collection_id in sorted(grouped):
            items = grouped[collection_id]
            try:
                model_id, model_name = await self._model_resolver.resolve(
                    collection_id
                )
                prompt = await self._prompt_resolver.resolve(
                    "knowledge_core.candidate_rerank"
                )
                labels = {
                    f"B{index}": item
                    for index, item in enumerate(items, 1)
                }
                response = await self._model_client.get_llm_json(
                    served_model_name=model_name,
                    prompt=[
                        {"role": "system", "content": prompt.content},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "query": query,
                                    "coverage_mode": coverage_mode,
                                    "candidates": [
                                        self._candidate_payload(label, item)
                                        for label, item in labels.items()
                                    ],
                                },
                                ensure_ascii=False,
                                default=str,
                            ),
                        },
                    ],
                    max_tokens=4096,
                )
                batch = _CandidateDecisionBatch.model_validate(response)
                decisions = self._complete_decisions(
                    expected=set(labels),
                    decision_count=len(batch.decisions),
                    actual={
                        item.candidate_label: item
                        for item in batch.decisions
                    },
                )
                breadth_preserved_count = 0
                selected = []
                for label, decision in sorted(
                    decisions.items(),
                    key=lambda pair: (
                        self._CANDIDATE_ORDER[pair[1].relevance],
                        labels[pair[0]].local_rank,
                        pair[0],
                    ),
                ):
                    candidate = labels[label]
                    preserve_for_breadth = (
                        coverage_mode == "BREADTH"
                        and decision.relevance == "IRRELEVANT"
                    )
                    if (
                        decision.relevance != "IRRELEVANT"
                        or preserve_for_breadth
                    ):
                        selected.append(candidate)
                    if preserve_for_breadth:
                        breadth_preserved_count += 1
                ranked[collection_id] = selected
                successful += 1
                details.append(
                    {
                        "collection_id": str(collection_id),
                        "model_id": str(model_id),
                        "served_model_name": model_name,
                        "prompt": prompt.ref(),
                        "input_count": len(items),
                        "output_count": len(selected),
                        "breadth_preserved_count": (
                            breadth_preserved_count
                        ),
                        "status": "SUCCEEDED",
                    }
                )
            except Exception as exc:
                ranked[collection_id] = items
                warnings.append(
                    f"Collection {collection_id} 的对象级 LLM 重排失败，"
                    "已保留 RRF 顺序"
                )
                details.append(
                    {
                        "collection_id": str(collection_id),
                        "input_count": len(items),
                        "output_count": len(items),
                        "status": "DEGRADED",
                        "error_type": type(exc).__name__,
                    }
                )
        output = self._fair_interleave(ranked)
        return output, {
            "enabled": True,
            "stage": "DISCOVERY_OBJECT",
            "status": self._status(successful, len(grouped)),
            "collections": details,
        }, warnings

    async def rerank_evidence(
        self,
        *,
        query: str,
        citations: Sequence[CitationGroup],
        coverage_mode: Literal["BREADTH", "BALANCED"] = "BALANCED",
    ) -> tuple[list[CitationGroup], dict[str, Any], list[str]]:
        grouped: dict[UUID, list[CitationGroup]] = defaultdict(list)
        for citation in citations:
            grouped[citation.collection_id].append(citation)
        decisions_by_group: dict[str, _EvidenceDecision] = {}
        fallback_groups: set[str] = set()
        details: list[dict[str, Any]] = []
        warnings: list[str] = []
        successful = 0
        for collection_id in sorted(grouped):
            items = grouped[collection_id]
            try:
                model_id, model_name = await self._model_resolver.resolve(
                    collection_id
                )
                prompt = await self._prompt_resolver.resolve(
                    "knowledge_core.evidence_group_rerank"
                )
                response = await self._model_client.get_llm_json(
                    served_model_name=model_name,
                    prompt=[
                        {"role": "system", "content": prompt.content},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "query": query,
                                    "coverage_mode": coverage_mode,
                                    "groups": [
                                        self._evidence_payload(item)
                                        for item in items
                                    ],
                                },
                                ensure_ascii=False,
                                default=str,
                            ),
                        },
                    ],
                    max_tokens=4096,
                )
                batch = _EvidenceDecisionBatch.model_validate(response)
                expected = {item.citation_label for item in items}
                decisions = self._complete_decisions(
                    expected=expected,
                    decision_count=len(batch.decisions),
                    actual={
                        item.group_label: item
                        for item in batch.decisions
                    },
                )
                for group in items:
                    self._validate_evidence_labels(
                        group, decisions[group.citation_label]
                    )
                decisions_by_group.update(decisions)
                successful += 1
                details.append(
                    {
                        "collection_id": str(collection_id),
                        "model_id": str(model_id),
                        "served_model_name": model_name,
                        "prompt": prompt.ref(),
                        "input_count": len(items),
                        "status": "SUCCEEDED",
                    }
                )
            except Exception as exc:
                fallback_groups.update(
                    item.citation_label for item in items
                )
                warnings.append(
                    f"Collection {collection_id} 的 Evidence Group LLM "
                    "重排失败，已保留确定性分组"
                )
                details.append(
                    {
                        "collection_id": str(collection_id),
                        "input_count": len(items),
                        "status": "DEGRADED",
                        "error_type": type(exc).__name__,
                    }
                )
        selected: list[tuple[int, int, CitationGroup]] = []
        breadth_preserved_count = 0
        for index, citation in enumerate(citations):
            if citation.citation_label in fallback_groups:
                selected.append((1, index, citation))
                continue
            decision = decisions_by_group[citation.citation_label]
            if decision.support in {"NO_SUPPORT", "CONTEXT_ONLY"}:
                if coverage_mode == "BREADTH":
                    selected.append(
                        (
                            self._EVIDENCE_ORDER[decision.support],
                            index,
                            citation,
                        )
                    )
                    breadth_preserved_count += 1
                continue
            filtered = self._filter_citation(citation, decision)
            if filtered is not None:
                selected.append(
                    (
                        self._EVIDENCE_ORDER[decision.support],
                        index,
                        filtered,
                    )
                )
        ordered = [
            item
            for _, _, item in sorted(
                selected, key=lambda value: (value[0], value[1])
            )
        ]
        relabeled = [
            CitationGroup(
                citation_label=f"C{index}",
                collection_id=item.collection_id,
                bundle_id=item.bundle_id,
                bundle_revision_id=item.bundle_revision_id,
                document_version_id=item.document_version_id,
                parse_view_id=item.parse_view_id,
                primary_evidence_ids=item.primary_evidence_ids,
                structural_context_ids=item.structural_context_ids,
                neighbor_evidence_ids=item.neighbor_evidence_ids,
                items=item.items,
            )
            for index, item in enumerate(ordered, 1)
        ]
        return relabeled, {
            "enabled": True,
            "stage": "EVIDENCE_GROUP",
            "status": self._status(successful, len(grouped)),
            "collections": details,
            "input_count": len(citations),
            "output_count": len(relabeled),
            "breadth_preserved_count": breadth_preserved_count,
        }, warnings

    @staticmethod
    def _candidate_payload(
        label: str, item: BundleCandidate
    ) -> dict[str, Any]:
        return {
            "candidate_label": label,
            "title": item.display_title,
            "candidate_scope": item.candidate_scope,
            "matched_members": item.matched_members,
            "match_signals": item.match_signals,
            "profile_text": item.profile_text[:12000],
        }

    @staticmethod
    def _evidence_payload(item: CitationGroup) -> dict[str, Any]:
        return {
            "group_label": item.citation_label,
            "items": [
                {
                    "item_label": group_item.item_label,
                    "role": group_item.final_role,
                    "evidence_type": group_item.evidence.evidence_type,
                    "heading_path": group_item.evidence.heading_path,
                    "content": group_item.evidence.content_text[:4000],
                }
                for group_item in item.items
            ],
        }

    @staticmethod
    def _complete_decisions(
        *, expected: set[str], decision_count: int, actual: dict
    ):
        if decision_count != len(expected) or set(actual) != expected:
            raise ValueError("LLM 重排结果的 Label 集合与输入不一致")
        return actual

    @staticmethod
    def _validate_evidence_labels(
        citation: CitationGroup, decision: _EvidenceDecision
    ) -> None:
        primary = {
            item.item_label
            for item in citation.items
            if item.final_role == "PRIMARY"
        }
        context = {
            item.item_label
            for item in citation.items
            if item.final_role == "STRUCTURAL_CONTEXT"
        }
        if not set(decision.primary_item_labels).issubset(primary):
            raise ValueError("LLM 返回了不存在的 PRIMARY Evidence Label")
        if not set(decision.structural_context_labels).issubset(context):
            raise ValueError("LLM 返回了不存在的上下文 Evidence Label")
        if (
            decision.support
            in {"DIRECT_SUPPORT", "PARTIAL_SUPPORT", "CONTRADICTS"}
            and not decision.primary_item_labels
        ):
            raise ValueError("受支持的 Evidence Group 必须选择 PRIMARY")

    @staticmethod
    def _filter_citation(
        citation: CitationGroup, decision: _EvidenceDecision
    ) -> CitationGroup | None:
        primary_labels = set(decision.primary_item_labels)
        context_labels = set(decision.structural_context_labels)
        items: list[EvidenceGroupItem] = [
            item
            for item in citation.items
            if (
                item.final_role == "PRIMARY"
                and item.item_label in primary_labels
            )
            or (
                item.final_role == "STRUCTURAL_CONTEXT"
                and (
                    not context_labels
                    or item.item_label in context_labels
                )
            )
        ]
        primary_ids = [
            item.evidence.evidence_id
            for item in items
            if item.final_role == "PRIMARY"
        ]
        if not primary_ids:
            return None
        return CitationGroup(
            citation_label=citation.citation_label,
            collection_id=citation.collection_id,
            bundle_id=citation.bundle_id,
            bundle_revision_id=citation.bundle_revision_id,
            document_version_id=citation.document_version_id,
            parse_view_id=citation.parse_view_id,
            primary_evidence_ids=primary_ids,
            structural_context_ids=[
                item.evidence.evidence_id
                for item in items
                if item.final_role == "STRUCTURAL_CONTEXT"
            ],
            neighbor_evidence_ids=[],
            items=items,
        )

    @staticmethod
    def _fair_interleave(
        grouped: dict[UUID, list[BundleCandidate]],
    ) -> list[BundleCandidate]:
        output: list[BundleCandidate] = []
        max_items = max((len(items) for items in grouped.values()), default=0)
        for rank in range(max_items):
            for collection_id in sorted(grouped):
                if rank < len(grouped[collection_id]):
                    output.append(grouped[collection_id][rank])
        return output

    @staticmethod
    def _status(successful: int, total: int) -> str:
        if total == 0 or successful == 0:
            return "DEGRADED"
        return "SUCCEEDED" if successful == total else "PARTIAL"


def public_candidate(candidate: BundleCandidate) -> dict[str, Any]:
    """Discovery API 不向调用方暴露仅供重排的完整 Profile 文本。"""
    payload = asdict(candidate)
    payload.pop("profile_text", None)
    return payload
