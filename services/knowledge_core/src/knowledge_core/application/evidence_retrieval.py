"""Evidence-stage retrieval, context grouping and citation pack DTOs."""
from uuid import UUID
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from loguru import logger

from knowledge_core.application.query_embeddings import QueryEmbeddingProvider


@dataclass(frozen=True)
class EvidenceScope:
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    document_version_ids: tuple[UUID, ...] = ()


@dataclass(frozen=True)
class EvidenceHit:
    evidence_id: UUID
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    bundle_revision_document_id: UUID | None
    document_id: UUID
    document_version_id: UUID
    parse_view_id: UUID
    evidence_key: str
    evidence_type: str
    content_text: str
    retrieval_text: str
    heading_path: tuple[str, ...]
    locator: dict[str, Any]
    locator_schema_version: str
    source_spans: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    section_key: str | None
    parent_evidence_key: str | None
    ordinal: int
    quality_score: float | None
    local_rank: int
    channel: str
    bundle_title: str | None = None
    document_name: str | None = None
    external_document_id: str | None = None
    document_role: str | None = None
    content_hash: str | None = None


@dataclass
class EvidenceGroupItem:
    item_label: str
    evidence: EvidenceHit
    input_role: str
    final_role: str
    promoted_from_context: bool = False


@dataclass
class EvidenceGroup:
    group_label: str
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    document_version_id: UUID
    parse_view_id: UUID
    items: list[EvidenceGroupItem]
    anchor_evidence_ids: list[UUID]
    token_count: int
    support_grade: str | None = None
    answerable_aspects: list[str] = field(default_factory=list)
    unsupported_aspects: list[str] = field(default_factory=list)


@dataclass
class CitationGroup:
    citation_label: str
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    document_version_id: UUID
    parse_view_id: UUID
    primary_evidence_ids: list[UUID]
    structural_context_ids: list[UUID]
    neighbor_evidence_ids: list[UUID]
    items: list[EvidenceGroupItem]


class EvidenceSearchPort(Protocol):
    async def search_text(self, *, scope: EvidenceScope, query: str, limit: int, max_security_level: int) -> Sequence[EvidenceHit]: ...
    async def search_vector(self, *, scope: EvidenceScope, vector: Sequence[float], limit: int, max_security_level: int) -> Sequence[EvidenceHit]: ...
    async def expand_context(self, *, anchors: Sequence[EvidenceHit], limit: int) -> Sequence[EvidenceHit]: ...


def assemble_groups(
    anchors: Sequence[EvidenceHit], contexts: Sequence[EvidenceHit], *, max_groups: int = 12,
    context_items_per_group: int = 4,
) -> list[EvidenceGroup]:
    """Group only same Document Version/Parse View; anchors become PRIMARY."""
    grouped: dict[tuple[UUID, UUID, UUID], list[EvidenceHit]] = defaultdict(list)
    for anchor in anchors:
        grouped[(anchor.document_version_id, anchor.parse_view_id, anchor.bundle_id)].append(anchor)
    context_by_scope: dict[tuple[UUID, UUID, UUID], list[EvidenceHit]] = defaultdict(list)
    for context in contexts:
        context_by_scope[(context.document_version_id, context.parse_view_id, context.bundle_id)].append(context)
    groups: list[EvidenceGroup] = []
    for group_index, (scope_key, scope_anchors) in enumerate(grouped.items(), 1):
        scope_anchors = _dedupe(scope_anchors)
        first = scope_anchors[0]
        context_items = [
            item for item in _dedupe(context_by_scope.get(scope_key, []))
            if item.evidence_id not in {anchor.evidence_id for anchor in scope_anchors}
        ][:context_items_per_group]
        items = [
            EvidenceGroupItem(
                item_label=f"G{group_index}-A{index}", evidence=anchor,
                input_role="ANCHOR", final_role="PRIMARY",
            ) for index, anchor in enumerate(scope_anchors, 1)
        ]
        items.extend(
            EvidenceGroupItem(
                item_label=f"G{group_index}-C{index}", evidence=context,
                input_role="STRUCTURAL_CONTEXT", final_role="STRUCTURAL_CONTEXT",
            ) for index, context in enumerate(context_items, 1)
        )
        groups.append(EvidenceGroup(
            group_label=f"G{group_index}", collection_id=first.collection_id,
            bundle_id=first.bundle_id, bundle_revision_id=first.bundle_revision_id,
            document_version_id=first.document_version_id, parse_view_id=first.parse_view_id,
            items=items, anchor_evidence_ids=[anchor.evidence_id for anchor in scope_anchors],
            token_count=sum(max(1, len(item.evidence.content_text) // 4) for item in items),
        ))
        if len(groups) >= max_groups:
            break
    return groups


def build_citation_pack(groups: Sequence[EvidenceGroup]) -> list[CitationGroup]:
    """Assign request-local labels only to groups containing PRIMARY items."""
    citations: list[CitationGroup] = []
    for index, group in enumerate(groups, 1):
        primary = [item.evidence.evidence_id for item in group.items if item.final_role == "PRIMARY"]
        if not primary:
            continue
        citations.append(CitationGroup(
            citation_label=f"C{index}", collection_id=group.collection_id,
            bundle_id=group.bundle_id, bundle_revision_id=group.bundle_revision_id,
            document_version_id=group.document_version_id, parse_view_id=group.parse_view_id,
            primary_evidence_ids=primary,
            structural_context_ids=[item.evidence.evidence_id for item in group.items if item.final_role == "STRUCTURAL_CONTEXT"],
            neighbor_evidence_ids=[item.evidence.evidence_id for item in group.items if item.final_role == "NEIGHBOR"],
            items=group.items,
        ))
    return citations


def _dedupe(items: Sequence[EvidenceHit]) -> list[EvidenceHit]:
    seen: set[int] = set()
    result: list[EvidenceHit] = []
    for item in sorted(items, key=lambda value: (value.local_rank, -float(value.quality_score or 0), value.evidence_id)):
        if item.evidence_id not in seen:
            seen.add(item.evidence_id)
            result.append(item)
    return result


class KnowledgeCoreEvidenceRetrievalService:
    def __init__(self, *, search_port: EvidenceSearchPort, query_embedding_provider: QueryEmbeddingProvider | None = None):
        self._search_port = search_port
        self._query_embedding_provider = query_embedding_provider

    async def retrieve(
        self, *, scopes: Sequence[EvidenceScope], query: str,
        query_vectors: dict[UUID, Sequence[float]] | None = None,
        max_evidence: int = 12, context_limit: int = 4, max_security_level: int = 3,
    ) -> list[CitationGroup]:
        citations, _ = await self.retrieve_with_diagnostics(
            scopes=scopes,
            query=query,
            query_vectors=query_vectors,
            max_evidence=max_evidence,
            context_limit=context_limit,
            max_security_level=max_security_level,
        )
        return citations

    async def retrieve_with_diagnostics(
        self, *, scopes: Sequence[EvidenceScope], query: str,
        query_vectors: dict[UUID, Sequence[float]] | None = None,
        max_evidence: int = 12, context_limit: int = 4,
        max_security_level: int = 3,
    ) -> tuple[list[CitationGroup], dict[str, Any]]:
        if not query.strip() or not scopes:
            raise ValueError("query and scopes are required")
        warnings: list[str] = []
        if self._query_embedding_provider is not None:
            try:
                query_vectors = (
                    await self._query_embedding_provider.embed_for_collections(
                        query=query,
                        collection_ids=[
                            scope.collection_id for scope in scopes
                        ],
                    )
                )
            except Exception as exc:
                query_vectors = {}
                warnings.append(
                    "查询向量生成失败，已仅使用全文检索通道"
                )
                logger.exception(
                    "KC Evidence 查询向量生成失败，已降级全文检索 | "
                    "error_type={}",
                    type(exc).__name__,
                )
        anchors: list[EvidenceHit] = []
        scope_reports: list[dict[str, Any]] = []
        successful_channels = 0
        first_failure: Exception | None = None
        for scope in scopes:
            text_error = None
            try:
                text_hits = list(await self._search_port.search_text(
                    scope=scope,
                    query=query,
                    limit=max_evidence,
                    max_security_level=max_security_level,
                ))
                successful_channels += 1
            except Exception as exc:
                text_hits = []
                text_error = type(exc).__name__
                first_failure = first_failure or exc
                warnings.append(
                    f"Bundle {scope.bundle_id} 全文检索失败，"
                    "已继续向量检索"
                )
                logger.exception(
                    "KC Evidence 全文通道失败，已尝试继续 | "
                    "bundle_id={} | error_type={}",
                    scope.bundle_id,
                    text_error,
                )
            anchors.extend(text_hits)
            vector_hits: list[EvidenceHit] = []
            vector_error = None
            if query_vectors and scope.collection_id in query_vectors:
                try:
                    vector_hits = list(await self._search_port.search_vector(
                        scope=scope,
                        vector=query_vectors[scope.collection_id],
                        limit=max_evidence,
                        max_security_level=max_security_level,
                    ))
                    successful_channels += 1
                except Exception as exc:
                    vector_error = type(exc).__name__
                    first_failure = first_failure or exc
                    warnings.append(
                        f"Bundle {scope.bundle_id} 向量检索失败，"
                        "已保留全文结果"
                    )
                    logger.exception(
                        "KC Evidence 向量通道失败，已尝试继续 | "
                        "bundle_id={} | error_type={}",
                        scope.bundle_id,
                        vector_error,
                    )
                anchors.extend(vector_hits)
            scope_reports.append(
                {
                    "collection_id": str(scope.collection_id),
                    "bundle_id": str(scope.bundle_id),
                    "text_hits": len(text_hits),
                    "vector_hits": len(vector_hits),
                    "vector_enabled": bool(
                        query_vectors
                        and scope.collection_id in query_vectors
                    ),
                    "text_error": text_error,
                    "vector_error": vector_error,
                }
            )
        if successful_channels == 0 and first_failure is not None:
            raise first_failure
        raw_anchor_count = len(anchors)
        anchors = _dedupe(anchors)[:max_evidence]
        contexts = await self._search_port.expand_context(anchors=anchors, limit=context_limit)
        groups = assemble_groups(
            anchors,
            contexts,
            max_groups=max_evidence,
            context_items_per_group=context_limit,
        )
        citations = build_citation_pack(groups)
        return citations, {
            "stage": "EVIDENCE",
            "scopes": scope_reports,
            "text_hits": sum(item["text_hits"] for item in scope_reports),
            "vector_hits": sum(
                item["vector_hits"] for item in scope_reports
            ),
            "raw_anchor_hits": raw_anchor_count,
            "selected_anchors": len(anchors),
            "expanded_contexts": len(contexts),
            "evidence_groups": len(groups),
            "citation_groups": len(citations),
            "max_evidence": max_evidence,
            "context_limit": context_limit,
            "warnings": warnings,
        }
