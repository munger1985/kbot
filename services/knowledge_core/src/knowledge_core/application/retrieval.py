"""Discovery-stage retrieval contracts and Bundle-level aggregation."""
from uuid import UUID
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from loguru import logger

from knowledge_core.application.query_embeddings import QueryEmbeddingProvider


@dataclass(frozen=True)
class DiscoveryHit:
    collection_id: UUID
    collection_key: str
    bundle_id: UUID
    bundle_revision_id: UUID
    object_type: str
    profile_key: str
    display_title: str
    local_rank: int
    channel: str
    score: float | None = None
    matched_member_key: str | None = None
    member_count: int = 0
    coverage: dict[str, Any] = field(default_factory=dict)
    profile_text: str = ""


@dataclass
class BundleCandidate:
    collection_id: UUID
    collection_key: str
    bundle_id: UUID
    bundle_revision_id: UUID
    display_title: str
    member_count: int
    matched_members: list[str]
    match_signals: list[str]
    local_rank: int
    rrf_score: float
    candidate_scope: str
    profile_text: str = ""


class DiscoverySearchPort(Protocol):
    async def search_text(self, *, collection_id: UUID, query: str, limit: int, max_security_level: int) -> Sequence[DiscoveryHit]: ...
    async def search_vector(self, *, collection_id: UUID, vector: Sequence[float], limit: int, max_security_level: int) -> Sequence[DiscoveryHit]: ...


def aggregate_candidates(
    hits: Sequence[DiscoveryHit], *, per_collection_limit: int = 20, rrf_k: int = 60,
) -> list[BundleCandidate]:
    """Collapse Bundle/Document hits before applying RRF and fair pooling."""
    if per_collection_limit < 1:
        raise ValueError("per_collection_limit must be positive")
    grouped: dict[tuple[UUID, UUID, UUID], list[DiscoveryHit]] = defaultdict(list)
    for hit in hits:
        grouped[(hit.collection_id, hit.bundle_id, hit.bundle_revision_id)].append(hit)
    candidates: list[BundleCandidate] = []
    for (collection_id, bundle_id, revision_id), bundle_hits in grouped.items():
        best = sorted(bundle_hits, key=lambda item: (item.local_rank, item.profile_key))
        first = best[0]
        signals = sorted({item.channel for item in bundle_hits})
        matched = sorted({item.matched_member_key or item.profile_key for item in bundle_hits if item.object_type == "DOCUMENT"})
        ranks_by_channel: dict[str, int] = {}
        for item in best:
            ranks_by_channel[item.channel] = min(ranks_by_channel.get(item.channel, item.local_rank), item.local_rank)
        rrf = sum(1.0 / (rrf_k + rank) for rank in ranks_by_channel.values())
        member_count = max(item.member_count for item in bundle_hits)
        scope = "SINGLE_MEMBER" if member_count == 1 else ("MATCHED_MEMBERS" if matched else "BUNDLE_ALL")
        profile_text = "\n\n".join(
            dict.fromkeys(
                item.profile_text.strip()
                for item in best
                if item.profile_text.strip()
            )
        )[:12000]
        candidates.append(BundleCandidate(
            collection_id=collection_id, collection_key=first.collection_key,
            bundle_id=bundle_id, bundle_revision_id=revision_id,
            display_title=first.display_title, member_count=member_count,
            matched_members=matched, match_signals=signals,
            local_rank=min(item.local_rank for item in bundle_hits),
            rrf_score=rrf, candidate_scope=scope,
            profile_text=profile_text,
        ))
    candidates.sort(key=lambda item: (-item.rrf_score, item.collection_key, item.bundle_id))
    # Equal Collection priority: take the same local budget from each
    # collection and interleave by rank, then fill unused slots globally.
    by_collection: dict[UUID, list[BundleCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_collection[candidate.collection_id].append(candidate)
    for local in by_collection.values():
        local.sort(key=lambda item: (-item.rrf_score, item.bundle_id))
    fair: list[BundleCandidate] = []
    for rank in range(per_collection_limit):
        for collection_id in sorted(by_collection):
            local = by_collection[collection_id]
            if rank < len(local):
                fair.append(local[rank])
    return fair


class KnowledgeCoreDiscoveryService:
    def __init__(self, *, search_port: DiscoverySearchPort, query_embedding_provider: QueryEmbeddingProvider | None = None):
        self._search_port = search_port
        self._query_embedding_provider = query_embedding_provider

    async def discover(
        self, *, collection_ids: Sequence[UUID], query: str,
        query_vectors: dict[UUID, Sequence[float]] | None = None,
        per_channel_limit: int = 20, per_collection_limit: int = 20,
        max_security_level: int = 3,
    ) -> list[BundleCandidate]:
        candidates, _ = await self.discover_with_diagnostics(
            collection_ids=collection_ids,
            query=query,
            query_vectors=query_vectors,
            per_channel_limit=per_channel_limit,
            per_collection_limit=per_collection_limit,
            max_security_level=max_security_level,
        )
        return candidates

    async def discover_with_diagnostics(
        self, *, collection_ids: Sequence[UUID], query: str,
        query_vectors: dict[UUID, Sequence[float]] | None = None,
        per_channel_limit: int = 20, per_collection_limit: int = 20,
        max_security_level: int = 3,
    ) -> tuple[list[BundleCandidate], dict[str, Any]]:
        if not query.strip():
            raise ValueError("query is required")
        warnings: list[str] = []
        if self._query_embedding_provider is not None:
            try:
                query_vectors = (
                    await self._query_embedding_provider.embed_for_collections(
                        query=query,
                        collection_ids=collection_ids,
                    )
                )
            except Exception as exc:
                query_vectors = {}
                warnings.append(
                    "查询向量生成失败，已仅使用全文检索通道"
                )
                logger.exception(
                    "KC Discovery 查询向量生成失败，已降级全文检索 | "
                    "error_type={}",
                    type(exc).__name__,
                )
        hits: list[DiscoveryHit] = []
        collection_reports: list[dict[str, Any]] = []
        successful_channels = 0
        first_failure: Exception | None = None
        for collection_id in sorted(set(collection_ids)):
            text_error = None
            try:
                text_hits = list(await self._search_port.search_text(
                    collection_id=collection_id,
                    query=query,
                    limit=per_channel_limit,
                    max_security_level=max_security_level,
                ))
                successful_channels += 1
            except Exception as exc:
                text_hits = []
                text_error = type(exc).__name__
                first_failure = first_failure or exc
                warnings.append(
                    f"Collection {collection_id} 全文检索失败，"
                    "已继续向量检索"
                )
                logger.exception(
                    "KC Discovery 全文通道失败，已尝试继续 | "
                    "collection_id={} | error_type={}",
                    collection_id,
                    text_error,
                )
            hits.extend(text_hits)
            vector_hits: list[DiscoveryHit] = []
            vector_error = None
            if query_vectors and collection_id in query_vectors:
                try:
                    vector_hits = list(await self._search_port.search_vector(
                        collection_id=collection_id,
                        vector=query_vectors[collection_id],
                        limit=per_channel_limit,
                        max_security_level=max_security_level,
                    ))
                    successful_channels += 1
                except Exception as exc:
                    vector_error = type(exc).__name__
                    first_failure = first_failure or exc
                    warnings.append(
                        f"Collection {collection_id} 向量检索失败，"
                        "已保留全文结果"
                    )
                    logger.exception(
                        "KC Discovery 向量通道失败，已尝试继续 | "
                        "collection_id={} | error_type={}",
                        collection_id,
                        vector_error,
                    )
                hits.extend(vector_hits)
            collection_reports.append(
                {
                    "collection_id": str(collection_id),
                    "text_hits": len(text_hits),
                    "vector_hits": len(vector_hits),
                    "vector_enabled": bool(
                        query_vectors and collection_id in query_vectors
                    ),
                    "text_error": text_error,
                    "vector_error": vector_error,
                }
            )
        if successful_channels == 0 and first_failure is not None:
            raise first_failure
        candidates = aggregate_candidates(
            hits,
            per_collection_limit=per_collection_limit,
        )
        return candidates, {
            "stage": "DISCOVERY",
            "collections": collection_reports,
            "text_hits": sum(
                item["text_hits"] for item in collection_reports
            ),
            "vector_hits": sum(
                item["vector_hits"] for item in collection_reports
            ),
            "raw_hits": len(hits),
            "bundle_candidates": len(candidates),
            "per_channel_limit": per_channel_limit,
            "per_collection_limit": per_collection_limit,
            "warnings": warnings,
        }
