"""Collection 锁定模型下的视觉资产索引与多图片检索。"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

from knowledge_core.domain.model_bindings import collection_model_id


@dataclass(frozen=True)
class VisualModelSnapshot:
    model_id: UUID
    served_model_name: str
    config_fingerprint: str


class KnowledgeCoreVisualService:
    """视觉向量只由 KC INDEX 阶段生成，查询模型由 Collection 决定。"""

    def __init__(self, *, uow_factory, model_config_client, model_client):
        self._uow_factory = uow_factory
        self._model_config_client = model_config_client
        self._model_client = model_client

    async def index_parse_view(
        self, *, collection_id: UUID, parse_view_id: UUID, batch_size: int
    ) -> int:
        model = await self._resolve_model(collection_id)
        if model is None:
            return 0
        indexed = 0
        while True:
            async with self._uow_factory() as uow:
                rows = await uow.visual_assets.list_needing_index(
                    parse_view_id=parse_view_id,
                    model_id=model.model_id,
                    served_model_name=model.served_model_name,
                    limit=batch_size,
                )
                if not rows:
                    return indexed
                payloads = await asyncio.gather(
                    *(asyncio.to_thread(Path(row.payload_uri).read_bytes) for row in rows)
                )
                vectors = await asyncio.gather(
                    *(
                        self._model_client.get_visual_embedding(
                            base64.b64encode(payload).decode("ascii"),
                            served_model_name=model.served_model_name,
                        )
                        for payload in payloads
                    )
                )
                now = datetime.now(timezone.utc)
                for row, vector in zip(rows, vectors, strict=True):
                    if not vector:
                        raise ValueError("视觉模型返回了空向量")
                    row.visual_embedding = [float(value) for value in vector]
                    row.visual_embedding_model_id = model.model_id
                    row.visual_embedding_served_model_name = (
                        model.served_model_name
                    )
                    row.visual_embedding_config_fingerprint = (
                        model.config_fingerprint
                    )
                    row.indexed_at = now
                await uow.session.flush()
                await uow.commit()
                indexed += len(rows)

    async def search(
        self,
        *,
        collection_ids: list[UUID],
        images_base64: list[str],
        per_image_limit: int,
        result_limit: int,
    ) -> dict[str, Any]:
        """分别编码多张查询图，再按视觉资产做 RRF 融合与文档去重。"""
        if not images_base64 or len(images_base64) > 8:
            raise ValueError("查询图片数量必须在 1 到 8 之间")
        fused: dict[tuple[UUID, UUID, int | None], dict[str, Any]] = {}
        searched_collections: list[str] = []
        skipped_collections: list[str] = []
        for collection_id in collection_ids:
            model = await self._resolve_model(collection_id)
            if model is None:
                skipped_collections.append(str(collection_id))
                continue
            searched_collections.append(str(collection_id))
            vectors = await asyncio.gather(
                *(
                    self._model_client.get_visual_embedding(
                        image,
                        served_model_name=model.served_model_name,
                    )
                    for image in images_base64
                )
            )
            async with self._uow_factory() as uow:
                for image_index, vector in enumerate(vectors):
                    rows = await uow.visual_assets.search(
                        collection_id=collection_id,
                        model_id=model.model_id,
                        query_vector=[float(value) for value in vector],
                        limit=per_image_limit,
                    )
                    for rank, (row, similarity) in enumerate(rows, start=1):
                        revision = await uow.revisions.get_by_id(
                            bundle_revision_id=row.bundle_revision_id
                        )
                        if revision is None:
                            continue
                        key = (
                            row.bundle_revision_id,
                            row.document_id,
                            int(row.page_no) if row.page_no else None,
                        )
                        item = fused.setdefault(
                            key,
                            {
                                "collection_id": str(row.collection_id),
                                "bundle_revision_id": str(
                                    row.bundle_revision_id
                                ),
                                "bundle_id": str(revision.bundle_id),
                                "document_id": str(row.document_id),
                                "document_version_id": str(
                                    row.document_version_id
                                ),
                                "evidence_id": (
                                    str(row.evidence_id)
                                    if row.evidence_id
                                    else None
                                ),
                                "visual_asset_id": str(row.visual_asset_id),
                                "asset_type": row.asset_type,
                                "page_no": (
                                    int(row.page_no) if row.page_no else None
                                ),
                                "payload_uri": row.payload_uri,
                                "description": row.description_text,
                                "score": 0.0,
                                "matched_query_images": [],
                            },
                        )
                        item["score"] += 1.0 / (60 + rank)
                        item["matched_query_images"].append(
                            {
                                "image_index": image_index,
                                "similarity": similarity,
                            }
                        )
        return {
            "results": sorted(
                fused.values(), key=lambda item: item["score"], reverse=True
            )[:result_limit],
            "searched_collection_ids": searched_collections,
            "skipped_collection_ids": skipped_collections,
        }

    async def _resolve_model(
        self, collection_id: UUID
    ) -> VisualModelSnapshot | None:
        async with self._uow_factory() as uow:
            collection = await uow.collections.get_by_id(
                collection_id=collection_id
            )
            if collection is None:
                raise ValueError("Collection 不存在")
            model_id = collection_model_id(
                collection, "visual_embedding"
            )
        if model_id is None:
            return None
        model = await self._model_config_client.get_model(model_id)
        if int(model.get("category") or 0) != 3:
            raise ValueError("Collection 绑定的模型不是视觉 Embedding")
        if model.get("status") != "ACTIVE":
            raise ValueError("Collection 绑定的视觉 Embedding 模型不可用")
        served_name = str(model.get("served_model_name") or "").strip()
        if not served_name:
            raise ValueError("视觉 Embedding 模型缺少 served_model_name")
        fingerprint = sha256(
            (
                str(model_id)
                + "|"
                + served_name
                + "|"
                + repr(model.get("model_params") or {})
            ).encode("utf-8")
        ).hexdigest()
        return VisualModelSnapshot(model_id, served_name, fingerprint)
