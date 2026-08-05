"""Knowledge Core 模型引用反查持久化。"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import (
    KcCollectionEntity,
    KcDiscoveryObjectEntity,
    KcEvidenceEntity,
    KcVisualAssetEntity,
)


class ModelReferenceRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def list_by_model(self, *, model_id: UUID) -> list[dict]:
        collections = list(
            (
                await self._session.execute(
                    select(KcCollectionEntity).order_by(
                        KcCollectionEntity.domain_id,
                        KcCollectionEntity.display_name,
                    )
                )
            ).scalars()
        )
        by_id = {item.collection_id: item for item in collections}
        references: list[dict] = []
        expected = str(model_id)
        for collection in collections:
            for role, value in sorted(
                dict(collection.models_json or {}).items()
            ):
                if str(value) == expected:
                    references.append(
                        self._reference(
                            collection,
                            resource_type="collection",
                            binding_role=role,
                        )
                    )

        indexed_sources = (
            (
                KcEvidenceEntity,
                KcEvidenceEntity.embedding_model_id,
                "evidence_embedding",
            ),
            (
                KcDiscoveryObjectEntity,
                KcDiscoveryObjectEntity.embedding_model_id,
                "discovery_embedding",
            ),
            (
                KcVisualAssetEntity,
                KcVisualAssetEntity.visual_embedding_model_id,
                "visual_embedding",
            ),
        )
        for entity, model_column, role in indexed_sources:
            collection_ids = (
                await self._session.execute(
                    select(entity.collection_id)
                    .where(model_column == model_id)
                    .distinct()
                )
            ).scalars()
            for collection_id in collection_ids:
                collection = by_id.get(collection_id)
                if collection is not None:
                    references.append(
                        self._reference(
                            collection,
                            resource_type="index_profile",
                            binding_role=role,
                        )
                    )
        return references

    @staticmethod
    def _reference(collection, *, resource_type: str, binding_role: str):
        return {
            "domain_id": int(collection.domain_id),
            "resource_type": resource_type,
            "resource_id": collection.collection_id,
            "display_name": collection.display_name,
            "status": collection.status,
            "binding_role": binding_role,
        }
