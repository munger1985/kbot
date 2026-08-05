"""Collection 两阶段级联清理持久化。"""

from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import (
    KcBundleEntity,
    KcBundleRevisionDocumentEntity,
    KcBundleRevisionEntity,
    KcCollectionBindingEntity,
    KcCollectionEntity,
    KcDiscoveryObjectEntity,
    KcDocumentEntity,
    KcDocumentVersionEntity,
    KcEvidenceEntity,
    KcIngestionJobEntity,
    KcIngestionReceiptEntity,
    KcParseViewEntity,
    KcRelationEntity,
    KcVisualAssetEntity,
)


class CollectionPurgeRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def purge_descendants(
        self, *, collection_id: UUID, purge_job_id: UUID
    ) -> list[str]:
        """删除数据库子孙事实，但保留 Collection 与可重试 Purge Job。"""
        object_uris: list[str] = []
        for entity, column in (
            (KcDocumentVersionEntity, KcDocumentVersionEntity.storage_uri),
            (KcEvidenceEntity, KcEvidenceEntity.payload_uri),
            (KcVisualAssetEntity, KcVisualAssetEntity.payload_uri),
        ):
            values = (
                await self._session.execute(
                    select(column).where(
                        entity.collection_id == collection_id,
                        column.is_not(None),
                    )
                )
            ).scalars()
            object_uris.extend(str(value) for value in values if value)
        manifests = (
            await self._session.execute(
                select(KcParseViewEntity.artifact_manifest_json).where(
                    KcParseViewEntity.collection_id == collection_id,
                    KcParseViewEntity.artifact_manifest_json.is_not(None),
                )
            )
        ).scalars()
        for manifest in manifests:
            if not isinstance(manifest, dict):
                continue
            for descriptor in manifest.values():
                if isinstance(descriptor, dict) and descriptor.get("uri"):
                    object_uris.append(str(descriptor["uri"]))

        await self._session.execute(
            delete(KcIngestionJobEntity).where(
                KcIngestionJobEntity.collection_id == collection_id,
                KcIngestionJobEntity.ingestion_job_id != purge_job_id,
            )
        )
        for entity in (
            KcIngestionReceiptEntity,
            KcVisualAssetEntity,
            KcEvidenceEntity,
            KcDiscoveryObjectEntity,
            KcRelationEntity,
            KcParseViewEntity,
            KcBundleRevisionDocumentEntity,
            KcDocumentVersionEntity,
            KcDocumentEntity,
        ):
            await self._session.execute(
                delete(entity).where(entity.collection_id == collection_id)
            )
        await self._session.execute(
            update(KcBundleEntity)
            .where(KcBundleEntity.collection_id == collection_id)
            .values(current_revision_id=None)
        )
        for entity in (KcBundleRevisionEntity, KcBundleEntity):
            await self._session.execute(
                delete(entity).where(entity.collection_id == collection_id)
            )
        return list(dict.fromkeys(object_uris))

    async def finalize(
        self, *, collection_id: UUID, purge_job_id: UUID
    ) -> None:
        """仅在全部对象已幂等删除后移除根实体和补偿 Job。"""
        await self._session.execute(
            delete(KcCollectionBindingEntity).where(
                KcCollectionBindingEntity.collection_id == collection_id
            )
        )
        await self._session.execute(
            delete(KcIngestionJobEntity).where(
                KcIngestionJobEntity.ingestion_job_id == purge_job_id,
                KcIngestionJobEntity.collection_id == collection_id,
            )
        )
        await self._session.execute(
            delete(KcCollectionEntity).where(
                KcCollectionEntity.collection_id == collection_id
            )
        )
