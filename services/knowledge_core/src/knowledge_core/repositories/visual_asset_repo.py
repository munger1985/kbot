"""Knowledge Core 视觉资产持久化。"""

from uuid import UUID

from sqlalchemy import bindparam, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import (
    KcBundleEntity,
    KcBundleRevisionEntity,
    KcVisualAssetEntity,
)


class VisualAssetRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, entity: KcVisualAssetEntity) -> KcVisualAssetEntity:
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def get_by_key(
        self, *, parse_view_id: UUID, asset_key: str
    ) -> KcVisualAssetEntity | None:
        statement = select(KcVisualAssetEntity).where(
            KcVisualAssetEntity.parse_view_id == parse_view_id,
            KcVisualAssetEntity.asset_key == asset_key,
        )
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def list_needing_index(
        self,
        *,
        parse_view_id: UUID,
        model_id: UUID,
        served_model_name: str,
        limit: int,
    ) -> list[KcVisualAssetEntity]:
        statement = (
            select(KcVisualAssetEntity)
            .where(
                KcVisualAssetEntity.parse_view_id == parse_view_id,
                KcVisualAssetEntity.status == "ACTIVE",
                (
                    KcVisualAssetEntity.visual_embedding.is_(None)
                    | (
                        KcVisualAssetEntity.visual_embedding_model_id
                        != model_id
                    )
                    | (
                        KcVisualAssetEntity.visual_embedding_served_model_name
                        != served_model_name
                    )
                ),
            )
            .order_by(
                KcVisualAssetEntity.page_no,
                KcVisualAssetEntity.visual_asset_id,
            )
            .limit(limit)
        )
        return list((await self.session.execute(statement)).scalars())

    async def activate_staged(self, *, parse_view_id: UUID) -> None:
        await self.session.execute(
            update(KcVisualAssetEntity)
            .where(
                KcVisualAssetEntity.parse_view_id == parse_view_id,
                KcVisualAssetEntity.status == "STAGED",
            )
            .values(status="ACTIVE")
        )

    async def delete_by_view_ids(self, parse_view_ids: list[UUID]) -> None:
        if parse_view_ids:
            await self.session.execute(
                delete(KcVisualAssetEntity).where(
                    KcVisualAssetEntity.parse_view_id.in_(parse_view_ids)
                )
            )

    async def search(
        self,
        *,
        collection_id: UUID,
        model_id: UUID,
        query_vector: list[float],
        limit: int,
    ) -> list[tuple[KcVisualAssetEntity, float]]:
        distance = KcVisualAssetEntity.visual_embedding.op("<=>")(
            bindparam("query_vector")
        )
        statement = (
            select(KcVisualAssetEntity, (1 - distance).label("similarity"))
            .join(
                KcBundleRevisionEntity,
                KcBundleRevisionEntity.bundle_revision_id
                == KcVisualAssetEntity.bundle_revision_id,
            )
            .join(
                KcBundleEntity,
                KcBundleEntity.bundle_id
                == KcBundleRevisionEntity.bundle_id,
            )
            .where(
                KcVisualAssetEntity.collection_id == collection_id,
                KcVisualAssetEntity.status == "ACTIVE",
                KcVisualAssetEntity.visual_embedding_model_id == model_id,
                KcVisualAssetEntity.visual_embedding.is_not(None),
                KcBundleEntity.current_revision_id
                == KcVisualAssetEntity.bundle_revision_id,
            )
            .order_by(distance)
            .limit(limit)
        )
        rows = await self.session.execute(
            statement, {"query_vector": query_vector}
        )
        return [(row[0], float(row[1])) for row in rows]
