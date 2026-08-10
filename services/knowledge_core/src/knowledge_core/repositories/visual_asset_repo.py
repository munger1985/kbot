"""Knowledge Core 视觉资产持久化。"""

from uuid import UUID

from sqlalchemy import Float, bindparam, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import (
    KcBundleEntity,
    KcBundleRevisionEntity,
    KcParseViewEntity,
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

    async def list_payload_uris_by_view_ids(
        self, parse_view_ids: list[UUID]
    ) -> list[str]:
        if not parse_view_ids:
            return []
        rows = await self.session.scalars(
            select(KcVisualAssetEntity.payload_uri).where(
                KcVisualAssetEntity.parse_view_id.in_(parse_view_ids)
            )
        )
        return list(rows)

    async def list_active_document_page(
        self, *, document_version_id: UUID, asset_type: str | None,
        page_no: int | None, offset: int, limit: int,
    ) -> tuple[list[KcVisualAssetEntity], int]:
        predicates = [
            KcVisualAssetEntity.document_version_id == document_version_id,
            KcVisualAssetEntity.status == "ACTIVE",
            KcParseViewEntity.view_status == "ACTIVE",
        ]
        if asset_type:
            predicates.append(KcVisualAssetEntity.asset_type == asset_type)
        if page_no is not None:
            predicates.append(KcVisualAssetEntity.page_no == page_no)
        base = (
            select(KcVisualAssetEntity)
            .join(
                KcParseViewEntity,
                KcParseViewEntity.parse_view_id
                == KcVisualAssetEntity.parse_view_id,
            )
            .where(*predicates)
        )
        total = int((await self.session.execute(
            select(func.count()).select_from(base.subquery())
        )).scalar_one())
        items = list((await self.session.execute(
            base.order_by(
                KcVisualAssetEntity.page_no.asc().nullslast(),
                KcVisualAssetEntity.asset_type,
                KcVisualAssetEntity.visual_asset_id,
            ).offset(offset).limit(limit)
        )).scalars())
        return items, total

    async def get_active_document_asset(
        self, *, document_version_id: UUID, visual_asset_id: UUID,
    ) -> KcVisualAssetEntity | None:
        statement = (
            select(KcVisualAssetEntity)
            .join(
                KcParseViewEntity,
                KcParseViewEntity.parse_view_id
                == KcVisualAssetEntity.parse_view_id,
            )
            .where(
                KcVisualAssetEntity.visual_asset_id == visual_asset_id,
                KcVisualAssetEntity.document_version_id == document_version_id,
                KcVisualAssetEntity.status == "ACTIVE",
                KcParseViewEntity.view_status == "ACTIVE",
            )
        )
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def search(
        self,
        *,
        collection_id: UUID,
        model_id: UUID,
        query_vector: list[float],
        limit: int,
    ) -> list[tuple[KcVisualAssetEntity, float]]:
        distance = KcVisualAssetEntity.visual_embedding.op(
            "<=>",
            return_type=Float(),
        )(
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
