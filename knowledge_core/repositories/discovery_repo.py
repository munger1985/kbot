from uuid import UUID
from sqlalchemy import Select, bindparam, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import KcBundleEntity, KcBundleRevisionEntity, KcCollectionEntity, KcDiscoveryObjectEntity
from knowledge_core.application.retrieval import DiscoveryHit


class DiscoveryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_key(self, *, bundle_revision_id: UUID, profile_key: str, lock: bool = False) -> KcDiscoveryObjectEntity | None:
        statement: Select = select(KcDiscoveryObjectEntity).where(
            KcDiscoveryObjectEntity.bundle_revision_id == bundle_revision_id,
            KcDiscoveryObjectEntity.profile_key == profile_key,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def add(self, entity: KcDiscoveryObjectEntity) -> KcDiscoveryObjectEntity:
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def list_active(self, *, collection_id: UUID, bundle_revision_id: UUID, object_type: str | None = None) -> list[KcDiscoveryObjectEntity]:
        statement = select(KcDiscoveryObjectEntity).where(
            KcDiscoveryObjectEntity.collection_id == collection_id,
            KcDiscoveryObjectEntity.bundle_revision_id == bundle_revision_id,
            KcDiscoveryObjectEntity.discovery_status == "ACTIVE",
        )
        if object_type:
            statement = statement.where(KcDiscoveryObjectEntity.object_type == object_type)
        return list((await self.session.execute(statement)).scalars())

    async def list_staged(self, *, bundle_revision_id: UUID, limit: int = 500, offset: int = 0) -> list[KcDiscoveryObjectEntity]:
        statement = (
            select(KcDiscoveryObjectEntity)
            .where(
                KcDiscoveryObjectEntity.bundle_revision_id == bundle_revision_id,
                KcDiscoveryObjectEntity.discovery_status == "STAGED",
            )
            .order_by(KcDiscoveryObjectEntity.discovery_object_id)
            .offset(offset)
            .limit(limit)
        )
        return list((await self.session.execute(statement)).scalars())

    async def activate_revision(self, *, bundle_revision_id: UUID) -> None:
        await self.session.execute(update(KcDiscoveryObjectEntity).where(
            KcDiscoveryObjectEntity.bundle_revision_id == bundle_revision_id,
            KcDiscoveryObjectEntity.discovery_status == "STAGED",
        ).values(discovery_status="ACTIVE"))

    async def retire_other_revisions(
        self,
        *,
        bundle_id: UUID,
        except_revision_id: UUID,
    ) -> None:
        # The caller must join Bundle/Revision scope when using this helper;
        # this repository only handles direct object state transitions.
        await self.session.execute(update(KcDiscoveryObjectEntity).where(
            KcDiscoveryObjectEntity.bundle_id == bundle_id,
            KcDiscoveryObjectEntity.bundle_revision_id != except_revision_id,
            KcDiscoveryObjectEntity.discovery_status == "ACTIVE",
        ).values(discovery_status="DELETING"))

    async def search_text(self, *, collection_id: UUID, query: str, limit: int = 20, max_security_level: int = 3) -> list[DiscoveryHit]:
        """Oracle Text candidate search scoped to current published revisions."""
        text_score = func.score(1)
        statement = (
            select(KcDiscoveryObjectEntity, text_score.label("text_score"), KcCollectionEntity.collection_key)
            .join(KcCollectionEntity, KcCollectionEntity.collection_id == KcDiscoveryObjectEntity.collection_id)
            .join(KcBundleEntity, KcBundleEntity.bundle_id == KcDiscoveryObjectEntity.bundle_id)
            .join(KcBundleRevisionEntity, KcBundleRevisionEntity.bundle_revision_id == KcDiscoveryObjectEntity.bundle_revision_id)
            .where(
                KcDiscoveryObjectEntity.collection_id == collection_id,
                KcCollectionEntity.status == "ACTIVE",
                KcDiscoveryObjectEntity.discovery_status == "ACTIVE",
                KcDiscoveryObjectEntity.object_type.in_(("BUNDLE", "DOCUMENT")),
                KcDiscoveryObjectEntity.security_level <= max_security_level,
                KcBundleEntity.current_revision_id == KcDiscoveryObjectEntity.bundle_revision_id,
                KcBundleRevisionEntity.status.in_(("READY", "PARTIAL")),
                func.contains(KcDiscoveryObjectEntity.profile_text, bindparam("discovery_query"), 1) > 0,
            )
            .order_by(text_score.desc(), KcDiscoveryObjectEntity.discovery_object_id)
            .limit(limit)
        ).params(discovery_query=query)
        rows = (await self.session.execute(statement)).all()
        return [self._to_hit(entity, rank, "TEXT", float(score or 0), collection_key) for rank, (entity, score, collection_key) in enumerate(rows, 1)]

    async def search_vector(self, *, collection_id: UUID, vector: list[float], limit: int = 20, max_security_level: int = 3) -> list[DiscoveryHit]:
        """Oracle VECTOR distance search; query vectors are model-grouped upstream."""
        distance = KcDiscoveryObjectEntity.embedding.op("<=>")(bindparam("query_vector"))
        statement = (
            select(KcDiscoveryObjectEntity, distance.label("distance"), KcCollectionEntity.collection_key)
            .join(KcCollectionEntity, KcCollectionEntity.collection_id == KcDiscoveryObjectEntity.collection_id)
            .join(KcBundleEntity, KcBundleEntity.bundle_id == KcDiscoveryObjectEntity.bundle_id)
            .join(KcBundleRevisionEntity, KcBundleRevisionEntity.bundle_revision_id == KcDiscoveryObjectEntity.bundle_revision_id)
            .where(
                KcDiscoveryObjectEntity.collection_id == collection_id,
                KcCollectionEntity.status == "ACTIVE",
                KcDiscoveryObjectEntity.discovery_status == "ACTIVE",
                KcBundleEntity.current_revision_id == KcDiscoveryObjectEntity.bundle_revision_id,
                KcBundleRevisionEntity.status.in_(("READY", "PARTIAL")),
                KcDiscoveryObjectEntity.embedding.is_not(None),
                KcDiscoveryObjectEntity.security_level <= max_security_level,
            )
            .order_by(distance, KcDiscoveryObjectEntity.discovery_object_id)
            .limit(limit)
        ).params(query_vector=vector)
        rows = (await self.session.execute(statement)).all()
        return [self._to_hit(entity, rank, "VECTOR", 1.0 - float(distance or 1.0), collection_key) for rank, (entity, distance, collection_key) in enumerate(rows, 1)]

    @staticmethod
    def _to_hit(entity: KcDiscoveryObjectEntity, rank: int, channel: str, score: float, collection_key: str) -> DiscoveryHit:
        coverage = entity.coverage_json or {}
        return DiscoveryHit(
            collection_id=entity.collection_id,
            collection_key=collection_key,
            bundle_id=entity.bundle_id,
            bundle_revision_id=entity.bundle_revision_id,
            object_type=entity.object_type, profile_key=entity.profile_key,
            display_title=entity.display_title, local_rank=rank, channel=channel, score=score,
            matched_member_key=entity.profile_key if entity.object_type == "DOCUMENT" else None,
            member_count=int(coverage.get("member_count", 0)), coverage=coverage,
            profile_text=entity.profile_text,
        )
