from uuid import UUID
from sqlalchemy import (
    Float,
    Select,
    bindparam,
    func,
    literal_column,
    delete,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import (
    KcBundleEntity,
    KcBundleRevisionDocumentEntity,
    KcBundleRevisionEntity,
    KcCollectionEntity,
    KcDiscoveryObjectEntity,
    KcDocumentVersionEntity,
    KcEvidenceEntity,
)
from knowledge_core.application.retrieval import DiscoveryHit
from knowledge_core.repositories.oracle_text_query import (
    build_oracle_text_query,
)


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

    async def delete_by_revision(self, *, bundle_revision_id: UUID) -> None:
        await self.session.execute(
            delete(KcDiscoveryObjectEntity).where(
                KcDiscoveryObjectEntity.bundle_revision_id
                == bundle_revision_id
            )
        )

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
        oracle_query = build_oracle_text_query(query)
        if not oracle_query:
            return []
        oracle_text_label = literal_column("1")
        text_score = func.score(oracle_text_label)
        statement = (
            select(KcDiscoveryObjectEntity, text_score.label("text_score"))
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
                func.contains(
                    KcDiscoveryObjectEntity.profile_text,
                    bindparam("discovery_query"),
                    oracle_text_label,
                )
                > 0,
            )
            .order_by(text_score.desc(), KcDiscoveryObjectEntity.discovery_object_id)
            .limit(limit)
        ).params(discovery_query=oracle_query)
        rows = (await self.session.execute(statement)).all()
        if rows:
            return [self._to_hit(entity, rank, "TEXT", float(score or 0)) for rank, (entity, score) in enumerate(rows, 1)]
        return await self._search_evidence_text(
            collection_id=collection_id,
            oracle_query=oracle_query,
            limit=limit,
            max_security_level=max_security_level,
        )

    async def _search_evidence_text(
        self,
        *,
        collection_id: UUID,
        oracle_query: str,
        limit: int,
        max_security_level: int,
    ) -> list[DiscoveryHit]:
        """Profile 未命中时，以 Evidence 全文索引桥接到所属 Bundle。"""
        oracle_text_label = literal_column("1")
        text_score = func.score(oracle_text_label)
        statement = (
            select(
                KcEvidenceEntity,
                KcBundleRevisionEntity.bundle_id,
                KcBundleRevisionEntity.title,
                KcBundleRevisionDocumentEntity.external_document_id,
                KcBundleRevisionDocumentEntity.declared_name,
                text_score.label("text_score"),
            )
            .join(
                KcCollectionEntity,
                KcCollectionEntity.collection_id
                == KcEvidenceEntity.collection_id,
            )
            .join(
                KcBundleRevisionEntity,
                KcBundleRevisionEntity.bundle_revision_id
                == KcEvidenceEntity.bundle_revision_id,
            )
            .join(
                KcBundleEntity,
                KcBundleEntity.bundle_id
                == KcBundleRevisionEntity.bundle_id,
            )
            .join(
                KcDocumentVersionEntity,
                KcDocumentVersionEntity.document_version_id
                == KcEvidenceEntity.document_version_id,
            )
            .outerjoin(
                KcBundleRevisionDocumentEntity,
                KcBundleRevisionDocumentEntity.bundle_revision_document_id
                == KcEvidenceEntity.bundle_revision_document_id,
            )
            .where(
                KcEvidenceEntity.collection_id == collection_id,
                KcCollectionEntity.status == "ACTIVE",
                KcEvidenceEntity.status == "ACTIVE",
                KcEvidenceEntity.embedding_input_hash.is_not(None),
                KcDocumentVersionEntity.security_level
                <= max_security_level,
                KcBundleRevisionEntity.security_level
                <= max_security_level,
                KcBundleEntity.current_revision_id
                == KcEvidenceEntity.bundle_revision_id,
                KcBundleRevisionEntity.status.in_(("READY", "PARTIAL")),
                func.contains(
                    KcEvidenceEntity.retrieval_text,
                    bindparam("evidence_discovery_query"),
                    oracle_text_label,
                )
                > 0,
            )
            .order_by(text_score.desc(), KcEvidenceEntity.evidence_id)
            .limit(limit)
        ).params(evidence_discovery_query=oracle_query)
        rows = (await self.session.execute(statement)).all()
        hits: list[DiscoveryHit] = []
        seen_documents: set[UUID] = set()
        for (
            entity,
            bundle_id,
            bundle_title,
            external_document_id,
            declared_name,
            score,
        ) in rows:
            if entity.document_version_id in seen_documents:
                continue
            seen_documents.add(entity.document_version_id)
            member_key = (
                external_document_id or str(entity.document_version_id)
            )
            hits.append(
                DiscoveryHit(
                    collection_id=entity.collection_id,
                    bundle_id=bundle_id,
                    bundle_revision_id=entity.bundle_revision_id,
                    object_type="DOCUMENT",
                    profile_key=f"evidence:{member_key}",
                    display_title=declared_name or bundle_title,
                    local_rank=len(hits) + 1,
                    channel="TEXT_EVIDENCE",
                    score=float(score or 0),
                    matched_member_key=member_key,
                    member_count=1,
                    coverage={"evidence_bridge": True},
                    profile_text=entity.retrieval_text[:12000],
                )
            )
        return hits

    async def search_vector(self, *, collection_id: UUID, vector: list[float], limit: int = 20, max_security_level: int = 3) -> list[DiscoveryHit]:
        """Oracle VECTOR distance search; query vectors are model-grouped upstream."""
        distance = KcDiscoveryObjectEntity.embedding.op(
            "<=>",
            return_type=Float(),
        )(bindparam("query_vector"))
        statement = (
            select(KcDiscoveryObjectEntity, distance.label("distance"))
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
        return [
            self._to_hit(
                entity,
                rank,
                "VECTOR",
                1.0 - float(distance if distance is not None else 1.0),
            )
            for rank, (entity, distance)
            in enumerate(rows, 1)
        ]

    @staticmethod
    def _to_hit(entity: KcDiscoveryObjectEntity, rank: int, channel: str, score: float) -> DiscoveryHit:
        coverage = entity.coverage_json or {}
        return DiscoveryHit(
            collection_id=entity.collection_id,
            bundle_id=entity.bundle_id,
            bundle_revision_id=entity.bundle_revision_id,
            object_type=entity.object_type, profile_key=entity.profile_key,
            display_title=entity.display_title, local_rank=rank, channel=channel, score=score,
            matched_member_key=entity.profile_key if entity.object_type == "DOCUMENT" else None,
            member_count=int(coverage.get("member_count", 0)), coverage=coverage,
            profile_text=entity.profile_text,
        )
