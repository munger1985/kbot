"""Persistence-only repositories for KC immutable ingestion aggregates."""
from datetime import datetime

from sqlalchemy import Select, and_, bindparam, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import KcBundleEntity, KcBundleRevisionDocumentEntity, KcBundleRevisionEntity, KcCollectionEntity, KcDocumentEntity, KcDocumentVersionEntity, KcEvidenceEntity, KcIngestionJobEntity, KcParseViewEntity
from knowledge_core.application.evidence_retrieval import EvidenceHit, EvidenceScope


class BundleRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_source(self, *, collection_id: int, source_system: str, source_type: str, source_id: str, lock: bool = False) -> KcBundleEntity | None:
        statement: Select = select(KcBundleEntity).where(KcBundleEntity.collection_id == collection_id, KcBundleEntity.source_system == source_system, KcBundleEntity.source_type == source_type, KcBundleEntity.source_id == source_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def add(self, entity: KcBundleEntity) -> KcBundleEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def get_by_id(self, *, bundle_id: int, lock: bool = False) -> KcBundleEntity | None:
        statement: Select = select(KcBundleEntity).where(KcBundleEntity.bundle_id == bundle_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()


class BundleRevisionRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_source_revision(self, *, bundle_id: int, source_revision: str) -> KcBundleRevisionEntity | None:
        statement: Select = select(KcBundleRevisionEntity).where(KcBundleRevisionEntity.bundle_id == bundle_id, KcBundleRevisionEntity.source_revision == source_revision)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def next_revision_no(self, *, bundle_id: int) -> int:
        statement = select(func.coalesce(func.max(KcBundleRevisionEntity.revision_no), 0)).where(KcBundleRevisionEntity.bundle_id == bundle_id)
        return int((await self.session.execute(statement)).scalar_one()) + 1
    async def add(self, entity: KcBundleRevisionEntity) -> KcBundleRevisionEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def get_by_id(self, *, bundle_revision_id: int, lock: bool = False) -> KcBundleRevisionEntity | None:
        statement: Select = select(KcBundleRevisionEntity).where(KcBundleRevisionEntity.bundle_revision_id == bundle_revision_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def list_by_bundle(self, *, bundle_id: int) -> list[KcBundleRevisionEntity]:
        statement: Select = select(KcBundleRevisionEntity).where(KcBundleRevisionEntity.bundle_id == bundle_id).order_by(KcBundleRevisionEntity.revision_no.desc())
        return list((await self.session.execute(statement)).scalars())


class DocumentRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_external_id(self, *, bundle_id: int, external_document_id: str) -> KcDocumentEntity | None:
        statement: Select = select(KcDocumentEntity).where(KcDocumentEntity.bundle_id == bundle_id, KcDocumentEntity.external_document_id == external_document_id)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def add(self, entity: KcDocumentEntity) -> KcDocumentEntity:
        self.session.add(entity); await self.session.flush(); return entity


class DocumentVersionRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_content_hash(self, *, document_id: int, content_hash: str) -> KcDocumentVersionEntity | None:
        statement: Select = select(KcDocumentVersionEntity).where(KcDocumentVersionEntity.document_id == document_id, KcDocumentVersionEntity.content_hash == content_hash)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def get_by_id(self, *, document_version_id: int) -> KcDocumentVersionEntity | None:
        statement: Select = select(KcDocumentVersionEntity).where(KcDocumentVersionEntity.document_version_id == document_version_id)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def next_version_no(self, *, document_id: int) -> int:
        statement = select(func.coalesce(func.max(KcDocumentVersionEntity.version_no), 0)).where(KcDocumentVersionEntity.document_id == document_id)
        return int((await self.session.execute(statement)).scalar_one()) + 1
    async def add(self, entity: KcDocumentVersionEntity) -> KcDocumentVersionEntity:
        self.session.add(entity); await self.session.flush(); return entity


class BundleRevisionDocumentRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def add(self, entity: KcBundleRevisionDocumentEntity) -> KcBundleRevisionDocumentEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def get_by_version(self, *, bundle_revision_id: int, document_version_id: int, lock: bool = False) -> KcBundleRevisionDocumentEntity | None:
        statement: Select = select(KcBundleRevisionDocumentEntity).where(KcBundleRevisionDocumentEntity.bundle_revision_id == bundle_revision_id, KcBundleRevisionDocumentEntity.document_version_id == document_version_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def list_by_revision(self, *, bundle_revision_id: int) -> list[KcBundleRevisionDocumentEntity]:
        statement: Select = select(KcBundleRevisionDocumentEntity).where(KcBundleRevisionDocumentEntity.bundle_revision_id == bundle_revision_id)
        return list((await self.session.execute(statement)).scalars())


class IngestionJobRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def claim_candidates(self, *, now: datetime, limit: int) -> list[KcIngestionJobEntity]:
        statement: Select = (
            select(KcIngestionJobEntity)
            .where(KcIngestionJobEntity.job_type == "PARSE", KcIngestionJobEntity.job_status.in_(("PENDING", "RETRY_WAIT")), KcIngestionJobEntity.available_at <= now)
            .order_by(KcIngestionJobEntity.priority.desc(), KcIngestionJobEntity.available_at, KcIngestionJobEntity.ingestion_job_id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return list((await self.session.execute(statement)).scalars())
    async def claim_index_candidates(self, *, now: datetime, limit: int) -> list[KcIngestionJobEntity]:
        """Claim only INDEX jobs; PARSE workers must never consume them."""
        statement: Select = (
            select(KcIngestionJobEntity)
            .where(
                KcIngestionJobEntity.job_type == "INDEX",
                KcIngestionJobEntity.job_status.in_(("PENDING", "RETRY_WAIT")),
                KcIngestionJobEntity.available_at <= now,
            )
            .order_by(
                KcIngestionJobEntity.priority.desc(),
                KcIngestionJobEntity.available_at,
                KcIngestionJobEntity.ingestion_job_id,
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return list((await self.session.execute(statement)).scalars())
    async def claim_profile_candidates(self, *, now: datetime, limit: int) -> list[KcIngestionJobEntity]:
        statement: Select = (
            select(KcIngestionJobEntity)
            .where(
                KcIngestionJobEntity.job_type == "PROFILE",
                KcIngestionJobEntity.job_status.in_(("PENDING", "RETRY_WAIT")),
                KcIngestionJobEntity.available_at <= now,
            )
            .order_by(
                KcIngestionJobEntity.priority.desc(),
                KcIngestionJobEntity.available_at,
                KcIngestionJobEntity.ingestion_job_id,
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return list((await self.session.execute(statement)).scalars())
    async def claim_purge_candidates(self, *, now: datetime, limit: int) -> list[KcIngestionJobEntity]:
        statement: Select = (
            select(KcIngestionJobEntity)
            .where(
                KcIngestionJobEntity.job_type == "COLLECTION_PURGE",
                KcIngestionJobEntity.job_status.in_(("PENDING", "RETRY_WAIT")),
                KcIngestionJobEntity.available_at <= now,
            )
            .order_by(KcIngestionJobEntity.priority.desc(), KcIngestionJobEntity.available_at)
            .limit(limit).with_for_update(skip_locked=True)
        )
        return list((await self.session.execute(statement)).scalars())
    async def get_by_id(self, *, ingestion_job_id: int, lock: bool = False) -> KcIngestionJobEntity | None:
        statement: Select = select(KcIngestionJobEntity).where(KcIngestionJobEntity.ingestion_job_id == ingestion_job_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def get_by_idempotency_key(
        self, *, collection_id: int, idempotency_key: str, input_fingerprint: str,
    ) -> KcIngestionJobEntity | None:
        statement: Select = select(KcIngestionJobEntity).where(
            KcIngestionJobEntity.collection_id == collection_id,
            KcIngestionJobEntity.idempotency_key == idempotency_key,
            KcIngestionJobEntity.input_fingerprint == input_fingerprint,
        )
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def add(self, entity: KcIngestionJobEntity) -> KcIngestionJobEntity:
        self.session.add(entity); await self.session.flush(); return entity


class ParseViewRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_input(self, *, document_version_id: int, view_kind: str, parse_config_fingerprint: str) -> KcParseViewEntity | None:
        statement: Select = select(KcParseViewEntity).where(KcParseViewEntity.document_version_id == document_version_id, KcParseViewEntity.view_kind == view_kind, KcParseViewEntity.parse_config_fingerprint == parse_config_fingerprint)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def add(self, entity: KcParseViewEntity) -> KcParseViewEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def get_by_id(self, *, parse_view_id: int, lock: bool = False) -> KcParseViewEntity | None:
        statement: Select = select(KcParseViewEntity).where(KcParseViewEntity.parse_view_id == parse_view_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def list_active_others(
        self, *, document_version_id: int, view_kind: str, except_parse_view_id: int,
    ) -> list[KcParseViewEntity]:
        statement = select(KcParseViewEntity).where(
            KcParseViewEntity.document_version_id == document_version_id,
            KcParseViewEntity.view_kind == view_kind,
            KcParseViewEntity.view_status == "ACTIVE",
            KcParseViewEntity.parse_view_id != except_parse_view_id,
        ).with_for_update()
        return list((await self.session.execute(statement)).scalars())
    async def delete_by_ids(self, parse_view_ids: list[int]) -> None:
        if parse_view_ids:
            await self.session.execute(delete(KcParseViewEntity).where(KcParseViewEntity.parse_view_id.in_(parse_view_ids)))


class EvidenceRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_key(self, *, parse_view_id: int, evidence_key: str) -> KcEvidenceEntity | None:
        statement: Select = select(KcEvidenceEntity).where(KcEvidenceEntity.parse_view_id == parse_view_id, KcEvidenceEntity.evidence_key == evidence_key)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def add(self, entity: KcEvidenceEntity) -> KcEvidenceEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def count_staged(self, *, parse_view_id: int) -> int:
        statement = select(func.count()).select_from(KcEvidenceEntity).where(KcEvidenceEntity.parse_view_id == parse_view_id, KcEvidenceEntity.status == "STAGED")
        return int((await self.session.execute(statement)).scalar_one())
    async def list_staged_keys(self, *, parse_view_id: int) -> list[str]:
        statement = select(KcEvidenceEntity.evidence_key).where(
            KcEvidenceEntity.parse_view_id == parse_view_id,
            KcEvidenceEntity.status == "STAGED",
        ).order_by(KcEvidenceEntity.ordinal)
        return list((await self.session.execute(statement)).scalars())
    async def activate_staged(self, *, parse_view_id: int) -> None:
        await self.session.execute(update(KcEvidenceEntity).where(KcEvidenceEntity.parse_view_id == parse_view_id, KcEvidenceEntity.status == "STAGED").values(status="ACTIVE"))
    async def delete_by_view_ids(self, parse_view_ids: list[int]) -> None:
        if parse_view_ids:
            await self.session.execute(delete(KcEvidenceEntity).where(KcEvidenceEntity.parse_view_id.in_(parse_view_ids)))

    async def list_needing_index(
        self, *, parse_view_id: int, model_id: int, model_key: str, limit: int = 500,
    ) -> list[KcEvidenceEntity]:
        statement = (
            select(KcEvidenceEntity)
            .where(
                KcEvidenceEntity.parse_view_id == parse_view_id,
                KcEvidenceEntity.status == "ACTIVE",
                (KcEvidenceEntity.embedding_input_hash.is_(None)
                 | (KcEvidenceEntity.embedding_model_id != model_id)
                 | (KcEvidenceEntity.embedding_model_key != model_key)),
            )
            .order_by(KcEvidenceEntity.ordinal, KcEvidenceEntity.fragment_index)
            .limit(limit)
        )
        return list((await self.session.execute(statement)).scalars())

    async def list_active(self, *, parse_view_id: int, limit: int = 500, offset: int = 0) -> list[KcEvidenceEntity]:
        statement = (
            select(KcEvidenceEntity)
            .where(
                KcEvidenceEntity.parse_view_id == parse_view_id,
                KcEvidenceEntity.status == "ACTIVE",
            )
            .order_by(KcEvidenceEntity.ordinal, KcEvidenceEntity.fragment_index)
            .offset(offset)
            .limit(limit)
        )
        return list((await self.session.execute(statement)).scalars())

    async def count_active_for_version(self, *, document_version_id: int) -> int:
        statement = select(func.count()).select_from(KcEvidenceEntity).where(
            KcEvidenceEntity.document_version_id == document_version_id,
            KcEvidenceEntity.status == "ACTIVE",
            KcEvidenceEntity.embedding_input_hash.is_not(None),
        )
        return int((await self.session.execute(statement)).scalar_one())

    async def list_section_titles(self, *, document_version_id: int, limit: int = 20) -> list[str]:
        statement = (
            select(KcEvidenceEntity.section_key)
            .where(
                KcEvidenceEntity.document_version_id == document_version_id,
                KcEvidenceEntity.status == "ACTIVE",
                KcEvidenceEntity.section_key.is_not(None),
            )
            .distinct()
            .limit(limit)
        )
        return [str(value) for value in (await self.session.execute(statement)).scalars() if value]

    async def search_text(self, *, scope: EvidenceScope, query: str, limit: int = 20, max_security_level: int = 3) -> list[EvidenceHit]:
        score = func.score(1)
        statement = (
            select(KcEvidenceEntity, score.label("text_score"))
            .join(KcCollectionEntity, KcCollectionEntity.collection_id == KcEvidenceEntity.collection_id)
            .join(KcBundleEntity, KcBundleEntity.bundle_id == KcEvidenceEntity.bundle_id)
            .join(KcBundleRevisionEntity, KcBundleRevisionEntity.bundle_revision_id == KcEvidenceEntity.bundle_revision_id)
            .join(KcDocumentVersionEntity, KcDocumentVersionEntity.document_version_id == KcEvidenceEntity.document_version_id)
            .where(
                KcEvidenceEntity.collection_id == scope.collection_id,
                KcCollectionEntity.status == "ACTIVE",
                KcEvidenceEntity.bundle_id == scope.bundle_id,
                KcEvidenceEntity.bundle_revision_id == scope.bundle_revision_id,
                KcEvidenceEntity.status == "ACTIVE",
                KcEvidenceEntity.embedding_input_hash.is_not(None),
                KcDocumentVersionEntity.security_level <= max_security_level,
                KcBundleRevisionEntity.security_level <= max_security_level,
                KcBundleEntity.current_revision_id == scope.bundle_revision_id,
                KcBundleRevisionEntity.status.in_(("READY", "PARTIAL")),
                func.contains(KcEvidenceEntity.retrieval_text, bindparam("evidence_query"), 1) > 0,
            )
            .order_by(score.desc(), KcEvidenceEntity.evidence_id)
            .limit(limit)
        )
        if scope.document_version_ids:
            statement = statement.where(KcEvidenceEntity.document_version_id.in_(scope.document_version_ids))
        rows = (await self.session.execute(statement.params(evidence_query=query))).all()
        return [self._to_hit(entity, scope.bundle_id, rank, "TEXT", float(hit_score or 0)) for rank, (entity, hit_score) in enumerate(rows, 1)]

    async def search_vector(self, *, scope: EvidenceScope, vector: list[float], limit: int = 20, max_security_level: int = 3) -> list[EvidenceHit]:
        distance = KcEvidenceEntity.embedding.op("<=>")(bindparam("evidence_vector"))
        statement = (
            select(KcEvidenceEntity, distance.label("distance"))
            .join(KcCollectionEntity, KcCollectionEntity.collection_id == KcEvidenceEntity.collection_id)
            .join(KcBundleEntity, KcBundleEntity.bundle_id == KcEvidenceEntity.bundle_id)
            .join(KcBundleRevisionEntity, KcBundleRevisionEntity.bundle_revision_id == KcEvidenceEntity.bundle_revision_id)
            .join(KcDocumentVersionEntity, KcDocumentVersionEntity.document_version_id == KcEvidenceEntity.document_version_id)
            .where(
                KcEvidenceEntity.collection_id == scope.collection_id,
                KcCollectionEntity.status == "ACTIVE",
                KcEvidenceEntity.bundle_id == scope.bundle_id,
                KcEvidenceEntity.bundle_revision_id == scope.bundle_revision_id,
                KcEvidenceEntity.status == "ACTIVE",
                KcEvidenceEntity.embedding.is_not(None),
                KcDocumentVersionEntity.security_level <= max_security_level,
                KcBundleRevisionEntity.security_level <= max_security_level,
                KcBundleEntity.current_revision_id == scope.bundle_revision_id,
                KcBundleRevisionEntity.status.in_(("READY", "PARTIAL")),
            )
            .order_by(distance, KcEvidenceEntity.evidence_id)
            .limit(limit)
        )
        if scope.document_version_ids:
            statement = statement.where(KcEvidenceEntity.document_version_id.in_(scope.document_version_ids))
        rows = (await self.session.execute(statement.params(evidence_vector=vector))).all()
        return [self._to_hit(entity, scope.bundle_id, rank, "VECTOR", 1.0 - float(distance_value or 1.0)) for rank, (entity, distance_value) in enumerate(rows, 1)]

    async def expand_context(self, *, anchors: list[EvidenceHit], limit: int = 4) -> list[EvidenceHit]:
        if not anchors:
            return []
        conditions = [and_(
            KcEvidenceEntity.document_version_id == anchor.document_version_id,
            KcEvidenceEntity.parse_view_id == anchor.parse_view_id,
            KcEvidenceEntity.section_key == anchor.section_key,
        ) for anchor in anchors if anchor.section_key]
        if not conditions:
            return []
        collection_ids = {anchor.collection_id for anchor in anchors}
        statement = select(KcEvidenceEntity).join(
            KcCollectionEntity, KcCollectionEntity.collection_id == KcEvidenceEntity.collection_id,
        ).where(
            KcEvidenceEntity.status == "ACTIVE",
            KcCollectionEntity.status == "ACTIVE",
            KcEvidenceEntity.collection_id.in_(collection_ids),
            or_(*conditions),
        ).order_by(KcEvidenceEntity.ordinal).limit(limit)
        rows = (await self.session.execute(statement)).scalars()
        anchor_ids = {item.evidence_id for item in anchors}
        bundle_id = anchors[0].bundle_id
        return [self._to_hit(entity, bundle_id, index, "CONTEXT", 0.0) for index, entity in enumerate(rows, 1) if entity.evidence_id not in anchor_ids]

    @staticmethod
    def _to_hit(entity: KcEvidenceEntity, bundle_id: int, rank: int, channel: str, score: float) -> EvidenceHit:
        return EvidenceHit(
            evidence_id=int(entity.evidence_id), collection_id=int(entity.collection_id),
            bundle_id=bundle_id,
            bundle_revision_id=int(entity.bundle_revision_id),
            bundle_revision_document_id=int(entity.bundle_revision_document_id) if entity.bundle_revision_document_id else None,
            document_id=int(entity.document_id), document_version_id=int(entity.document_version_id),
            parse_view_id=int(entity.parse_view_id), evidence_key=entity.evidence_key,
            evidence_type=entity.evidence_type, content_text=entity.content_text,
            retrieval_text=entity.retrieval_text, heading_path=tuple(entity.heading_path_json or []),
            locator=entity.locator_json, source_spans=tuple(entity.source_spans_json or []),
            provenance=entity.provenance_json, section_key=entity.section_key,
            parent_evidence_key=entity.parent_evidence_key, ordinal=int(entity.ordinal),
            quality_score=float(entity.quality_score) if entity.quality_score is not None else None,
            local_rank=rank, channel=channel,
        )
