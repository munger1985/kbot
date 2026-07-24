"""Persistence-only repositories for KC immutable ingestion aggregates."""
from uuid import UUID
from datetime import datetime

from sqlalchemy import Select, and_, bindparam, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import KcBundleEntity, KcBundleRevisionDocumentEntity, KcBundleRevisionEntity, KcCollectionEntity, KcDocumentEntity, KcDocumentVersionEntity, KcEvidenceEntity, KcIngestionJobEntity, KcParseViewEntity
from knowledge_core.application.evidence_retrieval import EvidenceHit, EvidenceScope


class BundleRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_source(self, *, collection_id: UUID, source_system: str, source_type: str, source_id: str, lock: bool = False) -> KcBundleEntity | None:
        statement: Select = select(KcBundleEntity).where(KcBundleEntity.collection_id == collection_id, KcBundleEntity.source_system == source_system, KcBundleEntity.source_type == source_type, KcBundleEntity.source_id == source_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def add(self, entity: KcBundleEntity) -> KcBundleEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def get_by_id(self, *, bundle_id: UUID, lock: bool = False) -> KcBundleEntity | None:
        statement: Select = select(KcBundleEntity).where(KcBundleEntity.bundle_id == bundle_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()


class BundleRevisionRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_source_revision(self, *, bundle_id: UUID, source_revision: str) -> KcBundleRevisionEntity | None:
        statement: Select = select(KcBundleRevisionEntity).where(KcBundleRevisionEntity.bundle_id == bundle_id, KcBundleRevisionEntity.source_revision == source_revision)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def next_revision_no(self, *, bundle_id: UUID) -> int:
        statement = select(func.coalesce(func.max(KcBundleRevisionEntity.revision_no), 0)).where(KcBundleRevisionEntity.bundle_id == bundle_id)
        return int((await self.session.execute(statement)).scalar_one()) + 1
    async def add(self, entity: KcBundleRevisionEntity) -> KcBundleRevisionEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def get_by_id(self, *, bundle_revision_id: UUID, lock: bool = False) -> KcBundleRevisionEntity | None:
        statement: Select = select(KcBundleRevisionEntity).where(KcBundleRevisionEntity.bundle_revision_id == bundle_revision_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def list_by_bundle(self, *, bundle_id: UUID) -> list[KcBundleRevisionEntity]:
        statement: Select = select(KcBundleRevisionEntity).where(KcBundleRevisionEntity.bundle_id == bundle_id).order_by(KcBundleRevisionEntity.revision_no.desc())
        return list((await self.session.execute(statement)).scalars())
    async def list_by_approval(
        self, *, collection_id: UUID, approval_status: str,
    ) -> list[KcBundleRevisionEntity]:
        statement: Select = (
            select(KcBundleRevisionEntity)
            .where(
                KcBundleRevisionEntity.collection_id == collection_id,
                KcBundleRevisionEntity.approval_status == approval_status,
            )
            .order_by(
                KcBundleRevisionEntity.accepted_at,
                KcBundleRevisionEntity.bundle_revision_id,
            )
        )
        return list((await self.session.execute(statement)).scalars())


class DocumentRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_external_id(self, *, bundle_id: UUID, external_document_id: str) -> KcDocumentEntity | None:
        statement: Select = select(KcDocumentEntity).where(KcDocumentEntity.bundle_id == bundle_id, KcDocumentEntity.external_document_id == external_document_id)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def add(self, entity: KcDocumentEntity) -> KcDocumentEntity:
        self.session.add(entity); await self.session.flush(); return entity


class DocumentVersionRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_content_hash(self, *, document_id: UUID, content_hash: str) -> KcDocumentVersionEntity | None:
        statement: Select = select(KcDocumentVersionEntity).where(KcDocumentVersionEntity.document_id == document_id, KcDocumentVersionEntity.content_hash == content_hash)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def get_by_id(self, *, document_version_id: UUID) -> KcDocumentVersionEntity | None:
        statement: Select = select(KcDocumentVersionEntity).where(KcDocumentVersionEntity.document_version_id == document_version_id)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def next_version_no(self, *, document_id: UUID) -> int:
        statement = select(func.coalesce(func.max(KcDocumentVersionEntity.version_no), 0)).where(KcDocumentVersionEntity.document_id == document_id)
        return int((await self.session.execute(statement)).scalar_one()) + 1
    async def add(self, entity: KcDocumentVersionEntity) -> KcDocumentVersionEntity:
        self.session.add(entity); await self.session.flush(); return entity


class BundleRevisionDocumentRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def add(self, entity: KcBundleRevisionDocumentEntity) -> KcBundleRevisionDocumentEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def get_by_version(self, *, bundle_revision_id: UUID, document_version_id: UUID, lock: bool = False) -> KcBundleRevisionDocumentEntity | None:
        statement: Select = select(KcBundleRevisionDocumentEntity).where(KcBundleRevisionDocumentEntity.bundle_revision_id == bundle_revision_id, KcBundleRevisionDocumentEntity.document_version_id == document_version_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def list_by_revision(self, *, bundle_revision_id: UUID) -> list[KcBundleRevisionDocumentEntity]:
        statement: Select = select(KcBundleRevisionDocumentEntity).where(KcBundleRevisionDocumentEntity.bundle_revision_id == bundle_revision_id)
        return list((await self.session.execute(statement)).scalars())


class IngestionJobRepository:
    _MIN_CLAIM_SCAN_LIMIT = 16
    _CLAIM_SCAN_FACTOR = 4

    def __init__(self, session: AsyncSession): self.session = session

    async def _claim_by_type(
        self,
        *,
        job_type: str,
        now: datetime,
        limit: int,
    ) -> list[KcIngestionJobEntity]:
        """先筛选候选主键，再逐行加锁，规避 Oracle ORA-02014。"""
        if limit <= 0:
            return []

        eligibility = (
            KcIngestionJobEntity.job_type == job_type,
            KcIngestionJobEntity.job_status.in_(("PENDING", "RETRY_WAIT")),
            KcIngestionJobEntity.available_at <= now,
        )
        scan_limit = max(
            self._MIN_CLAIM_SCAN_LIMIT,
            limit * self._CLAIM_SCAN_FACTOR,
        )
        candidate_statement: Select = (
            select(KcIngestionJobEntity.ingestion_job_id)
            .where(*eligibility)
            .order_by(
                KcIngestionJobEntity.priority.desc(),
                KcIngestionJobEntity.available_at,
                KcIngestionJobEntity.ingestion_job_id,
            )
            .limit(scan_limit)
        )
        candidate_ids = list(
            (await self.session.execute(candidate_statement)).scalars()
        )

        claimed: list[KcIngestionJobEntity] = []
        for ingestion_job_id in candidate_ids:
            lock_statement: Select = (
                select(KcIngestionJobEntity)
                .where(
                    KcIngestionJobEntity.ingestion_job_id == ingestion_job_id,
                    *eligibility,
                )
                .with_for_update(skip_locked=True)
            )
            job = (
                await self.session.execute(lock_statement)
            ).scalar_one_or_none()
            if job is None:
                continue
            claimed.append(job)
            if len(claimed) >= limit:
                break
        return claimed

    async def claim_candidates(self, *, now: datetime, limit: int) -> list[KcIngestionJobEntity]:
        return await self._claim_by_type(
            job_type="PARSE",
            now=now,
            limit=limit,
        )

    async def claim_index_candidates(self, *, now: datetime, limit: int) -> list[KcIngestionJobEntity]:
        """仅抢占 INDEX 任务，PARSE Worker 不得消费此类任务。"""
        return await self._claim_by_type(
            job_type="INDEX",
            now=now,
            limit=limit,
        )

    async def claim_profile_candidates(self, *, now: datetime, limit: int) -> list[KcIngestionJobEntity]:
        return await self._claim_by_type(
            job_type="PROFILE",
            now=now,
            limit=limit,
        )

    async def claim_purge_candidates(self, *, now: datetime, limit: int) -> list[KcIngestionJobEntity]:
        return await self._claim_by_type(
            job_type="COLLECTION_PURGE",
            now=now,
            limit=limit,
        )

    async def get_by_id(self, *, ingestion_job_id: UUID, lock: bool = False) -> KcIngestionJobEntity | None:
        statement: Select = select(KcIngestionJobEntity).where(KcIngestionJobEntity.ingestion_job_id == ingestion_job_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def get_by_idempotency_key(
        self, *, collection_id: UUID, idempotency_key: str, input_fingerprint: str,
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
    async def get_by_input(self, *, document_version_id: UUID, view_kind: str, parse_config_fingerprint: str) -> KcParseViewEntity | None:
        statement: Select = select(KcParseViewEntity).where(KcParseViewEntity.document_version_id == document_version_id, KcParseViewEntity.view_kind == view_kind, KcParseViewEntity.parse_config_fingerprint == parse_config_fingerprint)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def add(self, entity: KcParseViewEntity) -> KcParseViewEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def get_by_id(self, *, parse_view_id: UUID, lock: bool = False) -> KcParseViewEntity | None:
        statement: Select = select(KcParseViewEntity).where(KcParseViewEntity.parse_view_id == parse_view_id)
        if lock: statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def list_active_others(
        self, *, document_version_id: UUID, view_kind: str, except_parse_view_id: UUID,
    ) -> list[KcParseViewEntity]:
        statement = select(KcParseViewEntity).where(
            KcParseViewEntity.document_version_id == document_version_id,
            KcParseViewEntity.view_kind == view_kind,
            KcParseViewEntity.view_status == "ACTIVE",
            KcParseViewEntity.parse_view_id != except_parse_view_id,
        ).with_for_update()
        return list((await self.session.execute(statement)).scalars())
    async def delete_by_ids(self, parse_view_ids: list[UUID]) -> None:
        if parse_view_ids:
            await self.session.execute(delete(KcParseViewEntity).where(KcParseViewEntity.parse_view_id.in_(parse_view_ids)))


class EvidenceRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def get_by_key(self, *, parse_view_id: UUID, evidence_key: str) -> KcEvidenceEntity | None:
        statement: Select = select(KcEvidenceEntity).where(KcEvidenceEntity.parse_view_id == parse_view_id, KcEvidenceEntity.evidence_key == evidence_key)
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def add(self, entity: KcEvidenceEntity) -> KcEvidenceEntity:
        self.session.add(entity); await self.session.flush(); return entity
    async def get_by_source_ref(
        self, *, parse_view_id: UUID, source_item_ref: str
    ) -> KcEvidenceEntity | None:
        statement = (
            select(KcEvidenceEntity)
            .where(
                KcEvidenceEntity.parse_view_id == parse_view_id,
                KcEvidenceEntity.source_item_ref == source_item_ref,
            )
            .order_by(KcEvidenceEntity.ordinal)
            .limit(1)
        )
        return (await self.session.execute(statement)).scalar_one_or_none()
    async def count_staged(self, *, parse_view_id: UUID) -> int:
        statement = select(func.count()).select_from(KcEvidenceEntity).where(KcEvidenceEntity.parse_view_id == parse_view_id, KcEvidenceEntity.status == "STAGED")
        return int((await self.session.execute(statement)).scalar_one())
    async def list_staged_keys(self, *, parse_view_id: UUID) -> list[str]:
        statement = select(KcEvidenceEntity.evidence_key).where(
            KcEvidenceEntity.parse_view_id == parse_view_id,
            KcEvidenceEntity.status == "STAGED",
        ).order_by(KcEvidenceEntity.ordinal)
        return list((await self.session.execute(statement)).scalars())
    async def activate_staged(self, *, parse_view_id: UUID) -> None:
        await self.session.execute(update(KcEvidenceEntity).where(KcEvidenceEntity.parse_view_id == parse_view_id, KcEvidenceEntity.status == "STAGED").values(status="ACTIVE"))
    async def delete_by_view_ids(self, parse_view_ids: list[UUID]) -> None:
        if parse_view_ids:
            await self.session.execute(delete(KcEvidenceEntity).where(KcEvidenceEntity.parse_view_id.in_(parse_view_ids)))

    async def list_needing_index(
        self, *, parse_view_id: UUID, model_id: UUID,
        served_model_name: str, limit: int = 500,
    ) -> list[KcEvidenceEntity]:
        statement = (
            select(KcEvidenceEntity)
            .where(
                KcEvidenceEntity.parse_view_id == parse_view_id,
                KcEvidenceEntity.status == "ACTIVE",
                (KcEvidenceEntity.embedding_input_hash.is_(None)
                 | (KcEvidenceEntity.embedding_model_id != model_id)
                 | (
                     KcEvidenceEntity.embedding_served_model_name
                     != served_model_name
                 )),
            )
            .order_by(KcEvidenceEntity.ordinal, KcEvidenceEntity.fragment_index)
            .limit(limit)
        )
        return list((await self.session.execute(statement)).scalars())

    async def list_active(self, *, parse_view_id: UUID, limit: int = 500, offset: int = 0) -> list[KcEvidenceEntity]:
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

    async def count_active_for_version(self, *, document_version_id: UUID) -> int:
        statement = select(func.count()).select_from(KcEvidenceEntity).where(
            KcEvidenceEntity.document_version_id == document_version_id,
            KcEvidenceEntity.status == "ACTIVE",
            KcEvidenceEntity.embedding_input_hash.is_not(None),
        )
        return int((await self.session.execute(statement)).scalar_one())

    async def list_section_titles(self, *, document_version_id: UUID, limit: int = 20) -> list[str]:
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
            select(
                KcEvidenceEntity,
                KcBundleRevisionEntity.title,
                KcBundleRevisionDocumentEntity.declared_name,
                KcBundleRevisionDocumentEntity.external_document_id,
                KcBundleRevisionDocumentEntity.document_role,
                score.label("text_score"),
            )
            .join(KcCollectionEntity, KcCollectionEntity.collection_id == KcEvidenceEntity.collection_id)
            .join(KcBundleRevisionEntity, KcBundleRevisionEntity.bundle_revision_id == KcEvidenceEntity.bundle_revision_id)
            .join(KcBundleEntity, KcBundleEntity.bundle_id == KcBundleRevisionEntity.bundle_id)
            .join(KcDocumentVersionEntity, KcDocumentVersionEntity.document_version_id == KcEvidenceEntity.document_version_id)
            .outerjoin(
                KcBundleRevisionDocumentEntity,
                KcBundleRevisionDocumentEntity.bundle_revision_document_id
                == KcEvidenceEntity.bundle_revision_document_id,
            )
            .where(
                KcEvidenceEntity.collection_id == scope.collection_id,
                KcCollectionEntity.status == "ACTIVE",
                KcBundleRevisionEntity.bundle_id == scope.bundle_id,
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
        return [
            self._to_hit(
                entity, scope.bundle_id, rank, "TEXT",
                float(hit_score or 0), bundle_title, document_name,
                external_document_id, document_role,
            )
            for rank, (
                entity, bundle_title, document_name,
                external_document_id, document_role, hit_score,
            ) in enumerate(rows, 1)
        ]

    async def search_vector(self, *, scope: EvidenceScope, vector: list[float], limit: int = 20, max_security_level: int = 3) -> list[EvidenceHit]:
        distance = KcEvidenceEntity.embedding.op("<=>")(bindparam("evidence_vector"))
        statement = (
            select(
                KcEvidenceEntity,
                KcBundleRevisionEntity.title,
                KcBundleRevisionDocumentEntity.declared_name,
                KcBundleRevisionDocumentEntity.external_document_id,
                KcBundleRevisionDocumentEntity.document_role,
                distance.label("distance"),
            )
            .join(KcCollectionEntity, KcCollectionEntity.collection_id == KcEvidenceEntity.collection_id)
            .join(KcBundleRevisionEntity, KcBundleRevisionEntity.bundle_revision_id == KcEvidenceEntity.bundle_revision_id)
            .join(KcBundleEntity, KcBundleEntity.bundle_id == KcBundleRevisionEntity.bundle_id)
            .join(KcDocumentVersionEntity, KcDocumentVersionEntity.document_version_id == KcEvidenceEntity.document_version_id)
            .outerjoin(
                KcBundleRevisionDocumentEntity,
                KcBundleRevisionDocumentEntity.bundle_revision_document_id
                == KcEvidenceEntity.bundle_revision_document_id,
            )
            .where(
                KcEvidenceEntity.collection_id == scope.collection_id,
                KcCollectionEntity.status == "ACTIVE",
                KcBundleRevisionEntity.bundle_id == scope.bundle_id,
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
        return [
            self._to_hit(
                entity, scope.bundle_id, rank, "VECTOR",
                1.0 - float(distance_value or 1.0), bundle_title,
                document_name, external_document_id, document_role,
            )
            for rank, (
                entity, bundle_title, document_name,
                external_document_id, document_role, distance_value,
            ) in enumerate(rows, 1)
        ]

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
        statement = (
            select(
                KcEvidenceEntity,
                KcBundleRevisionEntity.bundle_id,
                KcBundleRevisionEntity.title,
                KcBundleRevisionDocumentEntity.declared_name,
                KcBundleRevisionDocumentEntity.external_document_id,
                KcBundleRevisionDocumentEntity.document_role,
            )
            .join(
                KcCollectionEntity,
                KcCollectionEntity.collection_id == KcEvidenceEntity.collection_id,
            )
            .join(
                KcBundleRevisionEntity,
                KcBundleRevisionEntity.bundle_revision_id
                == KcEvidenceEntity.bundle_revision_id,
            )
            .outerjoin(
                KcBundleRevisionDocumentEntity,
                KcBundleRevisionDocumentEntity.bundle_revision_document_id
                == KcEvidenceEntity.bundle_revision_document_id,
            )
            .where(
            KcEvidenceEntity.status == "ACTIVE",
            KcCollectionEntity.status == "ACTIVE",
            KcEvidenceEntity.collection_id.in_(collection_ids),
            or_(*conditions),
            )
            .order_by(KcEvidenceEntity.ordinal)
            .limit(limit)
        )
        rows = (await self.session.execute(statement)).all()
        anchor_ids = {item.evidence_id for item in anchors}
        return [
            self._to_hit(
                entity, bundle_id, index, "CONTEXT", 0.0,
                bundle_title, document_name, external_document_id,
                document_role,
            )
            for index, (
                entity, bundle_id, bundle_title, document_name,
                external_document_id, document_role,
            ) in enumerate(rows, 1)
            if entity.evidence_id not in anchor_ids
        ]

    @staticmethod
    def _to_hit(
        entity: KcEvidenceEntity,
        bundle_id: UUID,
        rank: int,
        channel: str,
        score: float,
        bundle_title: str | None = None,
        document_name: str | None = None,
        external_document_id: str | None = None,
        document_role: str | None = None,
    ) -> EvidenceHit:
        return EvidenceHit(
            evidence_id=entity.evidence_id,
            collection_id=entity.collection_id,
            bundle_id=bundle_id,
            bundle_revision_id=entity.bundle_revision_id,
            bundle_revision_document_id=entity.bundle_revision_document_id,
            document_id=entity.document_id,
            document_version_id=entity.document_version_id,
            parse_view_id=entity.parse_view_id,
            evidence_key=entity.evidence_key,
            evidence_type=entity.evidence_type, content_text=entity.content_text,
            retrieval_text=entity.retrieval_text, heading_path=tuple(entity.heading_path_json or []),
            locator=entity.locator_json, source_spans=tuple(entity.source_spans_json or []),
            provenance=entity.provenance_json, section_key=entity.section_key,
            parent_evidence_key=entity.parent_evidence_key, ordinal=int(entity.ordinal),
            quality_score=float(entity.quality_score) if entity.quality_score is not None else None,
            local_rank=rank, channel=channel,
            bundle_title=bundle_title,
            document_name=document_name,
            external_document_id=external_document_id,
            document_role=document_role,
            content_hash=entity.content_hash,
        )
