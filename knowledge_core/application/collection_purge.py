"""Lease-based physical cleanup for an unbound Collection."""
from uuid import UUID
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Callable

from sqlalchemy import delete, select, update

from knowledge_core.entities import (
    KcBundleEntity, KcBundleRevisionDocumentEntity, KcBundleRevisionEntity,
    KcCollectionBindingEntity, KcCollectionEntity, KcDiscoveryObjectEntity,
    KcDocumentEntity, KcDocumentVersionEntity, KcEvidenceEntity,
    KcIngestionJobEntity, KcIngestionReceiptEntity, KcParseViewEntity, KcRelationEntity,
)
from knowledge_core.domain.parse_tasks import ParseLeaseError, ParseTaskClaim, claim_job, verify_lease
from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class KnowledgeCoreCollectionPurgeService:
    def __init__(self, *, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._uow_factory = uow_factory

    async def claim(self, claim: ParseTaskClaim) -> list[dict]:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            jobs = await uow.jobs.claim_purge_candidates(now=now, limit=claim.max_tasks)
            result = []
            for job in jobs:
                lease_until = claim_job(job, claim, now)
                result.append({
                    "job_id": job.ingestion_job_id,
                    "worker_id": claim.worker_id,
                    "input_fingerprint": job.input_fingerprint,
                    "collection_id": job.collection_id,
                    "lease_until": lease_until.isoformat(),
                })
            if result:
                await uow.session.flush()
                await uow.commit()
            return result

    async def run(self, *, job_id: UUID, worker_id: str, input_fingerprint: str) -> dict:
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "COLLECTION_PURGE":
                raise ValueError("invalid COLLECTION_PURGE job")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint)
            collection_id = job.collection_id
            versions = list((await uow.session.execute(select(KcDocumentVersionEntity.storage_uri).where(
                KcDocumentVersionEntity.collection_id == collection_id,
            ))).scalars())
            await uow.session.execute(delete(KcIngestionJobEntity).where(
                KcIngestionJobEntity.collection_id == collection_id,
                KcIngestionJobEntity.ingestion_job_id != job_id,
            ))
            for entity in (
                KcIngestionReceiptEntity,
                KcEvidenceEntity,
                KcDiscoveryObjectEntity,
                KcRelationEntity,
                KcParseViewEntity,
                KcBundleRevisionDocumentEntity,
                KcDocumentVersionEntity,
                KcDocumentEntity,
            ):
                await uow.session.execute(delete(entity).where(entity.collection_id == collection_id))
            await uow.session.execute(
                update(KcBundleEntity)
                .where(KcBundleEntity.collection_id == collection_id)
                .values(current_revision_id=None)
            )
            for entity in (
                KcBundleRevisionEntity,
                KcBundleEntity,
                KcCollectionBindingEntity,
            ):
                await uow.session.execute(
                    delete(entity).where(entity.collection_id == collection_id)
                )
            await uow.session.execute(delete(KcIngestionJobEntity).where(
                KcIngestionJobEntity.ingestion_job_id == job_id,
            ))
            await uow.session.execute(delete(KcCollectionEntity).where(
                KcCollectionEntity.collection_id == collection_id,
            ))
            await uow.session.flush()
            await uow.commit()
        local_deleted = 0
        for uri in versions:
            parsed = urlparse(str(uri))
            if parsed.scheme in {"", "file"} and (parsed.path or parsed.scheme == ""):
                path = Path(parsed.path if parsed.scheme == "file" else str(uri))
                if path.is_file():
                    path.unlink()
                    local_deleted += 1
        return {"job_id": job_id, "status": "SUCCEEDED", "local_objects_deleted": local_deleted}

    async def heartbeat(self, *, job_id: UUID, worker_id: str, input_fingerprint: str, lease_seconds: int = 600) -> str:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "COLLECTION_PURGE":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            job.heartbeat_at = now
            job.lease_until = now + timedelta(seconds=lease_seconds)
            await uow.session.flush()
            await uow.commit()
            return job.lease_until.isoformat()

    async def fail(self, *, job_id: UUID, worker_id: str, input_fingerprint: str, failure_code: str, message: str) -> str:
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None or uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "COLLECTION_PURGE":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint)
            job.job_status, job.failure_code, job.failure_message = "FAILED", failure_code, message[:1000]
            job.lease_owner = job.lease_until = None
            collection = await uow.collections.get_by_id(collection_id=job.collection_id, lock=True)
            if collection is not None:
                collection.status = "DELETION_FAILED"
            await uow.session.flush()
            await uow.commit()
            return "FAILED"
