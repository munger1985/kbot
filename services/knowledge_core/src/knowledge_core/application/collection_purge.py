"""基于租约的 Collection 两阶段物理清理。"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from uuid import UUID

from knowledge_core.domain.parse_tasks import ParseLeaseError, verify_lease
from knowledge_core.persistence import KnowledgeCoreUnitOfWork
from knowledge_core.ports.object_store import KnowledgeObjectStore


def _now() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeCoreCollectionPurgeService:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], KnowledgeCoreUnitOfWork],
        object_store: KnowledgeObjectStore,
        notification_publisher,
    ):
        self._uow_factory = uow_factory
        self._object_store = object_store
        if notification_publisher is None:
            raise ValueError("Knowledge Core 必须配置通知 Outbox Publisher")
        self._notification_publisher = notification_publisher

    async def run(
        self, *, job_id: UUID, worker_id: str, input_fingerprint: str
    ) -> dict:
        object_uris, collection_id = await self._prepare(
            job_id=job_id,
            worker_id=worker_id,
            input_fingerprint=input_fingerprint,
        )
        deleted = 0
        try:
            for uri in object_uris:
                await self._object_store.delete(uri)
                deleted += 1
        except Exception as exc:
            status = await self._record_object_failure(
                job_id=job_id,
                worker_id=worker_id,
                input_fingerprint=input_fingerprint,
                exc=exc,
            )
            return {
                "job_id": job_id,
                "status": status,
                "objects_deleted": deleted,
                "objects_pending": len(object_uris) - deleted,
            }

        async with self._uow_factory() as uow:
            self._require_repositories(uow)
            job = await uow.jobs.get_by_id(
                ingestion_job_id=job_id, lock=True
            )
            if job is None or job.job_type != "COLLECTION_PURGE":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(
                job,
                worker_id=worker_id,
                input_fingerprint=input_fingerprint,
            )
            collection = await uow.collections.get_by_id(
                collection_id=collection_id,
                lock=True,
            )
            if collection is None:
                raise ParseLeaseError("COLLECTION_PURGE_ROOT_MISSING")
            await self._notification_publisher.publish(
                uow=uow,
                event_type="knowledge.collection.purge_completed",
                actor_id=str(getattr(collection, "created_by", None) or ""),
                resource_id=str(collection_id),
                payload={
                    "job_id": str(job_id),
                    "display_name": collection.display_name,
                    "objects_deleted": deleted,
                },
            )
            await uow.collection_purge.finalize(
                collection_id=collection_id,
                purge_job_id=job_id,
            )
            await uow.flush()
            await uow.commit()
        return {
            "job_id": job_id,
            "status": "SUCCEEDED",
            "objects_deleted": deleted,
            "objects_pending": 0,
        }

    async def _prepare(
        self, *, job_id: UUID, worker_id: str, input_fingerprint: str
    ) -> tuple[list[str], UUID]:
        async with self._uow_factory() as uow:
            self._require_repositories(uow)
            job = await uow.jobs.get_by_id(
                ingestion_job_id=job_id, lock=True
            )
            if job is None or job.job_type != "COLLECTION_PURGE":
                raise ValueError("COLLECTION_PURGE Job 无效")
            verify_lease(
                job,
                worker_id=worker_id,
                input_fingerprint=input_fingerprint,
            )
            collection_id = job.collection_id
            result = dict(job.result_json or {})
            if result.get("purge_phase") != "OBJECT_DELETE_PENDING":
                object_uris = await uow.collection_purge.purge_descendants(
                    collection_id=collection_id,
                    purge_job_id=job_id,
                )
                job.result_json = {
                    "purge_phase": "OBJECT_DELETE_PENDING",
                    "object_uris": object_uris,
                }
                await uow.flush()
                await uow.commit()
            else:
                raw_uris = result.get("object_uris")
                if not isinstance(raw_uris, list) or not all(
                    isinstance(item, str) for item in raw_uris
                ):
                    raise RuntimeError("Purge 补偿状态中的对象列表无效")
                object_uris = list(raw_uris)
            return object_uris, collection_id

    async def _record_object_failure(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        input_fingerprint: str,
        exc: Exception,
    ) -> str:
        async with self._uow_factory() as uow:
            self._require_repositories(uow)
            job = await uow.jobs.get_by_id(
                ingestion_job_id=job_id, lock=True
            )
            if job is None or job.job_type != "COLLECTION_PURGE":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(
                job,
                worker_id=worker_id,
                input_fingerprint=input_fingerprint,
            )
            exhausted = int(job.attempt_count) >= int(job.max_attempts)
            job.job_status = "FAILED" if exhausted else "RETRY_WAIT"
            job.available_at = _now() + timedelta(
                seconds=min(2 ** int(job.attempt_count), 60)
            )
            job.failure_class = "TRANSIENT"
            job.failure_code = "OBJECT_DELETE_FAILED"
            job.failure_message = (str(exc) or type(exc).__name__)[:1000]
            job.lease_owner = None
            job.lease_until = None
            job.heartbeat_at = None
            collection = await uow.collections.get_by_id(
                collection_id=job.collection_id,
                lock=True,
            )
            if collection is not None:
                collection.status = (
                    "DELETION_FAILED" if exhausted else "DELETING"
                )
            if exhausted:
                await self._notification_publisher.publish(
                    uow=uow,
                    event_type="knowledge.collection.purge_failed",
                    actor_id=str(getattr(collection, "created_by", None) or "") if collection else "",
                    resource_id=str(job.collection_id),
                    payload={
                        "job_id": str(job_id),
                        "error_code": "OBJECT_DELETE_FAILED",
                    },
                )
            await uow.flush()
            await uow.commit()
            return job.job_status

    async def heartbeat(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        input_fingerprint: str,
        lease_seconds: int = 600,
    ) -> str:
        now = _now()
        async with self._uow_factory() as uow:
            if uow.jobs is None:
                raise RuntimeError("Knowledge Core Unit of Work 未初始化")
            job = await uow.jobs.get_by_id(
                ingestion_job_id=job_id, lock=True
            )
            if job is None or job.job_type != "COLLECTION_PURGE":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(
                job,
                worker_id=worker_id,
                input_fingerprint=input_fingerprint,
                now=now,
            )
            job.heartbeat_at = now
            job.lease_until = now + timedelta(seconds=lease_seconds)
            await uow.flush()
            await uow.commit()
            return job.lease_until.isoformat()

    async def fail(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        input_fingerprint: str,
        failure_code: str,
        message: str,
    ) -> str:
        async with self._uow_factory() as uow:
            self._require_repositories(uow)
            job = await uow.jobs.get_by_id(
                ingestion_job_id=job_id, lock=True
            )
            if job is None or job.job_type != "COLLECTION_PURGE":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(
                job,
                worker_id=worker_id,
                input_fingerprint=input_fingerprint,
            )
            job.job_status = "FAILED"
            job.failure_code = failure_code
            job.failure_message = message[:1000]
            job.lease_owner = None
            job.lease_until = None
            collection = await uow.collections.get_by_id(
                collection_id=job.collection_id,
                lock=True,
            )
            if collection is not None:
                collection.status = "DELETION_FAILED"
            await self._notification_publisher.publish(
                uow=uow,
                event_type="knowledge.collection.purge_failed",
                actor_id=str(getattr(collection, "created_by", None) or "") if collection else "",
                resource_id=str(job.collection_id),
                payload={"job_id": str(job_id), "error_code": failure_code},
            )
            await uow.flush()
            await uow.commit()
            return "FAILED"

    @staticmethod
    def _require_repositories(uow: KnowledgeCoreUnitOfWork) -> None:
        if (
            uow.jobs is None
            or uow.collections is None
            or uow.collection_purge is None
        ):
            raise RuntimeError("Knowledge Core Unit of Work 未初始化")
