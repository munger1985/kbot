"""Knowledge Core Collection 两阶段清理补偿测试。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from knowledge_core.application.collection_purge import (
    KnowledgeCoreCollectionPurgeService,
)
from platform_core.identity import uuid7


class _PurgeRepository:
    def __init__(self, object_uris):
        self.object_uris = object_uris
        self.prepare_count = 0
        self.finalize_count = 0

    async def purge_descendants(self, **kwargs):
        del kwargs
        self.prepare_count += 1
        return list(self.object_uris)

    async def finalize(self, **kwargs):
        del kwargs
        self.finalize_count += 1


class _ObjectStore:
    def __init__(self, *, fail_uri=None):
        self.fail_uri = fail_uri
        self.deleted = []

    async def delete(self, uri):
        self.deleted.append(uri)
        if uri == self.fail_uri:
            raise OSError("对象存储暂时不可用")


class _Publisher:
    def __init__(self):
        self.calls = []

    async def publish(self, **kwargs):
        self.calls.append(kwargs)


class _Uow:
    def __init__(self, job, collection, purge):
        self.job = job
        self.jobs = SimpleNamespace(get_by_id=self._get_job)
        self.collection = collection
        self.collections = SimpleNamespace(get_by_id=self._get_collection)
        self.collection_purge = purge
        self.flush = AsyncMock()
        self.commit = AsyncMock()

    async def _get_job(self, **kwargs):
        del kwargs
        return self.job

    async def _get_collection(self, **kwargs):
        del kwargs
        return self.collection

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


class CollectionPurgeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.job_id = uuid7()
        self.collection_id = uuid7()
        self.fingerprint = "a" * 64
        self.job = SimpleNamespace(
            ingestion_job_id=self.job_id,
            collection_id=self.collection_id,
            job_type="COLLECTION_PURGE",
            job_status="RUNNING",
            input_fingerprint=self.fingerprint,
            lease_owner="worker-1",
            lease_until=datetime.now(timezone.utc) + timedelta(minutes=5),
            heartbeat_at=None,
            attempt_count=1,
            max_attempts=3,
            result_json=None,
            available_at=None,
            failure_class=None,
            failure_code=None,
            failure_message=None,
        )
        self.collection = SimpleNamespace(
            collection_id=self.collection_id,
            display_name="清理测试知识库",
            status="DELETING",
        )
        self.repository = _PurgeRepository(["object-a", "object-b"])
        self.uow = _Uow(self.job, self.collection, self.repository)

    def service(self, store, publisher=None):
        return KnowledgeCoreCollectionPurgeService(
            uow_factory=lambda: self.uow,
            object_store=store,
            notification_publisher=publisher or _Publisher(),
        )

    async def test_object_failure_persists_retryable_compensation_state(self):
        store = _ObjectStore(fail_uri="object-b")
        result = await self.service(store).run(
            job_id=self.job_id,
            worker_id="worker-1",
            input_fingerprint=self.fingerprint,
        )

        self.assertEqual("RETRY_WAIT", result["status"])
        self.assertEqual("OBJECT_DELETE_PENDING", self.job.result_json["purge_phase"])
        self.assertEqual("OBJECT_DELETE_FAILED", self.job.failure_code)
        self.assertEqual("DELETING", self.collection.status)
        self.assertEqual(0, self.repository.finalize_count)

    async def test_retry_resumes_object_phase_without_repeating_database_purge(self):
        first_store = _ObjectStore(fail_uri="object-b")
        await self.service(first_store).run(
            job_id=self.job_id,
            worker_id="worker-1",
            input_fingerprint=self.fingerprint,
        )
        self.job.job_status = "RUNNING"
        self.job.lease_owner = "worker-1"
        self.job.lease_until = datetime.now(timezone.utc) + timedelta(minutes=5)
        self.job.attempt_count = 2
        second_store = _ObjectStore()

        result = await self.service(second_store).run(
            job_id=self.job_id,
            worker_id="worker-1",
            input_fingerprint=self.fingerprint,
        )

        self.assertEqual("SUCCEEDED", result["status"])
        self.assertEqual(1, self.repository.prepare_count)
        self.assertEqual(1, self.repository.finalize_count)
        self.assertEqual(["object-a", "object-b"], second_store.deleted)

    async def test_exhausted_object_failure_marks_collection_failed(self):
        self.job.attempt_count = self.job.max_attempts
        result = await self.service(
            _ObjectStore(fail_uri="object-a")
        ).run(
            job_id=self.job_id,
            worker_id="worker-1",
            input_fingerprint=self.fingerprint,
        )

        self.assertEqual("FAILED", result["status"])
        self.assertEqual("DELETION_FAILED", self.collection.status)

    async def test_success_publishes_event_before_finalizing_root(self):
        publisher = _Publisher()
        result = await self.service(
            _ObjectStore(), publisher
        ).run(
            job_id=self.job_id,
            worker_id="worker-1",
            input_fingerprint=self.fingerprint,
        )

        self.assertEqual("SUCCEEDED", result["status"])
        self.assertEqual(1, len(publisher.calls))
        self.assertEqual(
            "knowledge.collection.purge_completed",
            publisher.calls[0]["event_type"],
        )


if __name__ == "__main__":
    unittest.main()
