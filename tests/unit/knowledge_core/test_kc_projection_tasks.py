"""KC Projection 统一抢占协议测试。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from knowledge_core.application.projection_tasks import (
    PROJECTION_JOB_TYPES,
    KnowledgeCoreProjectionTaskService,
)
from knowledge_core.domain.parse_tasks import ParseTaskClaim
from knowledge_core.workers.job_wait import AdaptiveJobWait
from knowledge_core.workers.projection.worker import KcIndexProfileWorker
from platform_core.identity import uuid7


class _Uow:
    def __init__(self, job):
        self.jobs = SimpleNamespace(
            claim_candidates_by_types=AsyncMock(return_value=[job])
        )
        self.session = SimpleNamespace(flush=AsyncMock())
        self.commit = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _Client:
    def __init__(self, task):
        self.task = task
        self.claim_calls = 0

    async def claim_projection(self, **kwargs):
        self.claim_calls += 1
        return [self.task]


def _waiter():
    return AdaptiveJobWait(
        listener=None,
        notification_timeout_seconds=30,
        fallback_min_seconds=1,
        fallback_max_seconds=30,
        fallback_multiplier=2,
        jitter_ratio=0,
    )


class KcProjectionTasksTest(unittest.IsolatedAsyncioTestCase):
    async def test_service_claims_all_projection_types_in_one_query(self):
        now = datetime.now(timezone.utc)
        job = SimpleNamespace(
            ingestion_job_id=uuid7(),
            job_type="INDEX",
            job_status="PENDING",
            available_at=now - timedelta(seconds=1),
            lease_owner=None,
            lease_until=None,
            heartbeat_at=None,
            started_at=None,
            attempt_count=0,
            row_version=1,
            input_fingerprint="a" * 64,
            collection_id=uuid7(),
            parse_view_id=uuid7(),
            payload_json={},
        )
        uow = _Uow(job)
        service = KnowledgeCoreProjectionTaskService(
            uow_factory=lambda: uow
        )

        tasks = await service.claim(
            ParseTaskClaim("projection-worker", 1, 600)
        )

        self.assertEqual(1, len(tasks))
        self.assertEqual("INDEX", tasks[0].job_type)
        uow.jobs.claim_candidates_by_types.assert_awaited_once()
        self.assertEqual(
            PROJECTION_JOB_TYPES,
            uow.jobs.claim_candidates_by_types.await_args.kwargs[
                "job_types"
            ],
        )
        uow.commit.assert_awaited_once()

    async def test_worker_uses_one_projection_claim_request(self):
        task = {
            "job_id": str(uuid7()),
            "job_type": "PROFILE",
            "worker_id": "projection-worker",
            "input_fingerprint": "a" * 64,
        }
        client = _Client(task)
        worker = KcIndexProfileWorker(
            client=client,
            worker_id="projection-worker",
            job_wait=_waiter(),
        )

        kind, claimed = await worker._next_task()

        self.assertEqual("profile", kind)
        self.assertIs(task, claimed)
        self.assertEqual(1, client.claim_calls)


if __name__ == "__main__":
    unittest.main()
