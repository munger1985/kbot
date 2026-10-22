"""KC 通知唤醒与自适应等待策略测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from platform_core.identity import uuid7
from knowledge_core.persistence.uow import KnowledgeCoreUnitOfWork
from knowledge_core.ports.job_wakeup import (
    PARSE_WAKEUP_CHANNEL,
    PROJECTION_WAKEUP_CHANNEL,
    wakeup_channel_for_job,
)
from knowledge_core.repositories.ingestion_repo import (
    IngestionJobRepository,
)
from knowledge_core.workers.job_wait import AdaptiveJobWait


class _Session:
    def __init__(self):
        self.added = []

    def add(self, entity):
        self.added.append(entity)

    async def flush(self):
        return None


class _Listener:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.wait_calls = 0
        self.closed = False

    async def wait(self, timeout_seconds):
        self.wait_calls += 1
        if self.fail:
            raise RuntimeError("通知连接失败")
        return True

    async def close(self):
        self.closed = True


class _ScalarSession:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    async def scalars(self, statement):
        self.statements.append(statement)
        return self.rows


class _NestedTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _UowSession(_Session):
    def __init__(self):
        super().__init__()
        self.commit = AsyncMock()

    def begin_nested(self):
        return _NestedTransaction()


class KcJobWakeupTest(unittest.IsolatedAsyncioTestCase):
    def test_job_types_map_to_coarse_worker_channels(self):
        self.assertEqual(
            PARSE_WAKEUP_CHANNEL,
            wakeup_channel_for_job("PARSE"),
        )
        self.assertEqual(
            PARSE_WAKEUP_CHANNEL,
            wakeup_channel_for_job("VISUAL_ENRICH"),
        )
        for job_type in ("INDEX", "PROFILE", "COLLECTION_PURGE"):
            self.assertEqual(
                PROJECTION_WAKEUP_CHANNEL,
                wakeup_channel_for_job(job_type),
            )

    async def test_repository_tracks_wakeup_after_job_is_flushed(self):
        seen = []
        session = _Session()
        entity = SimpleNamespace(job_type="INDEX")

        result = await IngestionJobRepository(
            session,
            on_job_added=seen.append,
        ).add(entity)

        self.assertIs(entity, result)
        self.assertEqual(["INDEX"], seen)
        self.assertEqual([entity], session.added)

    async def test_job_repository_lists_document_jobs_for_evidence_status(self):
        document_version_id = uuid7()
        jobs = [SimpleNamespace(ingestion_job_id=uuid7())]
        session = _ScalarSession(jobs)

        result = await IngestionJobRepository(
            session
        ).list_by_document_version(
            document_version_id=document_version_id
        )

        self.assertEqual(jobs, result)
        self.assertEqual(1, len(session.statements))
        statement = str(session.statements[0]).upper()
        self.assertIn(
            '"KBOT_KC_INGESTION_JOB".DOCUMENT_VERSION_ID', statement
        )
        self.assertIn('"KBOT_KC_INGESTION_JOB".JOB_TYPE IN', statement)
        self.assertIn('"KBOT_KC_INGESTION_JOB".CREATED_AT DESC', statement)

    async def test_notification_wait_returns_without_polling_delay(self):
        listener = _Listener()
        waiter = AdaptiveJobWait(
            listener=listener,
            notification_timeout_seconds=30,
            fallback_min_seconds=1,
            fallback_max_seconds=30,
            fallback_multiplier=2,
            jitter_ratio=0,
        )

        await waiter.wait()
        await waiter.close()

        self.assertEqual(1, listener.wait_calls)
        self.assertTrue(listener.closed)

    async def test_uow_signals_pending_channel_before_commit(self):
        session = _UowSession()
        publisher = SimpleNamespace(signal=AsyncMock())
        uow = KnowledgeCoreUnitOfWork(
            lambda: session,
            job_wakeup_publisher=publisher,
        )
        await uow.__aenter__()
        uow._track_pending_job("PARSE")

        await uow.commit()

        publisher.signal.assert_awaited_once_with(
            session,
            {PARSE_WAKEUP_CHANNEL},
        )
        session.commit.assert_awaited_once()

    async def test_wakeup_failure_does_not_block_job_commit(self):
        session = _UowSession()
        publisher = SimpleNamespace(
            signal=AsyncMock(side_effect=RuntimeError("无执行权限"))
        )
        uow = KnowledgeCoreUnitOfWork(
            lambda: session,
            job_wakeup_publisher=publisher,
        )
        await uow.__aenter__()
        uow._track_pending_job("INDEX")

        await uow.commit()

        session.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
