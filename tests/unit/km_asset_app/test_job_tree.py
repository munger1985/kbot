"""KM Asset 同步任务树数据契约测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from km_asset_app.application.assets import KmAssetService
from km_asset_app.application.worker import KmAssetWorker, _JobSnapshot
from platform_core.identity import uuid7


class KmAssetJobTreeTest(unittest.IsolatedAsyncioTestCase):
    async def test_reindex_tracking_uses_kc_operation_terminal_status(self):
        bundle_id = uuid7()
        revision_id = uuid7()
        generation = uuid7()
        knowledge_core = SimpleNamespace(
            get_reindex_discovery_status=AsyncMock(return_value={
                "status": "SUCCEEDED",
                "jobs": [],
            }),
            get_revision_status=AsyncMock(),
        )
        worker = KmAssetWorker(
            uow_factory=SimpleNamespace(),
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=knowledge_core,
        )
        job = _JobSnapshot(
            job_id=uuid7(),
            job_type="KC_STATUS_SYNC",
            domain_id=43,
            source_id=uuid7(),
            km_asset_id=uuid7(),
            asset_revision_id=uuid7(),
            payload_json={
                "operation_type": "DISCOVERY_REINDEX",
                "bundle_id": str(bundle_id),
                "bundle_revision_id": str(revision_id),
                "reindex_generation": str(generation),
            },
        )

        await worker._kc_status_sync(job)

        knowledge_core.get_reindex_discovery_status.assert_awaited_once()
        knowledge_core.get_revision_status.assert_not_awaited()

    async def test_processing_jobs_keep_source_and_kc_steps(self):
        class Source:
            def __init__(self):
                self.attached = True
                self.source_id = uuid7()
                self.collection_id = uuid7()
                self.display_name = "Asset MetaDB"

            def __getattribute__(self, name):
                if (
                    name not in {"attached", "__class__"}
                    and not object.__getattribute__(self, "attached")
                ):
                    raise RuntimeError("来源实体已脱离 Session")
                return object.__getattribute__(self, name)

        source = Source()
        source_id = source.source_id
        collection_id = source.collection_id

        class _Uow:
            assets = SimpleNamespace(
                list_sources=AsyncMock(return_value=[source])
            )

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                source.attached = False

        revision_id = uuid7()
        knowledge_core = SimpleNamespace(
            list_processing=AsyncMock(return_value={
                "items": [
                    {
                        "bundle_revision_id": revision_id,
                        "jobs": [
                            {
                                "job_type": "PARSE",
                                "job_status": "SUCCEEDED",
                            },
                            {
                                "job_type": "INDEX",
                                "job_status": "RUNNING",
                            },
                        ],
                    }
                ],
                "total": 1,
            })
        )
        service = KmAssetService(
            uow_factory=_Uow,
            credential_service=SimpleNamespace(),
            knowledge_core_client=knowledge_core,
        )

        result = await service.list_processing_jobs(
            domain_id=43,
            source_id=source_id,
            limit=500,
        )

        self.assertEqual(1, len(result))
        self.assertEqual(source_id, result[0]["source_id"])
        self.assertEqual("Asset MetaDB", result[0]["source_name"])
        self.assertEqual(
            ["PARSE", "INDEX"],
            [item["job_type"] for item in result[0]["jobs"]],
        )
        call = knowledge_core.list_processing.await_args.kwargs
        self.assertEqual(43, call["domain_id"])
        self.assertEqual(collection_id, call["collection_id"])

    def test_job_contract_exposes_asset_revision(self):
        revision_id = uuid7()
        row = SimpleNamespace(
            job_id=uuid7(),
            job_type="ATTACHMENT_DOWNLOAD",
            source_id=uuid7(),
            km_asset_id=uuid7(),
            asset_revision_id=revision_id,
            status="RUNNING",
            attempt_count=1,
            max_attempts=5,
            available_at=None,
            error_code=None,
            error_message=None,
            created_at=None,
            completed_at=None,
        )

        result = KmAssetService._job(row)

        self.assertEqual(revision_id, result["asset_revision_id"])


if __name__ == "__main__":
    unittest.main()
