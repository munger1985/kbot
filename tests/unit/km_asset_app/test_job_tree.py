"""KM Asset 同步任务树数据契约测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from km_asset_app.application.assets import KmAssetService
from platform_core.identity import uuid7


class KmAssetJobTreeTest(unittest.IsolatedAsyncioTestCase):
    async def test_processing_jobs_keep_source_and_kc_steps(self):
        source = SimpleNamespace(
            source_id=uuid7(),
            collection_id=uuid7(),
            display_name="Asset MetaDB",
        )

        class _Uow:
            assets = SimpleNamespace(
                list_sources=AsyncMock(return_value=[source])
            )

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback

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
            source_id=source.source_id,
            limit=500,
        )

        self.assertEqual(1, len(result))
        self.assertEqual(source.source_id, result[0]["source_id"])
        self.assertEqual("Asset MetaDB", result[0]["source_name"])
        self.assertEqual(
            ["PARSE", "INDEX"],
            [item["job_type"] for item in result[0]["jobs"]],
        )
        call = knowledge_core.list_processing.await_args.kwargs
        self.assertEqual(43, call["domain_id"])
        self.assertEqual(source.collection_id, call["collection_id"])

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
