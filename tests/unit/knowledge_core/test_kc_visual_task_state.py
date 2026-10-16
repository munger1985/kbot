import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from knowledge_core.application.parse_tasks import (
    KnowledgeCoreParseTaskService,
)
from knowledge_core.parsing import build_output_fingerprint


class FakeUow:
    def __init__(self, job, view, member):
        self.jobs = SimpleNamespace(get_by_id=AsyncMock(return_value=job))
        self.parse_views = SimpleNamespace(
            get_by_id=AsyncMock(return_value=view),
        )
        self.members = SimpleNamespace(
            get_by_version=AsyncMock(return_value=member),
        )
        self.evidence = SimpleNamespace()
        self.revisions = SimpleNamespace()
        self.bundles = SimpleNamespace()
        self.session = SimpleNamespace(flush=AsyncMock())
        self.commit = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class VisualTaskStateTest(unittest.IsolatedAsyncioTestCase):
    async def test_visual_parse_completion_schedules_async_enrichment(self):
        now = datetime.now(timezone.utc)
        job = SimpleNamespace(
            ingestion_job_id=UUID(int=10),
            job_type="PARSE",
            job_status="RUNNING",
            lease_owner="parser-a",
            lease_until=now + timedelta(minutes=5),
            input_fingerprint="a" * 64,
            collection_id=UUID(int=11),
            tenant_id=UUID(int=12),
            bundle_revision_id=UUID(int=13),
            document_version_id=UUID(int=14),
            parse_view_id=UUID(int=15),
            payload_json={},
            priority=0,
            max_attempts=3,
            created_by="parser-a",
            completed_at=None,
            result_json=None,
        )
        view = SimpleNamespace(
            parse_view_id=job.parse_view_id,
            document_version_id=job.document_version_id,
            view_kind="VISUAL",
            parse_config_json={"parse_strategy": "VISUAL"},
            view_status="PARSING",
            quality_score=None,
            quality_report_json=None,
            artifact_manifest_json=None,
            activated_at=None,
        )
        member = SimpleNamespace(
            document_role="CONTENT",
            member_status="PARSING",
            completed_at=None,
        )
        revision = SimpleNamespace(
            bundle_revision_id=job.bundle_revision_id,
            bundle_id=UUID(int=16),
            status="PROCESSING",
            completed_at=None,
            updated_by=None,
            created_by=job.created_by,
        )
        added_jobs = []

        async def add_job(entity):
            entity.ingestion_job_id = UUID(int=100 + len(added_jobs))
            added_jobs.append(entity)
            return entity

        jobs = SimpleNamespace(
            get_by_id=AsyncMock(return_value=job),
            get_by_idempotency_key=AsyncMock(return_value=None),
            add=AsyncMock(side_effect=add_job),
        )
        uow = FakeUow(job, view, member)
        uow.jobs = jobs
        uow.visual_assets = None
        uow.parse_views.list_active_others = AsyncMock(return_value=[])
        uow.parse_views.delete_by_ids = AsyncMock()
        uow.evidence.count_staged = AsyncMock(return_value=1)
        uow.evidence.list_staged_keys = AsyncMock(return_value=["ev-1"])
        uow.evidence.delete_by_view_ids = AsyncMock()
        uow.evidence.activate_staged = AsyncMock()
        uow.members.list_by_revision = AsyncMock(return_value=[member])
        uow.revisions.get_by_id = AsyncMock(return_value=revision)
        uow.platform_notifications = SimpleNamespace(
            emit_ingestion_terminal=AsyncMock(),
        )
        service = KnowledgeCoreParseTaskService(
            uow_factory=lambda: uow,
            artifact_store=SimpleNamespace(delete_manifest=AsyncMock()),
        )
        artifact_manifest = {
            name: {
                "uri": f"memory://{name}",
                "sha256": str(index) * 64,
                "schema": "test/v1",
                "generator": "test",
            }
            for index, name in enumerate(
                ("raw_docling", "atom_ir", "structure_ir", "evidence_manifest"),
                start=1,
            )
        }
        output_fingerprint = build_output_fingerprint(
            artifact_hashes={
                name: descriptor["sha256"]
                for name, descriptor in artifact_manifest.items()
            },
            evidence_keys=["ev-1"],
        )

        count = await service.complete(
            job_id=job.ingestion_job_id,
            worker_id="parser-a",
            input_fingerprint=job.input_fingerprint,
            artifact_manifest=artifact_manifest,
            output_fingerprint=output_fingerprint,
            quality_score=0.9,
            quality_report={"passed": True, "hard_failures": [], "metrics": {}},
        )

        self.assertEqual(1, count)
        self.assertEqual(["INDEX", "VISUAL_ENRICH"], [item.job_type for item in added_jobs])
        self.assertEqual("ACTIVE", view.view_status)
        self.assertEqual("INDEXING", member.member_status)
        self.assertEqual("SUCCEEDED", job.job_status)
        self.assertEqual(
            added_jobs[1].ingestion_job_id,
            job.result_json["visual_enrich_job_id"],
        )

    async def test_permanent_visual_failure_keeps_baseline_active(self):
        now = datetime.now(timezone.utc)
        job = SimpleNamespace(
            ingestion_job_id=UUID(int=1),
            job_type="VISUAL_ENRICH",
            job_status="RUNNING",
            lease_owner="parser-a",
            lease_until=now + timedelta(minutes=5),
            input_fingerprint="a" * 64,
            attempt_count=3,
            max_attempts=3,
            parse_view_id=UUID(int=2),
            bundle_revision_id=UUID(int=3),
            document_version_id=UUID(int=4),
            failure_class=None,
            failure_code=None,
            failure_message=None,
            completed_at=None,
        )
        view = SimpleNamespace(view_status="ACTIVE")
        member = SimpleNamespace(member_status="READY")
        uow = FakeUow(job, view, member)
        service = KnowledgeCoreParseTaskService(
            uow_factory=lambda: uow,
            artifact_store=SimpleNamespace(),
        )

        status = await service.fail(
            job_id=job.ingestion_job_id,
            worker_id="parser-a",
            input_fingerprint=job.input_fingerprint,
            failure_class="PERMANENT",
            failure_code="VISUAL_FAILED",
        )

        self.assertEqual("FAILED", status)
        self.assertEqual("ACTIVE", view.view_status)
        self.assertEqual("READY", member.member_status)
        self.assertEqual(UUID(int=2), job.parse_view_id)


if __name__ == "__main__":
    unittest.main()
