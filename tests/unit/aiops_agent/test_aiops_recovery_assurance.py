"""数据库资源级恢复策略、演练坐标和证据快照测试。"""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from aiops_agent.application.configuration.recovery_service import recovery_drill_view
from aiops_agent.application.recovery_assurance import build_recovery_assurance_snapshot
from aiops_agent.domain.diagnosis.evidence import normalize_evidence_artifacts
from platform_core.contracts.aiops import RecoveryDrillCreate
from platform_core.identity import uuid7


NOW = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[3]


def _request(marker, source, *, evidence=True):
    return {
        "scenario": "FULL_INSTANCE",
        "assurance_level": "DATABASE_OPEN",
        "backup_source_type": source,
        "environment": "ISOLATED",
        "result": "PASS",
        "simulated_failure_at": NOW,
        "recovered_through_at": NOW - timedelta(minutes=5),
        "service_validated_at": NOW + timedelta(minutes=15),
        "recovery_marker": marker,
        "evidence": ([{"evidence_kind": "DATABASE_CHECK", "reference": "report://restore/1", "content_hash": "a" * 64}] if evidence else []),
    }


def _profile(resource_id):
    return SimpleNamespace(
        recovery_profile_id=uuid7(), resource_id=resource_id, version_no=1,
        backup_policy_enabled=True, backup_frequency_hours=24,
        backup_success_max_age_hours=26, backup_retention_days=30,
        rpo_seconds=3600, rto_seconds=7200, drill_required=True,
        required_drill_interval_days=90, required_assurance_level="DATABASE_OPEN",
        required_backup_source_types_json=["POSTGRESQL_PGBACKREST"],
        rto_clock_basis="SERVICE_UNAVAILABLE_TO_VALIDATED", source_note=None,
        status="ACTIVE", effective_at=NOW - timedelta(days=30), retired_at=None,
        confirmed_by="dba", row_version=1, created_at=NOW, updated_at=NOW,
    )


def _drill(resource_id, profile, **overrides):
    values = {
        "drill_id": uuid7(), "resource_id": resource_id, "execution_target_id": uuid7(),
        "recovery_profile_id": profile.recovery_profile_id, "recovery_profile_version": 1,
        "db_type": "POSTGRESQL", "scenario": "FULL_INSTANCE",
        "assurance_level": "DATABASE_OPEN", "backup_source_type": "POSTGRESQL_PGBACKREST",
        "environment": "ISOLATED", "result": "PASS", "status": "VERIFIED",
        "simulated_failure_at": NOW - timedelta(days=10),
        "recovered_through_at": NOW - timedelta(days=10, minutes=30),
        "service_validated_at": NOW - timedelta(days=10) + timedelta(hours=1),
        "achieved_rpo_seconds": 1800, "achieved_rto_seconds": 3600,
        "recovery_marker_json": {"kind": "POSTGRESQL", "timeline_id": 2, "lsn": "0/16B6A50"},
        "evidence_json": [{"evidence_kind": "DATABASE_CHECK", "reference": "audit/1", "content_hash": "a" * 64}],
        "source_trust_level": "USER_PROVIDED", "reviewed_by": "reviewer", "reviewed_at": NOW,
        "review_note": None, "notes": None, "row_version": 1,
        "created_at": NOW, "created_by": "operator", "updated_at": NOW, "updated_by": "operator",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Repository:
    def __init__(self, profile, drills):
        self.profile = profile
        self.drills = drills

    async def get_active_profile(self, **_):
        return self.profile

    async def list_recent_drills(self, **_):
        return self.drills


class RecoveryAssuranceTest(unittest.IsolatedAsyncioTestCase):
    def test_three_database_markers_are_typed(self) -> None:
        cases = (
            ({"kind": "ORACLE", "scn": 123}, "ORACLE_RMAN"),
            ({"kind": "POSTGRESQL", "timeline_id": 3, "lsn": "16/B374D848"}, "POSTGRESQL_PGBACKREST"),
            ({"kind": "MYSQL", "gtid_executed": "uuid:1-10"}, "MYSQL_XTRABACKUP"),
        )
        for marker, source in cases:
            with self.subTest(kind=marker["kind"]):
                request = RecoveryDrillCreate.model_validate(_request(marker, source))
                self.assertEqual(marker["kind"], request.recovery_marker.kind)

    def test_marker_source_mismatch_is_rejected_and_manual_evidence_is_optional(self) -> None:
        with self.assertRaisesRegex(ValidationError, "数据库类型不匹配"):
            RecoveryDrillCreate.model_validate(_request({"kind": "POSTGRESQL", "timeline_id": 1}, "ORACLE_RMAN"))
        request = RecoveryDrillCreate.model_validate(
            _request({"kind": "ORACLE"}, "ORACLE_RMAN", evidence=False)
        )
        self.assertEqual((), request.evidence)

    def test_public_drill_view_hides_internal_execution_target(self) -> None:
        resource_id = uuid7()
        profile = _profile(resource_id)
        view = recovery_drill_view(_drill(resource_id, profile))
        self.assertEqual(resource_id, view.resource_id)
        self.assertNotIn("execution_target_id", view.model_dump())

    async def test_snapshot_reports_latest_failure_without_hiding_prior_success(self) -> None:
        resource_id = uuid7()
        profile = _profile(resource_id)
        success = _drill(resource_id, profile)
        failure = _drill(
            resource_id, profile, result="FAIL", simulated_failure_at=NOW - timedelta(days=1),
            recovered_through_at=None, service_validated_at=None,
            achieved_rpo_seconds=None, achieved_rto_seconds=None, evidence_json=[],
        )
        target = SimpleNamespace(target_id=resource_id, tenant_id=uuid7(), domain_id=uuid7(), db_type="POSTGRESQL")
        snapshot = await build_recovery_assurance_snapshot(recovery_repository=_Repository(profile, [failure, success]), target=target, now=NOW)
        self.assertTrue(snapshot["restore_demonstrated"])
        self.assertEqual("FAIL", snapshot["latest_attempt"]["result"])
        self.assertEqual("LATEST_DRILL_FAILED", snapshot["assurance_status"])

    def test_normalized_reviewed_drill_remains_user_provided(self) -> None:
        index = normalize_evidence_artifacts(({"artifact_id": str(uuid7()), "schema_version": "TARGET_RECOVERY_ASSURANCE.v1", "payload": {"captured_at": NOW.isoformat(), "profile": {"rpo_seconds": 1, "rto_seconds": 2, "required_drill_interval_days": 3, "required_assurance_level": "DATABASE_OPEN", "required_backup_source_types": ["ORACLE_RMAN"]}, "latest_attempt": {"result": "PASS", "status": "VERIFIED", "assurance_level": "DATABASE_OPEN", "source_trust_level": "USER_PROVIDED"}, "gaps": []}},), target_id=str(uuid7()))
        self.assertEqual({"USER_PROVIDED"}, {fact.trust_level for fact in index.facts})

    def test_schema_and_static_ui_use_database_resource_scope(self) -> None:
        manifest = json.loads((ROOT / "database/oracle/aiops_agent/schema_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(47, manifest["schema_version"])
        page = (ROOT / "ui/aiops/recovery-drills.html").read_text(encoding="utf-8")
        script = (ROOT / "ui/aiops/js/aiops-recovery-drills.js").read_text(encoding="utf-8")
        self.assertIn("数据库资源", page)
        self.assertIn("/managed-resources/", script)
        self.assertNotIn("execution_target_id", page)
        self.assertNotIn("/components/", script)


if __name__ == "__main__":
    unittest.main()
