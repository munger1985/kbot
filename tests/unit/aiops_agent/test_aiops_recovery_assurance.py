"""跨数据库恢复目标、演练记录和诊断证据测试。"""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from aiops_agent.application.recovery_assurance import (
    build_recovery_assurance_snapshot,
)
from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.configuration.recovery_service import (
    RecoveryConfigurationMixin,
)
from aiops_agent.domain.diagnosis.evidence import normalize_evidence_artifacts
from aiops_agent.entities import RecoveryDrillEntity, RecoveryProfileEntity
from platform_core.contracts.aiops import RecoveryDrillCreate
from platform_core.identity import uuid7
from pydantic import ValidationError


def _drill_request(*, marker: dict, source: str, evidence: bool = True):
    failure = datetime(2026, 10, 8, 1, tzinfo=UTC)
    return {
        "scenario": "FULL_INSTANCE",
        "assurance_level": "DATABASE_OPEN",
        "backup_source_type": source,
        "environment": "ISOLATED",
        "result": "PASS",
        "simulated_failure_at": failure,
        "recovered_through_at": failure - timedelta(minutes=5),
        "service_validated_at": failure + timedelta(minutes=15),
        "recovery_marker": marker,
        "evidence": (
            [{
                "evidence_kind": "DATABASE_CHECK",
                "reference": "report://restore/2026-10-08",
                "content_hash": "a" * 64,
            }]
            if evidence
            else []
        ),
    }


class RecoveryContractTest(unittest.TestCase):
    def test_three_database_markers_are_typed(self) -> None:
        cases = (
            ({"kind": "ORACLE", "scn": 123}, "ORACLE_RMAN"),
            (
                {"kind": "POSTGRESQL", "timeline_id": 3, "lsn": "16/B374D848"},
                "POSTGRESQL_PGBACKREST",
            ),
            (
                {
                    "kind": "MYSQL",
                    "gtid_executed": "uuid:1-10",
                    "binlog_file": "mysql-bin.000010",
                    "binlog_position": 120,
                },
                "MYSQL_XTRABACKUP",
            ),
        )
        for marker, source in cases:
            with self.subTest(kind=marker["kind"]):
                request = RecoveryDrillCreate.model_validate(
                    _drill_request(marker=marker, source=source)
                )
                self.assertEqual(marker["kind"], request.recovery_marker.kind)

    def test_marker_source_mismatch_and_pass_without_evidence_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "数据库类型不匹配"):
            RecoveryDrillCreate.model_validate(
                _drill_request(
                    marker={"kind": "POSTGRESQL", "timeline_id": 1},
                    source="ORACLE_RMAN",
                )
            )
        with self.assertRaisesRegex(ValidationError, "成功演练必须提供"):
            RecoveryDrillCreate.model_validate(
                _drill_request(
                    marker={"kind": "ORACLE", "scn": 1},
                    source="ORACLE_RMAN",
                    evidence=False,
                )
            )

    def test_schema_and_ui_include_recovery_contract(self) -> None:
        root = Path(__file__).resolve().parents[3]
        manifest = json.loads(
            (root / "database/oracle/aiops_agent/schema_manifest.json").read_text()
        )
        self.assertEqual(33, manifest["schema_version"])
        self.assertIn("KBOT_OPS_RECOVERY_PROFILE", manifest["tables"])
        self.assertIn("KBOT_OPS_RECOVERY_DRILL", manifest["tables"])
        html = (root / "ui/aiops/target-detail.html").read_text()
        javascript = (root / "ui/aiops/js/aiops-pages.js").read_text()
        self.assertIn("恢复保障", html)
        self.assertIn("recovery-drills", javascript)
        public_openapi = json.loads(
            (root / "docs/openapi/aiops_public_v1.json").read_text()
        )
        self.assertIn(
            "/api/v1/apps/aiops/targets/{target_id}/recovery-profile",
            public_openapi["paths"],
        )
        self.assertIn(
            "/api/v1/apps/aiops/targets/{target_id}/recovery-drills",
            public_openapi["paths"],
        )


class _RecoveryRepository:
    def __init__(self, profile, drills):
        self.profile = profile
        self.drills = drills

    async def get_active_profile(self, **_kwargs):
        return self.profile

    async def list_recent_drills(self, **_kwargs):
        return self.drills


class RecoverySnapshotTest(unittest.IsolatedAsyncioTestCase):
    async def test_oracle_without_external_provider_keeps_explicit_gap(self) -> None:
        target_id = uuid7()
        snapshot = await build_recovery_assurance_snapshot(
            recovery_repository=_RecoveryRepository(None, []),
            target=SimpleNamespace(
                target_id=target_id,
                domain_id=1,
                db_type="ORACLE",
                capabilities_json={},
            ),
            now=datetime(2026, 10, 8, tzinfo=UTC),
        )

        codes = {item["code"] for item in snapshot["gaps"]}
        self.assertIn("EXTERNAL_BACKUP_PROVIDER_NOT_CONFIGURED", codes)

    async def test_latest_failure_does_not_hide_older_success(self) -> None:
        now = datetime(2026, 10, 8, tzinfo=UTC)
        target_id = uuid7()
        profile_id = uuid7()
        profile = RecoveryProfileEntity(
            recovery_profile_id=profile_id,
            target_id=target_id,
            domain_id=1,
            version_no=2,
            rpo_seconds=300,
            rto_seconds=900,
            required_drill_interval_days=30,
            required_assurance_level="DATABASE_OPEN",
            rto_clock_basis="SERVICE_UNAVAILABLE_TO_VALIDATED",
            source_note="业务灾备规范",
            status="ACTIVE",
            effective_at=now,
            retired_at=None,
            confirmed_by="operator",
            row_version=1,
            created_by="operator",
            updated_by="operator",
            created_at=now,
            updated_at=now,
        )

        def drill(*, days: int, result: str, status: str, version: int):
            return RecoveryDrillEntity(
                drill_id=uuid7(), target_id=target_id, domain_id=1,
                recovery_profile_id=profile_id, recovery_profile_version=version,
                db_type="POSTGRESQL", scenario="FULL_INSTANCE",
                assurance_level="DATABASE_OPEN",
                backup_source_type="POSTGRESQL_PGBACKREST",
                environment="ISOLATED", result=result, status=status,
                simulated_failure_at=now - timedelta(days=days),
                recovered_through_at=now - timedelta(days=days, minutes=5),
                service_validated_at=now - timedelta(days=days) + timedelta(minutes=15),
                achieved_rpo_seconds=300, achieved_rto_seconds=900,
                recovery_marker_json={"kind": "POSTGRESQL", "timeline_id": 1},
                evidence_json=[{"reference": "report", "content_hash": "a" * 64, "evidence_kind": "DATABASE_CHECK"}],
                source_trust_level="USER_PROVIDED", source_ops_run_id=None,
                reviewed_by="reviewer", reviewed_at=now, review_note=None,
                notes=None, row_version=1, created_by="operator",
                updated_by="operator", created_at=now, updated_at=now,
            )

        latest_failure = drill(days=1, result="FAIL", status="VERIFIED", version=2)
        older_success = drill(days=40, result="PASS", status="VERIFIED", version=1)
        snapshot = await build_recovery_assurance_snapshot(
            recovery_repository=_RecoveryRepository(profile, [latest_failure, older_success]),
            target=SimpleNamespace(
                target_id=target_id,
                domain_id=1,
                db_type="POSTGRESQL",
                capabilities_json={},
            ),
            now=now,
        )

        self.assertEqual("FAIL", snapshot["latest_attempt"]["result"])
        self.assertEqual("PASS", snapshot["latest_verified_success"]["result"])
        codes = {item["code"] for item in snapshot["gaps"]}
        self.assertIn("DRILL_STALE", codes)
        self.assertIn("POLICY_CHANGED_SINCE_DRILL", codes)
        self.assertIn("EXTERNAL_BACKUP_PROVIDER_NOT_CONFIGURED", codes)

        index = normalize_evidence_artifacts(
            ({
                "artifact_id": str(uuid7()),
                "schema_version": "TARGET_RECOVERY_ASSURANCE.v1",
                "payload": snapshot,
            },),
            target_id=str(target_id),
        )
        self.assertTrue(
            any(fact.metric_or_fact_type == "recovery.drill.latest_attempt" for fact in index.facts)
        )
        self.assertIn("DRILL_STALE", {gap["code"] for gap in index.gaps})


class _TargetRepository:
    def __init__(self, target):
        self.target = target

    async def get_scoped(self, **_kwargs):
        return self.target


class _MutableRecoveryRepository(_RecoveryRepository):
    async def list_profile_versions(self, **_kwargs):
        return [self.profile] if self.profile else []

    async def add_profile(self, entity):
        self.profile = entity
        return entity

    async def add_drill(self, entity):
        self.drills.insert(0, entity)
        return entity

    async def get_drill_scoped(self, *, drill_id, **_kwargs):
        return next((row for row in self.drills if row.drill_id == drill_id), None)


class _Outbox:
    async def add(self, _entity):
        return None


class _Session:
    async def flush(self):
        return None


class _Uow:
    def __init__(self, target, recovery):
        self.targets = _TargetRepository(target)
        self.recovery = recovery
        self.outbox = _Outbox()
        self.session = _Session()


class _RecoveryService(RecoveryConfigurationMixin):
    def __init__(self, uow):
        self.uow = uow

    @staticmethod
    def _check_version(actual, expected):
        if int(actual) != int(expected):
            raise AssertionError("版本不一致")

    async def _idempotent(self, *, handler, **_kwargs):
        return await handler(self.uow, datetime(2026, 10, 8, tzinfo=UTC))


class RecoveryConfigurationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_profile_versions_drill_metrics_and_review_trust(self) -> None:
        target = SimpleNamespace(
            target_id=uuid7(), domain_id=1, db_type="MYSQL"
        )
        repository = _MutableRecoveryRepository(None, [])
        service = _RecoveryService(_Uow(target, repository))
        scope = ConfigurationScope(
            domain_id=1,
            principal_id="principal",
            actor_id="operator",
            request_id="request",
            trace_id="trace",
        )
        from platform_core.contracts.aiops import (
            RecoveryDrillReview,
            TargetRecoveryProfileUpsert,
        )

        first = await service.upsert_recovery_profile(
            scope=scope,
            target_id=target.target_id,
            request=TargetRecoveryProfileUpsert(
                rpo_seconds=300,
                rto_seconds=900,
                required_drill_interval_days=30,
                required_assurance_level="DATABASE_OPEN",
            ),
            idempotency_key="profile-1",
        )
        second = await service.upsert_recovery_profile(
            scope=scope,
            target_id=target.target_id,
            request=TargetRecoveryProfileUpsert(
                rpo_seconds=600,
                rto_seconds=1200,
                required_drill_interval_days=60,
                required_assurance_level="APPLICATION_VALIDATED",
            ),
            idempotency_key="profile-2",
        )
        self.assertEqual(1, first.version_no)
        self.assertEqual(2, second.version_no)

        drill = await service.create_recovery_drill(
            scope=scope,
            target_id=target.target_id,
            request=RecoveryDrillCreate.model_validate(
                _drill_request(
                    marker={
                        "kind": "MYSQL",
                        "gtid_executed": "uuid:1-10",
                        "binlog_file": "mysql-bin.000010",
                        "binlog_position": 120,
                    },
                    source="MYSQL_XTRABACKUP",
                )
            ),
            idempotency_key="drill-1",
        )
        self.assertEqual(300, drill.achieved_rpo_seconds)
        self.assertEqual(900, drill.achieved_rto_seconds)
        reviewed = await service.review_recovery_drill(
            scope=scope,
            target_id=target.target_id,
            drill_id=drill.drill_id,
            request=RecoveryDrillReview(decision="VERIFY"),
            expected_version=1,
            idempotency_key="review-1",
        )
        self.assertEqual("VERIFIED", reviewed.status)
        self.assertEqual("USER_PROVIDED", reviewed.source_trust_level)


if __name__ == "__main__":
    unittest.main()
