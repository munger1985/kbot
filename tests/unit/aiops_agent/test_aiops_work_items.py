"""DBA 工作项合同、SLA 与指纹规则测试。"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
import unittest

from pydantic import ValidationError
from sqlalchemy.dialects import oracle

from aiops_agent.application.work_items import (
    SituationRecoveryProjector,
    _apply_inspection_health_evidence,
    _fingerprint,
    _sla,
    route_completed_run_work_items,
)
from aiops_agent.repositories.work_item import WorkItemRepository
from platform_core.contracts.aiops import (
    FindingCard,
    FindingCompilation,
    WorkItemPriority,
    WorkItemStatus,
    WorkItemTransition,
)
from platform_core.identity import uuid7


ROOT = Path(__file__).resolve().parents[3]


class _CapturedResult:
    def scalars(self):
        return self

    def first(self):
        return None


class _CapturingSession:
    statement = None

    async def execute(self, statement):
        self.statement = statement
        return _CapturedResult()


class FakeWorkItemRepository:
    def __init__(self):
        self.item = None
        self.occurrences = []
        self.links = []
        self.activities = []

    async def get_open_by_fingerprint(self, **kwargs):
        if self.item is not None and self.item.fingerprint == kwargs["fingerprint"]:
            return self.item
        return None

    async def add(self, entity):
        self.item = entity
        return entity

    async def occurrence_exists(self, *, work_item_id, ops_run_id, finding_id):
        return any(
            row.work_item_id == work_item_id
            and row.ops_run_id == ops_run_id
            and row.finding_id == finding_id
            for row in self.occurrences
        )

    async def add_occurrence(self, entity):
        self.occurrences.append(entity)
        return entity

    async def existing_link(self, *, work_item_id, resource_kind, resource_id, link_role):
        return next((
            row for row in self.links
            if row.work_item_id == work_item_id
            and row.resource_kind == resource_kind
            and row.resource_id == resource_id
            and row.link_role == link_role
        ), None)

    async def add_link(self, entity):
        self.links.append(entity)
        return entity

    async def add_activity(self, entity):
        self.activities.append(entity)
        return entity

    async def list_by_resource(self, **kwargs):
        return [self.item] if self.item is not None else []

    async def work_item_ids_for_target_finding_types(self, **kwargs):
        return [self.item.work_item_id] if self.item is not None else []

    async def get_scoped(self, *, work_item_id, **kwargs):
        if self.item is not None and self.item.work_item_id == work_item_id:
            return self.item
        return None


class _FakeUowContext:
    def __init__(self, uow):
        self.uow = uow

    async def __aenter__(self):
        return self.uow

    async def __aexit__(self, exc_type, exc, traceback):
        return None


def finding(*, severity: str = "HIGH") -> FindingCard:
    return FindingCard(
        finding_id="tablespace-users",
        finding_type="TABLESPACE",
        severity=severity,
        confirmation="CONFIRMED",
        object_ref={"object_kind": "TABLESPACE", "object_name": "USERS"},
        threshold={"metric": "usage_percent", "current": 92, "operator": ">=", "limit": 90},
        impact="USERS 表空间剩余容量不足。",
        evidence_refs=("evidence://tablespace/users",),
    )


def fake_uow(repository: FakeWorkItemRepository, payload: FindingCompilation, *, situation=None):
    return SimpleNamespace(
        work_items=repository,
        turns=SimpleNamespace(list_finding_blocks_for_runs=lambda **kwargs: asyncio.sleep(
            0, result={kwargs["ops_run_ids"][0]: payload.model_dump(mode="json")}
        )),
        inspections=SimpleNamespace(get_current_report_for_run=lambda **kwargs: asyncio.sleep(0, result=None)),
        changes=SimpleNamespace(list_proposals_for_run=lambda **kwargs: asyncio.sleep(0, result=[])),
        situations=SimpleNamespace(get_situation=lambda **kwargs: asyncio.sleep(0, result=situation)),
    )


def fake_run(*, run_id, trigger_type: str = "ALERT"):
    return SimpleNamespace(
        ops_run_id=run_id,
        domain_id=7,
        target_id=uuid7(),
        agent_id=uuid7(),
        trigger_type=trigger_type,
        workflow_kind="ALERT_DIAGNOSIS",
        source_proposal_id=None,
        situation_id=uuid7(),
        status="COMPLETED",
        plan_snapshot_json=None,
    )


class WorkItemContractTest(unittest.TestCase):
    def test_locked_fingerprint_lookup_avoids_oracle_row_limit_view(self):
        session = _CapturingSession()
        repository = WorkItemRepository(session)

        asyncio.run(repository.get_open_by_fingerprint(
            domain_id=7,
            target_id=uuid7(),
            fingerprint="fingerprint",
            lock=True,
        ))

        sql = str(session.statement.compile(dialect=oracle.dialect())).upper()
        self.assertIn("FOR UPDATE", sql)
        self.assertNotIn("FETCH FIRST", sql)

    def test_fingerprint_is_stable_for_semantically_equal_object_refs(self):
        target_id = uuid7()
        first = _fingerprint(
            domain_id=7, target_id=target_id,
            finding_type="TABLESPACE",
            object_ref={"object_kind": "TABLESPACE", "object_name": "USERS"},
            condition_code="usage_percent:>=",
        )
        second = _fingerprint(
            domain_id=7, target_id=target_id,
            finding_type="TABLESPACE",
            object_ref={"object_name": "USERS", "object_kind": "TABLESPACE"},
            condition_code="usage_percent:>=",
        )
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))

    def test_priority_sla_has_distinct_acknowledgement_and_resolution(self):
        observed_at = datetime(2026, 10, 10, 0, 0, tzinfo=UTC)
        acknowledgement, resolution, verification = _sla(
            WorkItemPriority.P1, observed_at
        )
        self.assertLess(acknowledgement, resolution)
        self.assertLess(resolution, verification)

    def test_waiting_and_resolution_require_business_details(self):
        with self.assertRaises(ValidationError):
            WorkItemTransition(
                expected_row_version=1,
                status=WorkItemStatus.WAITING,
                phase="DIAGNOSIS",
            )
        with self.assertRaises(ValidationError):
            WorkItemTransition(
                expected_row_version=1,
                status=WorkItemStatus.RESOLVED,
                phase="VERIFICATION",
            )

    def test_canonical_ddl_has_no_business_check_or_unique_constraint(self):
        ddl = (ROOT / "database/oracle/aiops_agent/012_ops_work_items.sql").read_text(
            encoding="utf-8"
        ).upper()
        self.assertNotIn(" CHECK ", ddl)
        self.assertNotIn(" UNIQUE ", ddl)
        self.assertIn("KBOT_OPS_WORK_ITEM_ACTIVITY", ddl)

    def test_high_finding_creates_open_item_and_second_run_merges_occurrence(self):
        repository = FakeWorkItemRepository()
        payload = FindingCompilation(findings=(finding(),))
        first_run = fake_run(run_id=uuid7())
        second_run = fake_run(run_id=uuid7())
        second_run.domain_id = first_run.domain_id
        second_run.target_id = first_run.target_id

        first = asyncio.run(route_completed_run_work_items(
            uow=fake_uow(repository, payload), run=first_run,
            now=datetime(2026, 10, 10, 10, tzinfo=UTC), actor_id="system:test",
        ))
        second = asyncio.run(route_completed_run_work_items(
            uow=fake_uow(repository, payload), run=second_run,
            now=datetime(2026, 10, 10, 11, tzinfo=UTC), actor_id="system:test",
        ))

        self.assertEqual(first[0].work_item_id, second[0].work_item_id)
        self.assertEqual("OPEN", second[0].status)
        self.assertEqual(2, second[0].occurrence_count)
        self.assertEqual(2, len(repository.occurrences))

    def test_resolved_critical_alert_creates_low_priority_observation(self):
        repository = FakeWorkItemRepository()
        situation = SimpleNamespace(status="RESOLVED", severity="CRITICAL", event_count=1)
        result = asyncio.run(route_completed_run_work_items(
            uow=fake_uow(repository, FindingCompilation(findings=(finding(severity="CRITICAL"),)), situation=situation),
            run=fake_run(run_id=uuid7()),
            now=datetime(2026, 10, 10, 10, tzinfo=UTC), actor_id="system:test",
        ))
        self.assertEqual("RECOVERY_OBSERVATION", result[0].work_type)
        self.assertEqual("P4", result[0].priority)
        self.assertEqual("OBSERVE", result[0].routing_decision_json["decision"])

    def test_verification_run_only_records_evidence_before_dba_completion(self):
        repository = FakeWorkItemRepository()
        repository.item = SimpleNamespace(work_item_id=uuid7(), status="OPEN")
        run = fake_run(run_id=uuid7())
        run.workflow_kind = "VERIFICATION"
        run.source_proposal_id = uuid7()

        result = asyncio.run(route_completed_run_work_items(
            uow=fake_uow(repository, FindingCompilation()), run=run,
            now=datetime(2026, 10, 10, 10, tzinfo=UTC), actor_id="system:test",
        ))

        self.assertEqual("OPEN", result[0].status)
        self.assertEqual("VERIFICATION_COMPLETED_BEFORE_DBA_COMPLETION", repository.activities[-1].activity_type)

    def test_healthy_inspection_only_resolves_pending_verification(self):
        repository = FakeWorkItemRepository()
        repository.item = SimpleNamespace(
            work_item_id=uuid7(), status="OPEN", phase="REMEDIATION",
            verification_result=None, verification_note=None,
            verified_by=None, verified_at=None, resolved_at=None,
            updated_by=None, updated_at=None,
        )
        run = fake_run(run_id=uuid7(), trigger_type="SCHEDULE")
        run.plan_snapshot_json = {"client_metadata": {"inspection": {
            "selected_check_ids": ["oracle.storage.tablespace_headroom"]
        }}}
        uow = fake_uow(repository, FindingCompilation())

        asyncio.run(_apply_inspection_health_evidence(
            uow=uow, run=run, compilation=FindingCompilation(),
            actor_id="system:test", now=datetime(2026, 10, 10, 10, tzinfo=UTC),
        ))
        self.assertEqual("OPEN", repository.item.status)
        self.assertEqual("HEALTH_OBSERVED", repository.activities[-1].activity_type)

        repository.item.status = "PENDING_VERIFICATION"
        run.ops_run_id = uuid7()
        asyncio.run(_apply_inspection_health_evidence(
            uow=uow, run=run, compilation=FindingCompilation(),
            actor_id="system:test", now=datetime(2026, 10, 10, 11, tzinfo=UTC),
        ))
        self.assertEqual("RESOLVED", repository.item.status)
        self.assertEqual("PASSED", repository.item.verification_result)

    def test_inspection_gap_cannot_be_used_as_health_evidence(self):
        repository = FakeWorkItemRepository()
        repository.item = SimpleNamespace(
            work_item_id=uuid7(), status="PENDING_VERIFICATION", phase="VERIFICATION",
        )
        run = fake_run(run_id=uuid7(), trigger_type="SCHEDULE")
        run.plan_snapshot_json = {"client_metadata": {"inspection": {
            "selected_check_ids": ["oracle.storage.tablespace_headroom"]
        }}}
        compilation = FindingCompilation(gaps=({
            "finding_type": "TABLESPACE",
            "source_tool_id": "db.storage.capacity",
            "column": "used_percent",
            "code": "COLUMN_MISSING",
            "detail": "缺少容量百分比字段。",
        },))

        affected = asyncio.run(_apply_inspection_health_evidence(
            uow=fake_uow(repository, compilation), run=run,
            compilation=compilation, actor_id="system:test",
            now=datetime(2026, 10, 10, 10, tzinfo=UTC),
        ))
        self.assertEqual([], affected)
        self.assertEqual("PENDING_VERIFICATION", repository.item.status)
        self.assertEqual([], repository.activities)

    def test_situation_recovery_projection_is_idempotent(self):
        repository = FakeWorkItemRepository()
        repository.item = SimpleNamespace(
            work_item_id=uuid7(), status="OPEN", updated_by=None, updated_at=None,
        )
        situation = SimpleNamespace(
            situation_id=uuid7(), domain_id=7, severity="CRITICAL",
            status="RESOLVED", resolved_at=datetime(2026, 10, 10, 9, tzinfo=UTC),
        )
        now = datetime(2026, 10, 10, 10, tzinfo=UTC)
        uow = SimpleNamespace(
            situations=SimpleNamespace(get_situation=lambda **kwargs: asyncio.sleep(0, result=situation)),
            work_items=repository,
            runs=SimpleNamespace(
                database_now=lambda: asyncio.sleep(0, result=now),
                list_by_situation=lambda **kwargs: asyncio.sleep(0, result=[]),
            ),
            commit=lambda: asyncio.sleep(0),
        )
        projector = SituationRecoveryProjector(
            uow_factory=lambda: _FakeUowContext(uow)
        )
        payload = {
            "situation_id": str(situation.situation_id),
            "actor_id": "system:test",
            "recovery_event_id": str(uuid7()),
        }

        asyncio.run(projector.project(payload))
        asyncio.run(projector.project(payload))

        recovery_links = [row for row in repository.links if row.link_role == "RECOVERY_EVENT"]
        recovery_activities = [row for row in repository.activities if row.activity_type == "RECOVERY_OBSERVED"]
        self.assertEqual(1, len(recovery_links))
        self.assertEqual(1, len(recovery_activities))


if __name__ == "__main__":
    unittest.main()
