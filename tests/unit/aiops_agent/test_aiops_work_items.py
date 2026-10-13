"""DBA 工作项合同、SLA 与指纹规则测试。"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
import unittest

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects import oracle

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.work_items import (
    SituationRecoveryProjector,
    WorkItemService,
    _apply_inspection_health_evidence,
    _fingerprint,
    _sla,
    route_completed_run_work_items,
)
from aiops_agent.repositories.work_item import WorkItemRepository
from platform_core.contracts.aiops import (
    FindingCard,
    FindingCompilation,
    WorkItemCreate,
    WorkItemPriority,
    WorkItemCompletion,
    WorkItemStatus,
    WorkItemTransition,
    WorkItemVersionCommand,
)
from platform_core.identity import uuid7


ROOT = Path(__file__).resolve().parents[3]


class _CapturedResult:
    def all(self):
        return []

    def scalars(self):
        return self

    def first(self):
        return None

    def one(self):
        return (0, 0, 0, 0, 0)


class _CapturingSession:
    statement = None
    scalar_result = None

    async def execute(self, statement):
        self.statement = statement
        return _CapturedResult()

    async def scalar(self, statement):
        self.statement = statement
        return self.scalar_result


class FakeWorkItemRepository:
    def __init__(self):
        self.item = None
        self.occurrences = []
        self.links = []
        self.activities = []
        self.default_group_id = uuid7()

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

    async def get_visible(self, *, work_item_id, **kwargs):
        return await self.get_scoped(work_item_id=work_item_id, **kwargs)

    async def get_group_member(self, **kwargs):
        return SimpleNamespace(status="ACTIVE")

    async def target_responsibility_group(self, **kwargs):
        return self.default_group_id, "TARGET_RESPONSIBILITY"

    async def manual_work_item_routing(self, **kwargs):
        return self.default_group_id, "TARGET_RESPONSIBILITY"

    async def target_name(self, **kwargs):
        return "核心数据库"

    async def group_name(self, **kwargs):
        return "核心数据库责任组"

    async def agent_name(self, **kwargs):
        return None

    async def list_occurrences(self, **kwargs):
        return self.occurrences

    async def list_links(self, **kwargs):
        return self.links

    async def list_activities(self, **kwargs):
        return self.activities


class _FakeUowContext:
    def __init__(self, uow):
        self.uow = uow

    async def __aenter__(self):
        return self.uow

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _SessionBoundItem(SimpleNamespace):
    active = True

    def __getattribute__(self, name):
        if (
            name not in {"active", "__dict__", "__class__"}
            and not object.__getattribute__(self, "active")
        ):
            raise AssertionError("工作项不能在 Unit of Work 关闭后读取")
        return object.__getattribute__(self, name)


class _ListWorkItemRepository:
    def __init__(self, item):
        self.item = item
        self.default_group_id = uuid7()
        self.page_kwargs = None
        self.summary_kwargs = None

    async def page(self, **kwargs):
        self.page_kwargs = kwargs
        return [(self.item, "测试实例", None, "测试 Agent")]

    async def queue_summary(self, **kwargs):
        self.summary_kwargs = kwargs
        return {
            "my_open": 0,
            "unassigned": 1,
            "urgent": 1,
            "overdue": 0,
            "pending_verification": 0,
        }

    async def target_responsibility_group(self, **kwargs):
        return self.default_group_id, "TARGET_RESPONSIBILITY"

    async def group_name(self, **kwargs):
        return "测试责任组"


class _ListUowContext:
    def __init__(self, item, now):
        self.item = item
        self.uow = SimpleNamespace(
            work_items=_ListWorkItemRepository(item),
            runs=SimpleNamespace(database_now=lambda: asyncio.sleep(0, result=now)),
        )

    async def __aenter__(self):
        return self.uow

    async def __aexit__(self, exc_type, exc, traceback):
        self.item.active = False


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
    def test_manual_work_item_is_target_centered_without_agent(self):
        now = datetime(2026, 10, 13, tzinfo=UTC)
        repository = FakeWorkItemRepository()

        async def commit():
            return None

        uow = SimpleNamespace(
            work_items=repository,
            runs=SimpleNamespace(database_now=lambda: asyncio.sleep(0, result=now)),
            commit=commit,
        )
        service = WorkItemService(
            uow_factory=lambda: _FakeUowContext(uow),
            cursor_codec=SimpleNamespace(),
        )
        actor_id = "user:dba-test"
        scope = ConfigurationScope(
            domain_id=7, principal_id=actor_id, actor_id=actor_id,
            request_id="request-test", trace_id="trace-test",
        )
        target_id = uuid7()

        result = asyncio.run(service.create_item(
            body=WorkItemCreate(
                target_id=target_id,
                work_type="PROBLEM_INVESTIGATION",
                title="检查数据库空间增长",
                summary="DBA 人工登记的数据库处理事项。",
                severity="MEDIUM",
                priority="P3",
            ),
            scope=scope,
        ))

        self.assertEqual(target_id, result.target_id)
        self.assertIsNone(result.agent_id)
        self.assertEqual(repository.default_group_id, result.responsibility_group_id)
        self.assertNotIn("agent_id", repository.activities[0].detail_json)
        self.assertEqual(str(target_id), repository.activities[0].detail_json["target_id"])

    def test_claim_starts_processing_and_completion_needs_no_manual_transition(self):
        now = datetime(2026, 10, 11, tzinfo=UTC)
        actor_id = "dba:test"
        repository = FakeWorkItemRepository()
        repository.item = SimpleNamespace(
            work_item_id=uuid7(), row_version=1,
            responsibility_group_id=uuid7(), assignee_user_id=None,
            assignment_source=None, assigned_by=None, assigned_at=None,
            status="OPEN", phase="TRIAGE", wait_reason=None,
            resolution_code=None, completion_note=None,
            completed_by=None, completed_at=None,
            updated_by=None, updated_at=None,
        )
        uow = SimpleNamespace(
            work_items=repository,
            runs=SimpleNamespace(database_now=lambda: asyncio.sleep(0, result=now)),
            commit=lambda: asyncio.sleep(0),
        )
        service = WorkItemService(
            uow_factory=lambda: _FakeUowContext(uow),
            cursor_codec=SimpleNamespace(),
        )
        service.get_item = lambda **kwargs: asyncio.sleep(0, result=repository.item)
        scope = ConfigurationScope(
            domain_id=7, principal_id=actor_id, actor_id=actor_id,
            request_id="request-test", trace_id="trace-test",
        )

        asyncio.run(service.claim(
            work_item_id=repository.item.work_item_id,
            body=WorkItemVersionCommand(expected_row_version=1), scope=scope,
        ))
        self.assertEqual(actor_id, repository.item.assignee_user_id)
        self.assertEqual("IN_PROGRESS", repository.item.status)
        self.assertEqual("REMEDIATION", repository.item.phase)
        self.assertEqual("OPEN", repository.activities[-1].from_status)
        self.assertEqual("IN_PROGRESS", repository.activities[-1].to_status)

        asyncio.run(service.return_to_group(
            work_item_id=repository.item.work_item_id,
            body=WorkItemVersionCommand(expected_row_version=1), scope=scope,
        ))
        self.assertIsNone(repository.item.assignee_user_id)
        self.assertEqual("OPEN", repository.item.status)
        self.assertEqual("TRIAGE", repository.item.phase)
        self.assertEqual("IN_PROGRESS", repository.activities[-1].from_status)
        self.assertEqual("OPEN", repository.activities[-1].to_status)

        repository.item.responsibility_group_id = None
        repository.item.agent_id = uuid7()
        repository.item.target_id = uuid7()
        asyncio.run(service.claim(
            work_item_id=repository.item.work_item_id,
            body=WorkItemVersionCommand(expected_row_version=1), scope=scope,
        ))
        self.assertEqual(repository.default_group_id, repository.item.responsibility_group_id)
        self.assertTrue(repository.activities[-1].detail_json["routing_materialized"])
        asyncio.run(service.complete(
            work_item_id=repository.item.work_item_id,
            body=WorkItemCompletion(
                expected_row_version=1, resolution_code="FIXED",
                completion_note="已完成修复并观察稳定。",
            ),
            scope=scope,
        ))
        self.assertEqual("PENDING_VERIFICATION", repository.item.status)
        self.assertEqual(actor_id, repository.item.completed_by)
        self.assertEqual("DBA_COMPLETED", repository.activities[-1].activity_type)

    def test_list_materializes_summary_before_unit_of_work_closes(self):
        now = datetime(2026, 10, 11, tzinfo=UTC)
        item = _SessionBoundItem(
            work_item_id=uuid7(), item_key="WI-TEST", agent_id=uuid7(), target_id=uuid7(),
            work_type="PROBLEM_INVESTIGATION", source_kind="ALERT",
            title="联调工作项", summary="验证工作项列表事务边界。",
            severity="HIGH", priority="P2", status="OPEN", phase="TRIAGE",
            wait_reason=None, responsibility_group_id=None,
            assignee_user_id=None, assignment_source=None, assigned_by=None,
            assigned_at=None, acknowledgement_due_at=now,
            resolution_due_at=now, verification_due_at=now,
            first_observed_at=now, last_observed_at=now,
            occurrence_count=1, reopen_count=0, resolution_code=None,
            completion_note=None, completed_by=None, completed_at=None,
            verification_result=None, verified_by=None, verified_at=None,
            resolved_at=None, closed_at=None, row_version=1,
            created_at=now, updated_at=now,
        )
        context = _ListUowContext(item, now)
        service = WorkItemService(
            uow_factory=lambda: context,
            cursor_codec=SimpleNamespace(),
        )
        scope = ConfigurationScope(
            domain_id=7, principal_id="user:test", actor_id="user:test",
            request_id="request-test", trace_id="trace-test",
        )

        page = asyncio.run(service.list_items(
            scope=scope, status=None, priority=None, agent_id=item.agent_id,
            target_id=None,
            assignee_user_id=None, unassigned=False, overdue=False,
            visibility="SCOPE", cursor=None, limit=50,
        ))

        self.assertFalse(item.active)
        self.assertEqual("WI-TEST", page.items[0].item_key)
        self.assertEqual(context.uow.work_items.default_group_id, page.items[0].responsibility_group_id)
        self.assertEqual("测试责任组", page.items[0].responsibility_group_name)
        self.assertEqual(page.items[0].agent_id, context.uow.work_items.page_kwargs["agent_id"])
        self.assertEqual(page.items[0].agent_id, context.uow.work_items.summary_kwargs["agent_id"])
        self.assertIsNone(context.uow.work_items.summary_kwargs["target_id"])

    def test_locked_fingerprint_lookup_avoids_oracle_row_limit_view(self):
        session = _CapturingSession()
        repository = WorkItemRepository(session)

        asyncio.run(repository.get_open_by_fingerprint(
            domain_id=7,
            agent_id=uuid7(),
            target_id=uuid7(),
            fingerprint="fingerprint",
            lock=True,
        ))

        sql = str(session.statement.compile(dialect=oracle.dialect())).upper()
        self.assertIn("FOR UPDATE", sql)
        self.assertNotIn("FETCH FIRST", sql)

    def test_agent_filter_is_applied_to_work_item_page(self):
        session = _CapturingSession()
        repository = WorkItemRepository(session)
        agent_id = uuid7()

        asyncio.run(repository.page(
            domain_id=7,
            actor_id="user:dba-1",
            visibility="ALL",
            agent_id=agent_id,
        ))

        compiled = session.statement.compile(dialect=oracle.dialect())
        sql = str(compiled).upper()
        self.assertIn("AGENT_ID", sql)
        self.assertIn(agent_id, compiled.params.values())

    def test_queue_summary_compiles_conditional_aggregates(self):
        session = _CapturingSession()
        repository = WorkItemRepository(session)

        summary = asyncio.run(repository.queue_summary(
            domain_id=7,
            actor_id="user:dba-1",
            now=datetime(2026, 10, 12, tzinfo=UTC),
            visibility="ALL",
        ))

        sql = str(session.statement.compile(dialect=oracle.dialect())).upper()
        self.assertEqual(5, sql.count("CASE WHEN"))
        self.assertEqual({
            "my_open": 0,
            "unassigned": 0,
            "urgent": 0,
            "overdue": 0,
            "pending_verification": 0,
        }, summary)

    def test_target_filter_is_applied_to_queue_summary(self):
        session = _CapturingSession()
        repository = WorkItemRepository(session)
        target_id = uuid7()

        asyncio.run(repository.queue_summary(
            domain_id=7,
            actor_id="user:dba-1",
            now=datetime(2026, 10, 13, tzinfo=UTC),
            visibility="ALL",
            target_id=target_id,
        ))

        compiled = session.statement.compile(dialect=oracle.dialect())
        sql = str(compiled).upper()
        self.assertIn("TARGET_ID", sql)
        self.assertIn(target_id, compiled.params.values())

    def test_manual_routing_uses_target_responsibility_without_agent(self):
        session = _CapturingSession()
        group_id = uuid7()
        session.scalar_result = group_id
        repository = WorkItemRepository(session)

        routing = asyncio.run(repository.manual_work_item_routing(
            domain_id=7,
            target_id=uuid7(),
        ))

        compiled = session.statement.compile(dialect=oracle.dialect())
        sql = str(compiled).upper()
        self.assertEqual((group_id, "TARGET_RESPONSIBILITY"), routing)
        self.assertIn("KBOT_OPS_TARGET", sql)
        self.assertIn("KBOT_OPS_RESPONSIBILITY_GROUP", sql)
        self.assertIn("ENABLED", compiled.params.values())
        self.assertIn("ACTIVE", compiled.params.values())
        self.assertNotIn("KBOT_OPS_AGENT ", sql)

    def test_scope_visibility_follows_member_and_target_responsibility(self):
        actor_id = "user:dba-1"
        condition = WorkItemRepository._visibility_condition(
            actor_id=actor_id,
            visibility="SCOPE",
        )
        sql = str(
            select(1)
            .where(condition)
            .compile(
                dialect=oracle.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        ).upper()

        self.assertIn("KBOT_OPS_RESPONSIBILITY_GROUP_MEMBER", sql)
        self.assertIn("RESPONSIBILITY_GROUP_ID", sql)
        self.assertIn('"KBOT_OPS_WORK_ITEM".TARGET_ID', sql)
        self.assertIn("RESPONSIBILITY_GROUP_ID IS NULL", sql)
        self.assertIn(actor_id.upper(), sql)
        self.assertNotIn("KBOT_OPS_AGENT ", sql)
        self.assertNotIn("MONITOR", sql)

    def test_exception_visibility_and_processing_continuity_are_explicit(self):
        actor_id = "user:dba-1"
        scope_sql = str(
            select(1)
            .where(WorkItemRepository._visibility_condition(
                actor_id=actor_id,
                visibility="SCOPE",
            ))
            .compile(
                dialect=oracle.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        ).upper()
        exception_sql = str(
            select(1)
            .where(WorkItemRepository._visibility_condition(
                actor_id=actor_id,
                visibility="EXCEPTIONS",
            ))
            .compile(
                dialect=oracle.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        ).upper()

        self.assertIn("ASSIGNEE_USER_ID", scope_sql)
        self.assertIn("NOT IN", scope_sql)
        self.assertIn("NOT (EXISTS", exception_sql)
        self.assertNotIn("KBOT_OPS_RESPONSIBILITY_GROUP_MEMBER", exception_sql)
        self.assertIsNone(WorkItemRepository._visibility_condition(
            actor_id=actor_id,
            visibility="ALL",
        ))

    def test_fingerprint_is_stable_for_semantically_equal_object_refs(self):
        agent_id, target_id = uuid7(), uuid7()
        first = _fingerprint(
            domain_id=7, agent_id=agent_id, target_id=target_id,
            finding_type="TABLESPACE",
            object_ref={"object_kind": "TABLESPACE", "object_name": "USERS"},
            condition_code="usage_percent:>=",
        )
        second = _fingerprint(
            domain_id=7, agent_id=agent_id, target_id=target_id,
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
        second_run.agent_id = first_run.agent_id

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
