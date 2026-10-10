"""AIOps DBA Dashboard 投影与批量读取边界测试。"""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.runtime.dashboard import (
    DashboardRunSnapshot,
    DashboardSituationSnapshot,
    DashboardTargetSnapshot,
    parse_finding_payload,
    project_ops_dashboard,
)
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from platform_core.contracts.aiops.findings import (
    FindingCard,
    FindingCompilation,
    FindingConfirmation,
    FindingObjectRef,
    FindingSeverity,
    FindingType,
)
from platform_core.identity import uuid7


def _context(uow):
    context = AsyncMock()
    context.__aenter__.return_value = uow
    context.__aexit__.return_value = None
    return context


def _target(
    *,
    name: str,
    now: datetime,
    importance_level: int = 3,
    status: str = "ENABLED",
    connectivity_status: str = "CONNECTED",
    observed_status: str = "UP",
    readonly_connection_enabled: bool = True,
    last_observed_at: datetime | None | object = ...,
) -> DashboardTargetSnapshot:
    observed = now if last_observed_at is ... else last_observed_at
    return DashboardTargetSnapshot(
        target_id=uuid7(),
        display_name=name,
        db_type="ORACLE",
        environment="PROD",
        db_role="PRIMARY",
        importance_level=importance_level,
        status=status,
        connectivity_status=connectivity_status,
        observed_status=observed_status,
        readonly_connection_enabled=readonly_connection_enabled,
        last_observed_at=observed,
    )


def _situation(
    target_id,
    *,
    now: datetime,
    severity: str,
    status: str = "OPEN",
    title: str = "数据库告警",
) -> DashboardSituationSnapshot:
    return DashboardSituationSnapshot(
        situation_id=uuid7(),
        target_id=target_id,
        status=status,
        severity=severity,
        title=title,
        first_observed_at=now - timedelta(minutes=20),
        last_observed_at=now - timedelta(minutes=2),
    )


def _finding(
    *,
    finding_type: FindingType,
    severity: FindingSeverity,
    fields: dict | None = None,
    object_name: str | None = None,
) -> FindingCard:
    return FindingCard(
        finding_id=f"{finding_type}:{severity}:{object_name or '-'}",
        finding_type=finding_type,
        severity=severity,
        confirmation=FindingConfirmation.CONFIRMED,
        object_ref=FindingObjectRef(
            object_kind="DATABASE",
            object_name=object_name,
        ),
        fields=fields or {},
        impact="已确认影响范围",
    )


def _run(
    target_id,
    *,
    now: datetime,
    status: str = "COMPLETED",
    findings: tuple[FindingCard, ...] = (),
    completed_at: datetime | None | object = ...,
    error_code: str | None = None,
) -> DashboardRunSnapshot:
    completed = now if completed_at is ... else completed_at
    return DashboardRunSnapshot(
        target_id=target_id,
        ops_run_id=uuid7(),
        status=status,
        created_at=now - timedelta(minutes=5),
        completed_at=completed,
        error_code=error_code,
        findings=findings,
    )


class AIOpsDashboardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 29, 8, 0, tzinfo=UTC)

    def test_no_evidence_is_unknown_instead_of_healthy(self) -> None:
        target = _target(
            name="新接入库",
            now=self.now,
            last_observed_at=None,
        )

        dashboard = project_ops_dashboard(
            [target], (), (), (), now=self.now
        )

        self.assertEqual(dashboard.targets[0].health, "UNKNOWN")
        self.assertEqual(dashboard.targets[0].data_freshness, "UNKNOWN")
        self.assertEqual(dashboard.summary.stale_or_unknown_count, 1)
        self.assertEqual(dashboard.summary.attention_target_count, 1)

    def test_old_evidence_is_stale(self) -> None:
        target = _target(
            name="数据过期库",
            now=self.now,
            last_observed_at=self.now - timedelta(days=2),
        )

        dashboard = project_ops_dashboard(
            [target], (), (), (), now=self.now
        )

        self.assertEqual(dashboard.targets[0].health, "STALE")
        self.assertEqual(dashboard.attention_items[0].category, "DATA_FRESHNESS")

    def test_alert_severity_does_not_make_every_open_alert_critical(self) -> None:
        warning_target = _target(name="普通告警库", now=self.now)
        critical_target = _target(name="严重告警库", now=self.now)
        dashboard = project_ops_dashboard(
            [warning_target, critical_target],
            [
                _situation(
                    warning_target.target_id,
                    now=self.now,
                    severity="WARNING",
                ),
                _situation(
                    critical_target.target_id,
                    now=self.now,
                    severity="CRITICAL",
                ),
            ],
            (),
            (),
            now=self.now,
        )

        by_name = {item.display_name: item.health for item in dashboard.targets}
        self.assertEqual(by_name["普通告警库"], "WARNING")
        self.assertEqual(by_name["严重告警库"], "CRITICAL")
        self.assertEqual(dashboard.summary.critical_alert_count, 1)
        self.assertEqual(dashboard.summary.open_alert_count, 2)

    def test_unreachable_outranks_alert_and_finding(self) -> None:
        target = _target(
            name="不可达库",
            now=self.now,
            connectivity_status="UNREACHABLE",
        )
        run = _run(
            target.target_id,
            now=self.now,
            findings=(
                _finding(
                    finding_type=FindingType.TABLESPACE,
                    severity=FindingSeverity.CRITICAL,
                    fields={"used_percent": 99},
                ),
            ),
        )

        dashboard = project_ops_dashboard(
            [target],
            [_situation(target.target_id, now=self.now, severity="CRITICAL")],
            [run],
            (),
            now=self.now,
        )

        self.assertEqual(dashboard.targets[0].health, "UNREACHABLE")
        self.assertEqual(dashboard.attention_items[0].category, "CONNECTIVITY")

    def test_risks_keep_metric_semantics_separate(self) -> None:
        target = _target(name="风险库", now=self.now, importance_level=5)
        run = _run(
            target.target_id,
            now=self.now,
            findings=(
                _finding(
                    finding_type=FindingType.TABLESPACE,
                    severity=FindingSeverity.HIGH,
                    fields={"used_percent": 94},
                    object_name="USERS",
                ),
                _finding(
                    finding_type=FindingType.DG_LAG,
                    severity=FindingSeverity.HIGH,
                    fields={"lag_seconds": 45},
                ),
            ),
        )

        dashboard = project_ops_dashboard(
            [target], (), [run], (), now=self.now
        )

        row = dashboard.targets[0]
        self.assertEqual(row.capacity_percent, 94)
        self.assertIn("USERS", row.capacity_label)
        self.assertEqual(row.lag_seconds, 45)
        self.assertEqual(
            {item.category for item in dashboard.risk_items},
            {"CAPACITY", "REPLICATION"},
        )

    def test_attention_queue_uses_health_then_importance(self) -> None:
        important = _target(
            name="核心库",
            now=self.now,
            importance_level=5,
        )
        ordinary = _target(
            name="普通库",
            now=self.now,
            importance_level=2,
        )
        situations = [
            _situation(ordinary.target_id, now=self.now, severity="WARNING"),
            _situation(important.target_id, now=self.now, severity="WARNING"),
        ]

        dashboard = project_ops_dashboard(
            [ordinary, important], situations, (), (), now=self.now
        )

        self.assertEqual(dashboard.attention_items[0].target_name, "核心库")
        self.assertEqual(dashboard.attention_items[0].importance_level, 5)

    def test_recent_failure_and_automation_counts_are_exposed(self) -> None:
        target = _target(name="诊断失败库", now=self.now)
        failed = _run(
            target.target_id,
            now=self.now,
            status="FAILED",
            completed_at=self.now - timedelta(minutes=1),
            error_code="TARGET_CONNECTION_TIMEOUT",
        )

        dashboard = project_ops_dashboard(
            [target],
            (),
            (),
            [failed],
            failed_runs=[failed],
            run_status_counts={"COMPLETED": 7, "PARTIAL": 2, "FAILED": 3},
            inspection_status_counts={
                "COMPLETED": 5,
                "PARTIAL": 1,
                "FAILED": 2,
            },
            now=self.now,
        )

        self.assertEqual(dashboard.targets[0].health, "WARNING")
        self.assertEqual(dashboard.attention_items[0].category, "AUTOMATION")
        self.assertEqual(dashboard.automation.run_total, 12)
        self.assertEqual(dashboard.automation.inspection_total, 8)
        self.assertEqual(dashboard.summary.failed_automation_count, 5)

    def test_recent_failure_is_fresh_warning_even_without_other_evidence(self) -> None:
        target = _target(
            name="仅有失败证据的库",
            now=self.now,
            last_observed_at=None,
        )
        failed = _run(
            target.target_id,
            now=self.now,
            status="FAILED",
            completed_at=self.now - timedelta(minutes=1),
        )

        dashboard = project_ops_dashboard(
            [target], (), (), [failed], failed_runs=[failed], now=self.now
        )

        self.assertEqual(dashboard.targets[0].health, "WARNING")
        self.assertEqual(dashboard.targets[0].data_freshness, "CURRENT")
        self.assertEqual(dashboard.attention_items[0].category, "AUTOMATION")

    def test_disabled_target_is_not_an_attention_item(self) -> None:
        target = _target(
            name="停用库",
            now=self.now,
            status="DISABLED",
            last_observed_at=None,
        )
        dashboard = project_ops_dashboard(
            [target], (), (), (), now=self.now
        )
        self.assertEqual(dashboard.targets[0].health, "DISABLED")
        self.assertEqual(dashboard.summary.disabled_count, 1)
        self.assertEqual(dashboard.attention_items, ())

    def test_dashboard_dump_does_not_include_sid_or_sql_details(self) -> None:
        target = _target(name="生产库", now=self.now)
        run = _run(
            target.target_id,
            now=self.now,
            findings=(
                _finding(
                    finding_type=FindingType.LOCK_WAIT,
                    severity=FindingSeverity.HIGH,
                    fields={"holder_session_id": 88, "sql_id": "abc"},
                ),
            ),
        )
        dashboard = project_ops_dashboard(
            [target], (), [run], (), now=self.now
        )

        dumped = json.dumps(dashboard.model_dump(mode="json"))
        self.assertNotIn("holder_session_id", dumped)
        self.assertNotIn('"sql_id"', dumped)
        self.assertIn("target_id", dumped)

    def test_invalid_finding_payload_is_empty(self) -> None:
        self.assertEqual(parse_finding_payload(None), ())
        self.assertEqual(parse_finding_payload({"findings": "bad"}), ())
        payload = FindingCompilation(
            findings=(
                _finding(
                    finding_type=FindingType.TABLESPACE,
                    severity=FindingSeverity.LOW,
                    fields={"used_percent": 10},
                ),
            )
        ).model_dump(mode="json")
        self.assertEqual(len(parse_finding_payload(payload)), 1)

    def test_runtime_service_uses_batch_dashboard_queries(self) -> None:
        runtime_now = datetime.now(UTC)
        target = SimpleNamespace(
            target_id=uuid7(),
            display_name="核心库",
            db_type="ORACLE",
            environment="PROD",
            db_role="PRIMARY",
            importance_level=5,
            status="ENABLED",
            connectivity_status="CONNECTED",
            observed_status="UP",
            readonly_connection_enabled=True,
            last_observed_at=runtime_now,
            last_error_code=None,
        )
        run = SimpleNamespace(
            target_id=target.target_id,
            ops_run_id=uuid7(),
            status="COMPLETED",
            created_at=runtime_now - timedelta(minutes=5),
            completed_at=runtime_now,
            error_code=None,
        )
        payload = FindingCompilation(
            findings=(
                _finding(
                    finding_type=FindingType.TABLESPACE,
                    severity=FindingSeverity.HIGH,
                    fields={"used_percent": 91},
                ),
            )
        ).model_dump(mode="json")
        targets = SimpleNamespace(list_scoped=AsyncMock(return_value=[target]))
        situations = SimpleNamespace(
            list_open_for_domain=AsyncMock(return_value=[])
        )
        runs = SimpleNamespace(
            list_latest_completed_by_target=AsyncMock(return_value=[run]),
            list_recent_for_dashboard=AsyncMock(return_value=[run]),
            list_latest_failed_by_target_since=AsyncMock(return_value=[]),
            count_statuses_since=AsyncMock(
                return_value={"COMPLETED": 1}
            ),
        )
        inspections = SimpleNamespace(
            count_fire_statuses_since=AsyncMock(
                return_value={"COMPLETED": 1}
            )
        )
        turns = SimpleNamespace(
            list_finding_blocks_for_runs=AsyncMock(
                return_value={run.ops_run_id: payload}
            )
        )
        service = AIOpsRuntimeService(
            uow_factory=lambda: _context(
                SimpleNamespace(
                    targets=targets,
                    situations=situations,
                    runs=runs,
                    inspections=inspections,
                    turns=turns,
                )
            ),
            blueprint_registry=Mock(),
            handler_registry=Mock(),
        )
        scope = ConfigurationScope(
            domain_id=200,
            principal_id="user:tester",
            actor_id="tester",
            request_id="req-dashboard",
            trace_id="trc-dashboard",
        )

        dashboard = asyncio.run(service.get_dashboard(scope=scope))

        self.assertEqual(dashboard.targets[0].health, "CRITICAL")
        self.assertEqual(dashboard.targets[0].capacity_percent, 91)
        targets.list_scoped.assert_awaited_once_with(domain_id=200)
        situations.list_open_for_domain.assert_awaited_once_with(domain_id=200)
        runs.list_latest_completed_by_target.assert_awaited_once_with(
            domain_id=200
        )
        runs.list_recent_for_dashboard.assert_awaited_once()
        runs.list_latest_failed_by_target_since.assert_awaited_once()
        runs.count_statuses_since.assert_awaited_once()
        inspections.count_fire_statuses_since.assert_awaited_once()
        turns.list_finding_blocks_for_runs.assert_awaited_once_with(
            ops_run_ids=(run.ops_run_id,)
        )


if __name__ == "__main__":
    unittest.main()
