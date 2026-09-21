"""AIOps 库群总览投影与批量读取边界测试。"""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.runtime.fleet import (
    FleetRunSnapshot,
    FleetSituationSnapshot,
    FleetTargetSnapshot,
    parse_finding_payload,
    project_fleet_dashboard,
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
    status: str = "ENABLED",
    connectivity_status: str = "CONNECTED",
    observed_status: str = "UP",
    readonly_connection_enabled: bool = True,
    db_type: str = "ORACLE",
    environment: str = "PROD",
) -> FleetTargetSnapshot:
    return FleetTargetSnapshot(
        target_id=uuid7(),
        display_name=name,
        db_type=db_type,
        environment=environment,
        status=status,
        connectivity_status=connectivity_status,
        observed_status=observed_status,
        readonly_connection_enabled=readonly_connection_enabled,
    )


def _finding(
    *,
    finding_type: FindingType,
    severity: FindingSeverity,
    fields: dict | None = None,
) -> FindingCard:
    return FindingCard(
        finding_id=f"{finding_type}:{severity}",
        finding_type=finding_type,
        severity=severity,
        confirmation=FindingConfirmation.CONFIRMED,
        object_ref=FindingObjectRef(object_kind="DATABASE"),
        fields=fields or {},
        impact="已确认影响范围",
    )


def _run(
    target_id,
    *,
    findings: tuple[FindingCard, ...] = (),
    completed_at: datetime | None = None,
) -> FleetRunSnapshot:
    return FleetRunSnapshot(
        target_id=target_id,
        ops_run_id=uuid7(),
        completed_at=completed_at or datetime.now(UTC),
        findings=findings,
    )


class AIOpsFleetDashboardTest(unittest.TestCase):
    def test_disabled_outranks_alerts_and_findings(self) -> None:
        target = _target(name="停用库", status="DISABLED")
        dashboard = project_fleet_dashboard(
            [target],
            [FleetSituationSnapshot(target_id=target.target_id, status="OPEN")],
            [_run(target.target_id, findings=(_finding(
                finding_type=FindingType.TABLESPACE,
                severity=FindingSeverity.CRITICAL,
                fields={"used_percent": 99},
            ),))],
        )
        self.assertEqual(dashboard.items[0].health, "DISABLED")
        self.assertEqual(dashboard.summary.disabled_count, 1)

    def test_unreachable_from_connectivity_or_down_observation(self) -> None:
        misconfigured = _target(
            name="配置错误",
            connectivity_status="MISCONFIGURED",
        )
        down = _target(
            name="只读探活失败",
            observed_status="DOWN",
            readonly_connection_enabled=True,
        )
        monitor_only_down = _target(
            name="仅监控",
            observed_status="DOWN",
            readonly_connection_enabled=False,
        )
        dashboard = project_fleet_dashboard(
            [misconfigured, down, monitor_only_down],
            (),
            (),
        )
        by_name = {item.display_name: item.health for item in dashboard.items}
        self.assertEqual(by_name["配置错误"], "UNREACHABLE")
        self.assertEqual(by_name["只读探活失败"], "UNREACHABLE")
        self.assertEqual(by_name["仅监控"], "HEALTHY")
        self.assertEqual(dashboard.summary.unreachable_count, 2)

    def test_open_alert_makes_critical(self) -> None:
        target = _target(name="告警库")
        dashboard = project_fleet_dashboard(
            [target],
            [
                FleetSituationSnapshot(target_id=target.target_id, status="OPEN"),
                FleetSituationSnapshot(target_id=target.target_id, status="ACKNOWLEDGED"),
                FleetSituationSnapshot(target_id=target.target_id, status="RESOLVED"),
                FleetSituationSnapshot(target_id=target.target_id, status="CLOSED"),
            ],
            (),
        )
        card = dashboard.items[0]
        self.assertEqual(card.health, "CRITICAL")
        self.assertEqual(card.open_alert_count, 2)
        self.assertEqual(dashboard.summary.open_alert_count, 2)
        self.assertEqual(dashboard.summary.critical_count, 1)

    def test_high_tablespace_is_critical_and_exposes_capacity(self) -> None:
        target = _target(name="容量库")
        dashboard = project_fleet_dashboard(
            [target],
            (),
            [_run(
                target.target_id,
                findings=(
                    _finding(
                        finding_type=FindingType.TABLESPACE,
                        severity=FindingSeverity.HIGH,
                        fields={"used_percent": 92},
                    ),
                    _finding(
                        finding_type=FindingType.CONNECTION_USAGE,
                        severity=FindingSeverity.MEDIUM,
                        fields={"utilization_percent": 88},
                    ),
                ),
            )],
        )
        card = dashboard.items[0]
        self.assertEqual(card.health, "CRITICAL")
        self.assertEqual(card.max_capacity_percent, 92)

    def test_high_replication_lag_is_critical_and_exposes_seconds(self) -> None:
        target = _target(name="延迟库")
        dashboard = project_fleet_dashboard(
            [target],
            (),
            [_run(
                target.target_id,
                findings=(
                    _finding(
                        finding_type=FindingType.DG_LAG,
                        severity=FindingSeverity.HIGH,
                        fields={"lag_seconds": 12},
                    ),
                    _finding(
                        finding_type=FindingType.REPLICATION_LAG,
                        severity=FindingSeverity.CRITICAL,
                        fields={"lag_seconds": 45},
                    ),
                ),
            )],
        )
        card = dashboard.items[0]
        self.assertEqual(card.health, "CRITICAL")
        self.assertEqual(card.max_lag_seconds, 45)

    def test_medium_or_other_high_finding_is_warning(self) -> None:
        medium = _target(name="中等库")
        other_high = _target(name="等待类")
        dashboard = project_fleet_dashboard(
            [medium, other_high],
            (),
            [
                _run(
                    medium.target_id,
                    findings=(_finding(
                        finding_type=FindingType.IDLE_SESSION,
                        severity=FindingSeverity.MEDIUM,
                    ),),
                ),
                _run(
                    other_high.target_id,
                    findings=(_finding(
                        finding_type=FindingType.WAIT_CLASS,
                        severity=FindingSeverity.HIGH,
                    ),),
                ),
            ],
        )
        by_name = {item.display_name: item.health for item in dashboard.items}
        self.assertEqual(by_name["中等库"], "WARNING")
        self.assertEqual(by_name["等待类"], "WARNING")
        self.assertEqual(dashboard.summary.warning_count, 2)

    def test_healthy_summary_and_sort_order(self) -> None:
        healthy = _target(name="B-健康")
        warning = _target(name="A-警告")
        critical = _target(name="C-严重")
        disabled = _target(name="D-停用", status="DISABLED")
        unreachable = _target(
            name="E-不可达",
            connectivity_status="UNREACHABLE",
        )
        now = datetime.now(UTC)
        dashboard = project_fleet_dashboard(
            [healthy, warning, critical, disabled, unreachable],
            [FleetSituationSnapshot(target_id=critical.target_id, status="OPEN")],
            [
                _run(
                    warning.target_id,
                    findings=(_finding(
                        finding_type=FindingType.QUERY_RATE,
                        severity=FindingSeverity.MEDIUM,
                    ),),
                    completed_at=now - timedelta(hours=2),
                ),
                _run(healthy.target_id, completed_at=now),
            ],
        )
        self.assertEqual(
            [item.health for item in dashboard.items],
            ["CRITICAL", "UNREACHABLE", "WARNING", "DISABLED", "HEALTHY"],
        )
        self.assertEqual(dashboard.summary.target_count, 5)
        self.assertEqual(dashboard.summary.healthy_count, 1)
        self.assertEqual(dashboard.summary.warning_count, 1)
        self.assertEqual(dashboard.summary.critical_count, 1)
        self.assertEqual(dashboard.summary.unreachable_count, 1)
        self.assertEqual(dashboard.summary.disabled_count, 1)
        self.assertEqual(dashboard.summary.last_diagnosed_at, now)

    def test_dashboard_dump_does_not_include_sid_fields(self) -> None:
        target = _target(name="生产库")
        dashboard = project_fleet_dashboard(
            [target],
            (),
            [_run(
                target.target_id,
                findings=(_finding(
                    finding_type=FindingType.LOCK_WAIT,
                    severity=FindingSeverity.LOW,
                    fields={"holder_session_id": 88},
                ),),
            )],
        )
        dumped = json.dumps(dashboard.model_dump(mode="json"))
        self.assertNotIn("sid", dumped.lower())
        self.assertNotIn("instance_id", dumped)
        self.assertNotIn("sql_id", dumped)
        self.assertIn("target_id", dumped)

    def test_invalid_finding_payload_is_empty(self) -> None:
        self.assertEqual(parse_finding_payload(None), ())
        self.assertEqual(parse_finding_payload({"findings": "bad"}), ())
        payload = FindingCompilation(
            findings=(_finding(
                finding_type=FindingType.TABLESPACE,
                severity=FindingSeverity.LOW,
                fields={"used_percent": 10},
            ),)
        ).model_dump(mode="json")
        findings = parse_finding_payload(payload)
        self.assertEqual(len(findings), 1)
        self.assertEqual(str(findings[0].finding_type), "TABLESPACE")

    def test_runtime_service_uses_batch_repositories(self) -> None:
        now = datetime.now(UTC)
        target = SimpleNamespace(
            target_id=uuid7(),
            display_name="核心库",
            db_type="ORACLE",
            environment="PROD",
            status="ENABLED",
            connectivity_status="CONNECTED",
            observed_status="UP",
            readonly_connection_enabled=True,
        )
        run = SimpleNamespace(
            target_id=target.target_id,
            ops_run_id=uuid7(),
            completed_at=now,
        )
        payload = FindingCompilation(
            findings=(_finding(
                finding_type=FindingType.TABLESPACE,
                severity=FindingSeverity.HIGH,
                fields={"used_percent": 91},
            ),)
        ).model_dump(mode="json")
        targets = SimpleNamespace(list_scoped=AsyncMock(return_value=[target]))
        situations = SimpleNamespace(
            list_open_for_domain=AsyncMock(return_value=[])
        )
        runs = SimpleNamespace(
            list_latest_completed_by_target=AsyncMock(return_value=[run])
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
            request_id="req-fleet",
            trace_id="trc-fleet",
        )

        dashboard = asyncio.run(service.get_fleet_dashboard(scope=scope))

        self.assertEqual(dashboard.items[0].health, "CRITICAL")
        self.assertEqual(dashboard.items[0].max_capacity_percent, 91)
        self.assertEqual(dashboard.items[0].latest_run_id, run.ops_run_id)
        targets.list_scoped.assert_awaited_once_with(domain_id=200)
        situations.list_open_for_domain.assert_awaited_once_with(domain_id=200)
        runs.list_latest_completed_by_target.assert_awaited_once_with(
            domain_id=200
        )
        turns.list_finding_blocks_for_runs.assert_awaited_once_with(
            ops_run_ids=(run.ops_run_id,)
        )


if __name__ == "__main__":
    unittest.main()
