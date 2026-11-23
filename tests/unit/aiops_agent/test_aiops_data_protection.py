"""数据保护配置、恢复门禁和指标的纯代码回归；不启动任何数据库工具。"""

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.application.configuration.data_protection_service import DataProtectionConfigurationMixin
from aiops_agent.application.data_protection_backup import backup_storage_key, backup_template
from aiops_agent.application.data_protection_jobs import ApprovedProtection
from aiops_agent.application.data_protection_execution import (
    BackupArtifact, DrillDestination, DrillPreflight, DrillRejected, DrillRequest,
    DrillVerification, execute_isolated_drill,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.configuration.data_protection_help import TOOLS
from aiops_agent.domain.data_protection_metrics import evaluate_protection_metrics
from platform_core.contracts.aiops.data_protection import (
    ProtectionBackupScheduleChange, ProtectionPlanSave, ProtectionPlanSettings,
)
from platform_core.identity import uuid7


def test_plan_cannot_set_enabled_or_embed_a_command():
    with pytest.raises(ValidationError):
        ProtectionPlanSettings(tool="RMAN", enabled=True)
    with pytest.raises(ValidationError):
        ProtectionPlanSettings(tool="RMAN", backup_template_key="/usr/bin/rman")
    with pytest.raises(ValidationError):
        ProtectionPlanSettings(tool="RMAN", backup_schedule={
            "timezone": "UTC", "once_at": "2027-01-01T01:00:00Z",
        })


@pytest.mark.asyncio
@pytest.mark.parametrize(("engine", "db_version"), [
    ("ORACLE", "19c"), ("ORACLE", "26ai"), ("ORACLE", "UNKNOWN"),
    ("POSTGRESQL", "16"), ("POSTGRESQL", "17"), ("POSTGRESQL", "18"),
    ("MYSQL", "5.7"), ("MYSQL", "8.0"), ("MYSQL", "8.4"),
])
async def test_plan_draft_scoped_and_identity_uses_existing_secret_store(engine, db_version):
    resource_id = uuid7()
    scope = SimpleNamespace(domain_id=19, actor_id="owner")
    target = SimpleNamespace(db_type=engine, version_code=db_version, db_role="PRIMARY")
    rows, secrets = {}, {}

    async def put(*, external_key, credential_kind, values, **kwargs):
        secrets[str(external_key), credential_kind] = values

    async def find(*, external_key, credential_kind, domain_id, **kwargs):
        assert domain_id == scope.domain_id
        return SimpleNamespace(status="ACTIVE") if (external_key, credential_kind) in secrets else None

    async def get(*, resource_id, domain_id, **kwargs):
        assert domain_id == scope.domain_id
        return rows.get(resource_id)

    async def add(row):
        rows[row.resource_id] = row
        return row

    uow = SimpleNamespace(
        protection_plans=SimpleNamespace(get=get, add=add, history=AsyncMock(return_value=[])),
        managed_credentials=SimpleNamespace(find=find), commit=AsyncMock(),
    )

    @asynccontextmanager
    async def uow_factory():
        yield uow

    service = DataProtectionConfigurationMixin()
    service._uow_factory = uow_factory
    service._managed_credentials = SimpleNamespace(put=AsyncMock(side_effect=put))
    service._get_database_resource_and_target = AsyncMock(return_value=(object(), target))
    initial = await service.get_data_protection(scope=scope, resource_id=resource_id)
    assert initial.status == "NOT_ENROLLED"
    settings = ProtectionPlanSettings(
        tool=TOOLS[engine], importance="HIGH", topology="ASYNC_REPLICA",
        storage_key="warehouse" if engine == "POSTGRESQL" else None,
        backup_directory=None if engine == "POSTGRESQL" else "/srv/backup",
        backup_schedule={"cron": "0 2 * * *", "timezone": "Asia/Shanghai"},
    )
    request = ProtectionPlanSave(expected_revision=0, settings=settings,
                                 backup_identity={"username": "owner", "password": "sensitive-password"})
    saved = await service.save_data_protection(scope=scope, resource_id=resource_id, request=request)
    assert saved.status == "DRAFT" and saved.revision == 1
    assert saved.has_backup_identity and not saved.has_restore_identity
    assert "sensitive-password" not in saved.model_dump_json()
    assert "sensitive-password" not in str(rows[resource_id].settings_json)
    assert saved.advice.rto_seconds == 3600 and saved.advice.rpo_seconds == 300
    assert {row.status for row in saved.readiness} == {"MISSING", "UNVERIFIED"}
    service._get_database_resource_and_target.assert_awaited_with(
        uow=uow, scope=scope, resource_id=resource_id, lock=True,
    )
    with pytest.raises(AIOpsApplicationError):
        await service.save_data_protection(scope=scope, resource_id=resource_id, request=request)
    assert service._managed_credentials.put.await_count == 1
    uow.commit.assert_awaited_once()
    change = ProtectionBackupScheduleChange(expected_revision=saved.revision, enabled=True)
    preset = backup_template(settings.tool, db_version)
    assert saved.db_version == db_version
    if preset is None:
        assert saved.backup_template_key is None and saved.backup_template_version is None
        with pytest.raises(AIOpsApplicationError, match="数据库版本没有内置备份预设"):
            await service.change_data_protection_backup_schedule(scope=scope, resource_id=resource_id, request=change)
        assert rows[resource_id].next_backup_at is None
        return
    with pytest.raises(AIOpsApplicationError, match="未配置获准的备份执行位置"):
        await service.change_data_protection_backup_schedule(scope=scope, resource_id=resource_id, request=change)
    assert rows[resource_id].next_backup_at is None
    binding = ApprovedProtection(
        uuid7(), 1, settings.tool, preset.key, preset.revision, backup_storage_key(settings), db_version,
    )
    service._protection_backend = SimpleNamespace(preflight=AsyncMock(return_value=replace(binding, db_version="changed")))
    with pytest.raises(AIOpsApplicationError, match="保护配置已变化"):
        await service.change_data_protection_backup_schedule(scope=scope, resource_id=resource_id, request=change)
    assert rows[resource_id].next_backup_at is None
    service._protection_backend.preflight.return_value = binding
    enabled = await service.change_data_protection_backup_schedule(scope=scope, resource_id=resource_id, request=change)
    assert enabled.next_backup_at is not None and enabled.status == "SCHEDULED"
    assert enabled.backup_template_key == preset.key and enabled.backup_template_version == preset.revision
    assert enabled.backup_tool_requirement == preset.tool_requirement
    binding = rows[resource_id].approval_json["BACKUP"]["binding"]
    assert binding["template_key"] == preset.key and binding["db_version"] == db_version
    disabled = await service.change_data_protection_backup_schedule(
        scope=scope, resource_id=resource_id,
        request=ProtectionBackupScheduleChange(expected_revision=saved.revision, enabled=False),
    )
    assert disabled.next_backup_at is None and disabled.status == "DRAFT"


T0 = datetime(2026, 11, 18, 2, 0, tzinfo=UTC)
DESTINATION = DrillDestination("drill", "drill-instance", "/srv/drill/data")
ARTIFACT = BackupArtifact("repository", "backup-1", T0 - timedelta(hours=1), True)
REQUEST = DrillRequest("RMAN", DESTINATION, "repository", ARTIFACT, T0)
FACTS = DrillPreflight(DESTINATION, True, True, True, True, True, True, True, True)


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", [
    "dedicated_drill_environment", "no_production_overlap", "isolated_from_production", "empty_and_idle",
    "permissions_verified", "tool_compatible", "capacity_verified", "backup_readable",
])
async def test_every_unverified_destination_condition_blocks_write(condition):
    runner = SimpleNamespace(probe_drill=AsyncMock(return_value=replace(FACTS, **{condition: False})),
                             restore_and_verify=AsyncMock())
    with pytest.raises(DrillRejected):
        await execute_isolated_drill(REQUEST, runner)
    runner.restore_and_verify.assert_not_awaited()


@pytest.mark.asyncio
async def test_changed_destination_or_backup_not_available_at_t0_never_writes():
    runner = SimpleNamespace(probe_drill=AsyncMock(return_value=replace(
        FACTS, destination=replace(DESTINATION, instance_identity="other"),
    )), restore_and_verify=AsyncMock())
    with pytest.raises(DrillRejected):
        await execute_isolated_drill(REQUEST, runner)
    runner.probe_drill.reset_mock()
    with pytest.raises(DrillRejected):
        await execute_isolated_drill(replace(REQUEST, backup=replace(ARTIFACT, available_at=T0 + timedelta(seconds=1))), runner)
    runner.probe_drill.assert_not_awaited()
    runner.restore_and_verify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", list(TOOLS.values()))
async def test_verified_empty_drill_calls_only_approved_restore_port(tool):
    request = replace(REQUEST, tool=tool)
    verified = DrillVerification(True, T0 + timedelta(minutes=15), "coordinate", T0 - timedelta(minutes=3))
    runner = SimpleNamespace(probe_drill=AsyncMock(return_value=FACTS),
                             restore_and_verify=AsyncMock(return_value=verified))
    assert await execute_isolated_drill(request, runner) == verified
    runner.probe_drill.assert_awaited_once_with(request)
    runner.restore_and_verify.assert_awaited_once_with(request)


def test_metrics_exceed_target_even_if_restore_passes_and_uncertainty_is_explicit():
    options = dict(simulated_failure_at=T0, verified_at=T0 + timedelta(minutes=47),
                   recovery_coordinate="SCN", recovered_at_latest=T0 - timedelta(minutes=18),
                   recovered_at_earliest=T0 - timedelta(minutes=18),
                   target_rto=1800, target_rpo=300, database_verified=True, latest_recovery_scenario=True)
    rto, rpo = evaluate_protection_metrics(**options)
    assert rto.verdict == "OVER_TARGET" and rto.gap_max_seconds == 17 * 60
    assert rpo.verdict == "OVER_TARGET" and rpo.gap_max_seconds == 13 * 60
    assert evaluate_protection_metrics(**(options | {"target_rto": 2820}))[0].verdict == "PASS"
    assert evaluate_protection_metrics(**(options | {"recovery_coordinate": None}))[1].verdict == "NOT_MEASURED"
    assert evaluate_protection_metrics(**(options | {"database_verified": False}))[0].verdict == "NOT_MEASURED"
    assert evaluate_protection_metrics(**(options | {"latest_recovery_scenario": False}))[1].verdict == "NOT_COMPARABLE"
    interval = evaluate_protection_metrics(**(options | {"recovered_at_latest": T0 - timedelta(minutes=1)}))[1]
    assert interval.verdict == "INCONCLUSIVE" and (interval.measured_min_seconds, interval.measured_max_seconds) == (60, 1080)
