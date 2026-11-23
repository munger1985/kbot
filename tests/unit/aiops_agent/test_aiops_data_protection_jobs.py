"""三库数据保护的 Worker 代码级验证：无数据库、工具或服务调用。"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.data_protection_execution import (
    BackupArtifact, DrillDestination, DrillPreflight, DrillRejected, DrillRequest, DrillVerification,
)
from aiops_agent.application.data_protection_backup import backup_template
from aiops_agent.application.data_protection_jobs import ApprovedProtection, BackupResult, ProtectionJobs
from aiops_agent.application.data_protection_report import publish_next_drill_report
from aiops_agent.application.data_protection_schedule import next_backup_at, required_full_at, scheduled_backup_kind
from aiops_agent.application.data_protection_restore import (
    MySQLMedia, OracleMedia, PostgreSQLMedia, render_restore, restore_preset,
)
from aiops_agent.application.configuration.data_protection_service import DataProtectionConfigurationMixin
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.reporting import report_presentation, render_pdf, resolve_system_template
from platform_core.contracts.aiops.data_protection import ProtectionDrillScheduleChange, ProtectionPlanSettings
from platform_core.identity import uuid7


BASE = datetime(2027, 1, 1, tzinfo=UTC)
WEEKLY = {
    "backup_schedule": {"timezone": "Asia/Shanghai", "cron": "0 0 * * 0"},
    "backup_incremental_schedule": {"timezone": "Asia/Shanghai", "cron": "30 1 * * 1-6"},
}
ROTATION = {"backup_rotation": {
    "timezone": "Asia/Shanghai", "start_date": "2027-01-03", "cycle_days": 3,
    "full_time": "00:00", "incremental_time": "01:30",
}}
DEST = DrillDestination("drill", "independent-instance", "/data/drill")
TARGET = uuid7()
ACTOR = uuid7()
SCOPE = {"domain_id": 25}


class MemoryRepository:
    """模拟 Plan 行认领、每个时点唯一、目的地活动键唯一。"""

    def __init__(self, tool, *, interval=timedelta(minutes=2)):
        self.plans = {}
        self.runs = {}
        settings = ProtectionPlanSettings(
            tool=tool, storage_key="warehouse", restore_template_key="restore",
            backup_directory=None if tool == "PGBACKREST" else "/data/backups",
            drill_destination_key=DEST.key, rto_seconds=15, rpo_seconds=30,
            timeout_seconds=60,
            backup_schedule={"timezone": "UTC", "cron": "1 * * * *"},
            drill_schedule={"timezone": "UTC", "once_at": BASE + interval},
        )
        self.plan = self.add_plan(settings)

    def add_plan(self, settings):
        plan = SimpleNamespace(
            resource_id=uuid7(), **SCOPE, revision=1, settings_json=settings.model_dump(mode="json"),
            approval_json=None, next_backup_at=None, next_drill_at=None,
        )
        self.plans[plan.resource_id] = plan
        return plan

    async def get(self, *, resource_id, lock=False, **scope):
        return self.plans.get(resource_id) if scope == SCOPE else None

    async def due_plan(self, *, operation, now):
        key = "next_backup_at" if operation == "BACKUP" else "next_drill_at"
        return next((p for p in self.plans.values() if getattr(p, key) and getattr(p, key) <= now), None)

    async def add_run(self, row):
        assert not any(
            (r.resource_id, r.operation, r.scheduled_at) == (row.resource_id, row.operation, row.scheduled_at)
            for r in self.runs.values()
        )
        self.runs[row.run_id] = row

    async def reserve(self, row):
        if any(r.active_key == row.active_key for r in self.runs.values()):
            return False
        await self.add_run(row)
        return True

    async def get_run(self, run_id, *, lock=False):
        return self.runs.get(run_id)

    async def expiring_run(self, *, now):
        return next((r for r in self.runs.values() if r.status == "RUNNING" and r.deadline_at <= now), None)

    async def report_pending(self):
        return next((r for r in self.runs.values() if r.operation == "DRILL" and r.completed_at and not r.report_id), None)

    async def plan_for_run(self, run):
        return self.plans[run.resource_id]

    async def latest_backup(self, resource_id, *, at):
        return max((r for r in self.runs.values() if r.resource_id == resource_id and r.operation == "BACKUP"
                    and r.status == "SUCCEEDED" and r.completed_at <= at), key=lambda r: r.completed_at, default=None)


class Scenario:
    def __init__(self, tool, coordinate_kind="SCN", coordinate="921300:1"):
        self.now = BASE
        self.repo = MemoryRepository(tool)
        self.run_repo = SimpleNamespace(add_run=AsyncMock(), add_task=AsyncMock(), add_artifact=AsyncMock())
        self.reports = SimpleNamespace(add_report=AsyncMock(), add_report_sources=AsyncMock())
        self.lock = asyncio.Lock()
        self.uow = SimpleNamespace(protection_plans=self.repo, runs=self.run_repo, inspections=self.reports, commit=AsyncMock())
        self.preflight_ok = True
        self.db_version = {"RMAN": "19c", "PGBACKREST": "16", "XTRABACKUP": "5.7"}[tool]
        self.backend = SimpleNamespace(
            preflight=AsyncMock(side_effect=self.preflight), backup=AsyncMock(side_effect=self.backup),
            drill_runner=AsyncMock(side_effect=self.drill_runner),
        )
        self.jobs = ProtectionJobs(uow_factory=self.uow_factory, backend=self.backend, clock=lambda: self.now)
        self.runner = SimpleNamespace(
            probe_drill=AsyncMock(return_value=DrillPreflight(DEST, True, True, True, True, True, True, True, True)),
            restore_and_verify=AsyncMock(side_effect=self.verify),
        )
        self.coordinate, self.coordinate_kind = coordinate, coordinate_kind

    @asynccontextmanager
    async def uow_factory(self):
        async with self.lock:
            yield self.uow

    async def preflight(self, *, scope, actor_id, resource_id, settings, operation):
        assert not self.lock.locked()
        assert scope == SCOPE and actor_id == str(ACTOR)
        if not self.preflight_ok:
            raise DrillRejected("sensitive-backend-error")
        if operation == "BACKUP":
            preset = backup_template(settings.tool, self.db_version)
            key, version = preset.key, preset.revision
        else:
            key, version = "restore", "approved-1"
        return ApprovedProtection(TARGET, 1, settings.tool, key, version, settings.storage_key, self.db_version,
                                  DEST if operation == "DRILL" else None)

    async def backup(self, work, invocation):
        assert not self.lock.locked() and work.operation == "BACKUP"
        assert "destination" in work.snapshot["approval"]["binding"]
        assert work.snapshot["approval"]["binding"]["destination"] is None
        assert invocation.tool == self.repo.plan.settings_json["tool"]
        assert (invocation.tool == "RMAN") == bool(invocation.command_file)
        return BackupResult(0, BackupArtifact("warehouse", "artifact-1", self.now, True), False)

    async def drill_runner(self, work):
        assert not self.lock.locked() and work.operation == "DRILL"
        return self.runner

    async def verify(self, request):
        assert not self.lock.locked()
        assert request.destination == DEST and request.backup.reference == "artifact-1"
        self.now += timedelta(seconds=25)
        return DrillVerification(
            True, self.now, self.coordinate, request.simulated_failure_at - timedelta(seconds=70),
            recovery_coordinate_kind=self.coordinate_kind, retrieval_seconds=3, restore_seconds=20, verification_seconds=2,
        )

    async def confirm(self, operation, plan=None):
        plan = plan or self.repo.plan
        await self.jobs.confirm(scope=SCOPE, resource_id=plan.resource_id, revision=plan.revision, actor_id=ACTOR, operation=operation)

    async def complete_backup(self):
        await self.confirm("BACKUP")
        self.now = BASE + timedelta(minutes=1)
        work = await self.jobs.claim("BACKUP")
        assert work and (await self.jobs.claim("BACKUP")) is None
        await self.jobs.execute(work)
        return self.repo.runs[work.run_id]

    async def claim_drill(self):
        self.now = datetime.fromisoformat(self.repo.plan.settings_json["drill_schedule"]["once_at"])
        return await self.jobs.claim("DRILL")


@pytest.mark.asyncio
@pytest.mark.parametrize("rules", [WEEKLY, ROTATION], ids=["cron", "rotation"])
@pytest.mark.parametrize("fault", ["none", "old_full", "different_slot", "wrong_binding", "no_base", "wrong_revision", "uncertified"])
async def test_configured_backup_chain_and_no_fallback(rules, fault):
    demo = Scenario("PGBACKREST")
    plan = demo.repo.plan
    plan.settings_json = ProtectionPlanSettings.model_validate({
        **plan.settings_json, "backup_schedule": None, **rules,
    }).model_dump(mode="json")
    full = datetime(2027, 1, 2, 16, tzinfo=UTC)  # 上海周日 00:00，UTC 仍是周六
    monday = datetime(2027, 1, 3, 17, 30, tzinfo=UTC)
    tuesday = monday + timedelta(days=1)
    labels = ("20270103-000100F", "20270103-000100F_20270104-013100I", "20270103-000100F_20270105-013100I")
    demo.now = full - timedelta(minutes=5)
    await demo.confirm("BACKUP")
    assert plan.next_backup_at == full

    async def fake_backup(work, invocation):
        index = (full, monday, tuesday).index(work.scheduled_at)
        kind = "INCREMENTAL" if index else "FULL"
        assert invocation.kind == kind
        assert ("--type=incr" if index else "--type=full") in invocation.arguments
        assert work.snapshot["backup_kind"] == kind
        deps = labels[:index]
        if index:
            assert work.backup.reference == deps[-1]
        return BackupResult(0, BackupArtifact("warehouse", labels[index], demo.now, True, kind, deps))

    demo.backend.backup.side_effect = fake_backup
    demo.now = full + timedelta(minutes=2)
    if fault != "no_base":
        work = await demo.jobs.claim("BACKUP")
        assert work.scheduled_at == full
        await demo.jobs.execute(work)
        base = demo.repo.runs[work.run_id]
        assert base.status == "SUCCEEDED"
        if fault == "old_full":
            base.scheduled_at -= timedelta(days=7)
        elif fault == "different_slot":
            base.scheduled_at += timedelta(minutes=30)
        elif fault == "wrong_binding":
            base.snapshot_json = {**base.snapshot_json, "approval": {"binding": {"tool": "PGBACKREST"}}}
        elif fault == "wrong_revision":
            base.snapshot_json["revision"] -= 1
        elif fault == "uncertified":
            base.result_json["artifact"]["checksum_verified"] = False
    else:
        plan.next_backup_at = monday
    assert plan.next_backup_at == monday
    demo.now = monday + timedelta(minutes=3)
    work = await demo.jobs.claim("BACKUP")
    assert plan.next_backup_at == tuesday
    if fault != "none":
        assert work is None
        rejected = next(r for r in demo.repo.runs.values() if r.scheduled_at == monday)
        assert rejected.status == "REJECTED" and rejected.snapshot_json["backup_kind"] == "INCREMENTAL"
        assert "未自动改跑全量" in rejected.result_json["message"]
        assert demo.backend.backup.await_count == (0 if fault == "no_base" else 1)
        return
    await demo.jobs.execute(work)
    assert demo.repo.runs[work.run_id].result_json["artifact"]["dependencies"] == [labels[0]]
    demo.now = tuesday
    work = await demo.jobs.claim("BACKUP")
    assert work.snapshot["baseline_run_id"] == str(base.run_id)
    await demo.jobs.execute(work)
    assert demo.repo.runs[work.run_id].result_json["artifact"]["dependencies"] == list(labels[:2])
    drill_at = tuesday + timedelta(hours=1)
    plan.settings_json["drill_schedule"] = {"timezone": "Asia/Shanghai", "once_at": drill_at.isoformat()}
    await demo.confirm("DRILL")
    demo.now = drill_at
    drill = await demo.jobs.claim("DRILL")
    assert drill.backup.reference == labels[-1] and drill.backup.dependencies == labels[:2]
    if rules == ROTATION:
        # 后一轮全量错过，不能把上一轮成功的增量或全量当作新一轮的基线。
        demo.now = full + timedelta(days=3, minutes=16)
        assert await demo.jobs.claim("BACKUP") is None
        assert plan.next_backup_at == monday + timedelta(days=3)
        demo.now = plan.next_backup_at
        assert await demo.jobs.claim("BACKUP") is None
        result = next(row for row in demo.repo.runs.values()
                      if row.scheduled_at == demo.now and row.operation == "BACKUP")
        assert result.status == "REJECTED" and result.snapshot_json["backup_kind"] == "INCREMENTAL"


def test_weekly_schedule_skips_repeated_wall_clock_hour():
    settings = ProtectionPlanSettings(tool="RMAN", **{
        key: {**value, "timezone": "America/New_York"} for key, value in WEEKLY.items()
    })
    settings = ProtectionPlanSettings.model_validate({
        **settings.model_dump(), "backup_schedule": {"timezone": "America/New_York", "cron": "30 1 * * 0"},
    })
    first = datetime(2027, 11, 7, 5, 30, tzinfo=UTC)
    assert next_backup_at(settings, first - timedelta(minutes=1)) == first
    assert next_backup_at(settings, first) == datetime(2027, 11, 8, 6, 30, tzinfo=UTC)


def test_daily_cron_rules_can_overlap_and_full_wins():
    settings = ProtectionPlanSettings(
        tool="RMAN",
        backup_schedule={"timezone": "UTC", "cron": "15 3 * * *"},
        backup_incremental_schedule={"timezone": "UTC", "cron": "15 3,18 * * *"},
    )
    full = datetime(2027, 6, 1, 3, 15, tzinfo=UTC)
    increment = full + timedelta(hours=15)
    assert next_backup_at(settings, full - timedelta(minutes=1)) == full
    assert scheduled_backup_kind(settings, full) == "FULL"
    assert next_backup_at(settings, full) == increment
    assert scheduled_backup_kind(settings, increment) == "INCREMENTAL"
    assert required_full_at(settings, increment) == full
    assert scheduled_backup_kind(settings, full + timedelta(minutes=1)) is None


def test_three_day_rotation_continues_across_month_and_year():
    settings = ProtectionPlanSettings(tool="RMAN", backup_rotation={
        "timezone": "Asia/Shanghai", "start_date": "2027-12-30", "cycle_days": 3,
        "full_time": "02:20", "incremental_time": "05:15",
    })
    at = datetime(2027, 12, 29, 17, tzinfo=UTC)
    expected = [
        ("2027-12-29T18:20:00+00:00", "FULL"),
        ("2027-12-30T21:15:00+00:00", "INCREMENTAL"),
        ("2027-12-31T21:15:00+00:00", "INCREMENTAL"),
        ("2028-01-01T18:20:00+00:00", "FULL"),
        ("2028-01-02T21:15:00+00:00", "INCREMENTAL"),
    ]
    for timestamp, kind in expected:
        at = next_backup_at(settings, at)
        assert at.isoformat() == timestamp and scheduled_backup_kind(settings, at) == kind
        if kind == "FULL":
            full = at
        else:
            assert required_full_at(settings, at) == full


def test_rotation_rejects_incremental_when_full_clock_time_does_not_exist():
    settings = ProtectionPlanSettings(tool="RMAN", backup_rotation={
        "timezone": "America/New_York", "start_date": "2027-03-14", "cycle_days": 3,
        "full_time": "02:30", "incremental_time": "02:00",
    })
    at = next_backup_at(settings, datetime(2027, 3, 13, tzinfo=UTC))
    assert at == datetime(2027, 3, 15, 6, tzinfo=UTC)
    assert required_full_at(settings, at) is None


@pytest.mark.parametrize(("changes", "message"), [
    ({"backup_schedule": None}, "同一时区"),
    ({"backup_incremental_schedule": {"timezone": "UTC", "cron": "30 1 * * 1-6"}}, "同一时区"),
    ({"backup_incremental_schedule": {"timezone": "Asia/Shanghai", "once_at": BASE}}, "周期任务"),
    (ROTATION, "不能与备份 Cron 同时设置"),
])
def test_invalid_mixed_backup_schedules(changes, message):
    with pytest.raises(ValueError, match=message):
        ProtectionPlanSettings.model_validate({"tool": "RMAN", **WEEKLY, **changes})


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "kind", "coordinate"), [
    ("RMAN", "SCN", "921300:1"),
    ("PGBACKREST", "TIMELINE_LSN", "3:0/4002FA"),
    ("XTRABACKUP", "BINLOG", "mysql-bin.000005:21472"),
])
async def test_scheduled_backup_drill_and_formal_report(tool, kind, coordinate):
    demo = Scenario(tool, kind, coordinate)
    await demo.confirm("DRILL")
    backup = await demo.complete_backup()
    assert backup.status == "SUCCEEDED" and backup.result_json["continuous_logs_verified"] is False
    work = await demo.claim_drill()
    assert work
    await demo.jobs.execute(work)
    row = demo.repo.runs[work.run_id]
    assert row.status == "SUCCEEDED" and row.active_key is None
    demo.runner.restore_and_verify.assert_awaited_once()

    await publish_next_drill_report(uow_factory=demo.uow_factory, now=demo.now)
    await publish_next_drill_report(uow_factory=demo.uow_factory, now=demo.now)
    demo.reports.add_report.assert_awaited_once()
    report = demo.reports.add_report.await_args.args[0]
    assert report.report_id == row.report_id and report.report_type == "RECOVERY_DRILL"
    raw, display = (call.args[0] for call in demo.run_repo.add_artifact.await_args_list)
    facts = raw.payload_json
    assert facts["coordinate"] == coordinate
    assert facts["rto"]["gap_max_seconds"] == 10
    assert facts["rpo"]["gap_max_seconds"] == 40
    assert facts["database_verified"] and not facts["application_verified"]
    assert facts["backup"]["continuous_logs_verified"] is False
    assert "/data/drill" not in str(facts)
    presentation = report_presentation(payload=display.payload_json, template=resolve_system_template("system:recovery-drill"))
    assert any("超标" in text for section in presentation["sections"] for text in section["items"])
    assert render_pdf(presentation).startswith(b"%PDF")


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", [
    "revoked", "nonempty", "changed", "incompatible", "backup_unreadable", "production", "overlap", "unisolated",
])
async def test_restore_not_called_when_any_runtime_fact_changes(condition):
    demo = Scenario("RMAN")
    await demo.confirm("DRILL")
    await demo.complete_backup()
    work = await demo.claim_drill()
    if condition == "revoked":
        demo.preflight_ok = False
    elif condition == "changed":
        demo.repo.plan.revision += 1
    else:
        facts = demo.runner.probe_drill.return_value
        prop = {
            "nonempty": "empty_and_idle", "incompatible": "tool_compatible", "backup_unreadable": "backup_readable",
            "production": "dedicated_drill_environment", "overlap": "no_production_overlap", "unisolated": "isolated_from_production",
        }[condition]
        demo.runner.probe_drill.return_value = replace(facts, **{prop: False})
    await demo.jobs.execute(work)
    row = demo.repo.runs[work.run_id]
    assert row.status == "REJECTED" and "sensitive-backend-error" not in str(row.result_json)
    demo.runner.restore_and_verify.assert_not_awaited()
    await publish_next_drill_report(uow_factory=demo.uow_factory, now=demo.now)
    assert row.report_id


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "version", "kind", "coordinate"), [
    ("RMAN", "19c", "SCN", "921300:1"),
    ("RMAN", "26ai", "SCN", "921300:1"),
    ("PGBACKREST", "16", "TIMELINE_LSN", "3:0/4002FA"),
    ("PGBACKREST", "17", "TIMELINE_LSN", "3:0/4002FA"),
    ("PGBACKREST", "18", "TIMELINE_LSN", "3:0/4002FA"),
    ("XTRABACKUP", "5.7", "BINLOG", "mysql-bin.000005:21472"),
    ("XTRABACKUP", "8.0", "BINLOG", "mysql-bin.000005:21472"),
    ("XTRABACKUP", "8.4", "BINLOG", "mysql-bin.000005:21472"),
])
async def test_native_templates_feed_offline_report_without_inventing_rpo(tool, version, kind, coordinate):
    demo = Scenario(tool, kind, coordinate)
    demo.db_version = version
    preset = restore_preset(tool, version, version)
    demo.repo.plan.settings_json["restore_template_key"] = preset.key

    async def preflight(**kwargs):
        approval = await demo.preflight(**kwargs)
        return replace(approval, template_key=preset.key, template_revision=preset.revision) if kwargs["operation"] == "DRILL" else approval

    async def backup(work, invocation):
        reference = "20261118-153010F" if tool == "PGBACKREST" else f"aiops-{work.run_id}"
        return BackupResult(0, BackupArtifact("warehouse", reference, demo.now, True))

    demo.backend.preflight.side_effect = preflight
    demo.backend.backup.side_effect = backup
    await demo.confirm("DRILL")
    await demo.complete_backup()
    work = await demo.claim_drill()
    invoked = AsyncMock()

    async def run_offline(request):
        directory = f"/data/backups/{request.backup.reference}"
        media = {
            "RMAN": OracleMedia(directory, "DRILL01"),
            "PGBACKREST": PostgreSQLMedia("/etc/pgbackrest.conf", "/data/backups", "/data/tablespaces-drill"),
            "XTRABACKUP": MySQLMedia(directory, f"/data/drill-stage/aiops-{work.run_id}"),
        }[tool]
        invocation = render_restore(
            ProtectionPlanSettings.model_validate(work.snapshot["settings"]), request,
            db_version=version, target_version=version, run_id=work.run_id, media=media,
        )
        await invoked(invocation)
        demo.now += timedelta(seconds=25)
        return DrillVerification(True, demo.now, coordinate, None, recovery_coordinate_kind=kind)

    demo.runner.restore_and_verify.side_effect = run_offline
    await demo.jobs.execute(work)
    steps = invoked.await_args.args[0].steps
    if tool == "RMAN":
        assert steps[0].arguments == ("auxiliary",)
        assert "BACKUP LOCATION" in steps[0].command_file and "NOOPEN" in steps[0].command_file
        assert "NOFILENAMECHECK" not in steps[0].command_file and "FROM ACTIVE" not in steps[0].command_file
    elif tool == "PGBACKREST":
        assert "--set=20261118-153010F" in steps[0].arguments
        assert "--target-action=pause" in steps[0].arguments and "--no-delta" in steps[0].arguments
    else:
        copy = f"--target-dir=/data/drill-stage/aiops-{work.run_id}"
        assert copy in steps[0].arguments and "--prepare" in steps[0].arguments
        assert copy in steps[1].arguments and "--copy-back" in steps[1].arguments
        assert not any(request_arg.startswith("--force") for step in steps for request_arg in step.arguments)
    await publish_next_drill_report(uow_factory=demo.uow_factory, now=demo.now)
    evidence = demo.run_repo.add_artifact.await_args_list[0].args[0].payload_json
    assert evidence["template_key"] == preset.key and evidence["coordinate"] == coordinate
    assert evidence["rto"]["gap_max_seconds"] == 10 and evidence["rpo"]["verdict"] == "NOT_MEASURED"
    assert demo.reports.add_report.await_args.args[0].status == "PARTIAL"


@pytest.mark.asyncio
async def test_drill_schedule_requires_restore_identity_and_approved_preflight_and_pauses_only_drills():
    demo = Scenario("RMAN")
    plan = demo.repo.plan
    target = SimpleNamespace(db_type="ORACLE", version_code="19c", db_role="PRIMARY")
    service = DataProtectionConfigurationMixin()
    service._get_database_resource_and_target = AsyncMock(return_value=(None, target))
    service._uow_factory = demo.uow_factory
    demo.uow.managed_credentials = SimpleNamespace(find=AsyncMock(return_value=None))
    demo.repo.history = AsyncMock(return_value=())
    plan.settings_json["drill_schedule"]["once_at"] = None
    plan.settings_json["drill_schedule"]["cron"] = "0 4 * * *"
    plan.approval_json = {"BACKUP": {"prior": True}}
    plan.next_backup_at = datetime.now(UTC) + timedelta(hours=1)
    scope = SimpleNamespace(**SCOPE, actor_id=ACTOR)
    request = ProtectionDrillScheduleChange(expected_revision=plan.revision, enabled=True)
    with pytest.raises(AIOpsApplicationError, match="演练专用身份"):
        await service.change_data_protection_drill_schedule(scope=scope, resource_id=plan.resource_id, request=request)
    demo.uow.managed_credentials.find.assert_awaited_once()
    assert demo.uow.managed_credentials.find.await_args.kwargs["credential_kind"] == "protection_restore"
    demo.uow.managed_credentials.find.return_value = SimpleNamespace(status="ACTIVE")
    with pytest.raises(AIOpsApplicationError, match="预检未通过"):
        await service.change_data_protection_drill_schedule(scope=scope, resource_id=plan.resource_id, request=request)
    assert plan.next_drill_at is None
    service._protection_backend = demo.backend
    view = await service.change_data_protection_drill_schedule(scope=scope, resource_id=plan.resource_id, request=request)
    assert view.next_drill_at and view.next_backup_at == plan.next_backup_at
    assert demo.backend.preflight.await_args.kwargs["operation"] == "DRILL"
    view = await service.change_data_protection_drill_schedule(
        scope=scope, resource_id=plan.resource_id, request=request.model_copy(update={"enabled": False}),
    )
    assert view.next_drill_at is None and view.next_backup_at == plan.next_backup_at
    assert plan.approval_json == {"BACKUP": {"prior": True}}
    demo.runner.restore_and_verify.assert_not_awaited()


@pytest.mark.parametrize(("fault", "code"), [
    ("version", "BINDING_CHANGED"), ("template", "BINDING_CHANGED"), ("destination", "BINDING_CHANGED"),
    ("future", "BACKUP_UNAVAILABLE"), ("unverified", "BACKUP_UNAVAILABLE"), ("wrong_backup", "BACKUP_UNAVAILABLE"),
    ("source_as_staging", "DESTINATION_CHANGED"), ("occupied_path", "DESTINATION_CHANGED"),
    ("different_run", "DESTINATION_CHANGED"), ("path_jump", "DESTINATION_CHANGED"),
])
def test_native_restore_template_rejects_unbound_media(fault, code):
    run_id = uuid7()
    reference = f"aiops-{uuid7()}"
    preset = restore_preset("XTRABACKUP", "8.4", "8.4")
    settings = ProtectionPlanSettings(
        tool="XTRABACKUP", storage_key="warehouse", backup_directory="/data/backups",
        restore_template_key=preset.key, drill_destination_key=DEST.key,
    )
    request = DrillRequest("XTRABACKUP", DEST, "warehouse", BackupArtifact("warehouse", reference, BASE, True), BASE)
    media = MySQLMedia(f"/data/backups/{reference}", f"/data/drill-stage/aiops-{run_id}")
    version = "8.4"
    if fault == "version":
        version = "8.0"
    elif fault in {"template", "destination"}:
        field = "restore_template_key" if fault == "template" else "drill_destination_key"
        settings = settings.model_copy(update={field: "other"})
    elif fault == "future":
        request = replace(request, backup=replace(request.backup, available_at=BASE + timedelta(seconds=1)))
    elif fault == "unverified":
        request = replace(request, backup=replace(request.backup, checksum_verified=False))
    elif fault == "wrong_backup":
        media = replace(media, backup_directory=f"/data/backups/aiops-{uuid7()}")
    elif fault == "source_as_staging":
        media = replace(media, staging_directory=f"/data/backups/aiops-{run_id}")
    elif fault == "occupied_path":
        request = replace(request, destination=replace(DEST, data_path="/data/backups/db"))
    elif fault == "different_run":
        media = replace(media, staging_directory=f"/data/drill-stage/aiops-{uuid7()}")
    else:
        request = replace(request, destination=replace(DEST, data_path="/data/drill/../backups"))
    with pytest.raises(DrillRejected) as error:
        render_restore(settings, request, db_version="8.4", target_version=version, run_id=run_id, media=media)
    assert error.value.code == code


@pytest.mark.asyncio
async def test_success_exit_without_artifact_and_no_restore_before_t0():
    demo = Scenario("RMAN")
    await demo.confirm("DRILL")
    demo.backend.backup.side_effect = None
    demo.backend.backup.return_value = BackupResult(0, None)
    backup = await demo.complete_backup()
    assert backup.status == "FAILED"
    assert await demo.claim_drill() is None
    assert list(demo.repo.runs.values())[-1].status == "REJECTED"
    demo.runner.restore_and_verify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["scheduled", "rejected", "cancelled"])
async def test_api_registers_request_and_only_designated_worker_enables_drill(outcome):
    demo = Scenario("RMAN")
    plan = demo.repo.plan
    demo.backend.drill_resource_ids = frozenset({plan.resource_id})
    api = ProtectionJobs(uow_factory=demo.uow_factory, backend=None, clock=lambda: demo.now)
    await api.request_drill(
        scope=SCOPE, resource_id=plan.resource_id, revision=plan.revision, actor_id=ACTOR, db_version="19c",
    )
    assert plan.approval_json["DRILL_REQUEST"]["status"] == "PENDING" and plan.next_drill_at is None
    demo.backend.preflight.assert_not_awaited()

    async def load(ids):
        assert ids == {plan.resource_id}
        return plan

    demo.repo.pending_drill = load
    if outcome == "cancelled":
        async def preflight(**kwargs):
            await api.pause_drill(scope=SCOPE, resource_id=plan.resource_id, revision=plan.revision)
            return await demo.preflight(**kwargs)

        demo.backend.preflight.side_effect = preflight
    elif outcome == "rejected":
        demo.preflight_ok = False

    await api.confirm_pending_drill()
    demo.backend.preflight.assert_not_awaited()
    await demo.jobs.confirm_pending_drill()
    assert demo.backend.preflight.await_args.kwargs["operation"] == "DRILL"
    if outcome == "scheduled":
        assert plan.next_drill_at is not None and "DRILL_REQUEST" not in plan.approval_json
    elif outcome == "rejected":
        assert plan.next_drill_at is None and plan.approval_json["DRILL_REQUEST"]["status"] == "REJECTED"
    else:
        assert plan.next_drill_at is None and plan.approval_json is None
    demo.runner.restore_and_verify.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_and_restart_retain_destination_lock():
    demo = Scenario("RMAN")
    await demo.confirm("DRILL")
    backup = await demo.complete_backup()
    second = demo.repo.add_plan(ProtectionPlanSettings.model_validate(demo.repo.plan.settings_json))
    await demo.confirm("DRILL", second)
    second_backup = SimpleNamespace(**{k: getattr(backup, k) for k in (
        "operation", "completed_at", "status", "result_json", "snapshot_json",
    )}, run_id=uuid7(), resource_id=second.resource_id, active_key=None, scheduled_at=backup.scheduled_at)
    demo.repo.runs[second_backup.run_id] = second_backup
    first = await demo.claim_drill()
    assert first
    demo.now = first.deadline_at
    restarted = ProtectionJobs(uow_factory=demo.uow_factory, backend=demo.backend, clock=lambda: demo.now)
    await restarted.reconcile()
    assert demo.repo.runs[first.run_id].status == "UNKNOWN"
    assert demo.repo.runs[first.run_id].active_key
    assert await restarted.claim("DRILL") is None
    assert list(demo.repo.runs.values())[-1].status == "SKIPPED"
    demo.runner.restore_and_verify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError, DrillRejected])
async def test_unknown_write_state_report_retry_does_not_execute(failure):
    demo = Scenario("RMAN")
    await demo.confirm("DRILL")
    await demo.complete_backup()
    work = await demo.claim_drill()
    demo.runner.restore_and_verify.side_effect = failure
    await demo.jobs.execute(work)
    row = demo.repo.runs[work.run_id]
    assert row.status == "UNKNOWN" and row.active_key
    demo.reports.add_report.side_effect = RuntimeError
    with pytest.raises(RuntimeError):
        await publish_next_drill_report(uow_factory=demo.uow_factory, now=demo.now)
    assert not row.report_id
    demo.reports.add_report.side_effect = None
    await publish_next_drill_report(uow_factory=demo.uow_factory, now=demo.now)
    assert row.report_id
    demo.runner.restore_and_verify.assert_awaited_once()


@pytest.mark.asyncio
async def test_unconfigured_worker_never_claims_and_missed_window_is_not_replayed():
    demo = Scenario("RMAN")
    await demo.confirm("DRILL")
    disabled = ProtectionJobs(uow_factory=demo.uow_factory, backend=None, clock=lambda: demo.now)
    assert await disabled.claim("BACKUP") is None
    demo.now = BASE + timedelta(hours=1)
    assert await demo.jobs.claim("DRILL") is None
    assert list(demo.repo.runs.values())[-1].status == "SKIPPED"
    assert await demo.jobs.claim("DRILL") is None


@pytest.mark.asyncio
async def test_concurrent_workers_claim_a_fire_once():
    demo = Scenario("RMAN")
    await demo.confirm("BACKUP")
    demo.now = BASE + timedelta(minutes=1)
    claims = await asyncio.gather(demo.jobs.claim("BACKUP"), demo.jobs.claim("BACKUP"))
    assert sum(claim is not None for claim in claims) == 1
    assert len(demo.repo.runs) == 1


@pytest.mark.asyncio
async def test_pause_backup_is_scoped_and_leaves_drill_schedule_and_history():
    demo = Scenario("RMAN")
    await demo.confirm("DRILL")
    await demo.complete_backup()
    plan = demo.repo.plan
    next_drill = plan.next_drill_at
    with pytest.raises(DrillRejected):
        await demo.jobs.pause_backup(scope=SCOPE, resource_id=plan.resource_id, revision=0)
    await demo.jobs.pause_backup(scope=SCOPE, resource_id=plan.resource_id, revision=plan.revision)
    assert plan.next_backup_at is None and "BACKUP" not in plan.approval_json
    assert plan.next_drill_at == next_drill and len(demo.repo.runs) == 1


@pytest.mark.asyncio
async def test_stale_template_or_unavailable_backend_cannot_enable_backup():
    demo = Scenario("RMAN")
    plan = demo.repo.plan
    unconfigured = ProtectionJobs(uow_factory=demo.uow_factory, backend=None, clock=lambda: demo.now)
    with pytest.raises(DrillRejected):
        await unconfigured.confirm(scope=SCOPE, resource_id=plan.resource_id, revision=plan.revision,
                                   actor_id=ACTOR, operation="BACKUP")
    demo.backend.preflight.side_effect = None
    demo.backend.preflight.return_value = ApprovedProtection(TARGET, 1, "RMAN", "rman.oracle19c.backup", "outdated", "warehouse", "19c")
    with pytest.raises(DrillRejected):
        await demo.confirm("BACKUP")
    assert plan.next_backup_at is None and plan.approval_json is None
    demo.backend.backup.assert_not_awaited()


@pytest.mark.asyncio
async def test_changed_db_version_never_runs_old_backup_preset():
    demo = Scenario("XTRABACKUP")
    await demo.confirm("BACKUP")
    demo.now = BASE + timedelta(minutes=1)
    work = await demo.jobs.claim("BACKUP")
    assert work.snapshot["approval"]["binding"]["db_version"] == "5.7"
    demo.db_version = "8.4"
    await demo.jobs.execute(work)
    assert demo.repo.runs[work.run_id].status == "REJECTED"
    demo.backend.backup.assert_not_awaited()

    # 重新启用后才可使用新版本的预设。
    await demo.confirm("BACKUP")
    demo.now = demo.repo.plan.next_backup_at
    new_work = await demo.jobs.claim("BACKUP")
    await demo.jobs.execute(new_work)
    assert demo.repo.runs[new_work.run_id].status == "SUCCEEDED"
    invocation = demo.backend.backup.await_args.args[1]
    assert invocation.preset.tool_requirement == "Percona XtraBackup 8.4"


@pytest.mark.asyncio
async def test_drill_does_not_choose_backup_from_another_database_version():
    demo = Scenario("XTRABACKUP")
    await demo.complete_backup()
    demo.db_version = "8.4"
    await demo.confirm("DRILL")
    assert await demo.claim_drill() is None
    assert list(demo.repo.runs.values())[-1].status == "REJECTED"
    demo.backend.drill_runner.assert_not_awaited()

