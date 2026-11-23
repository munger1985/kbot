"""独立演练主机本机恢复；使用假工具、临时目录和假验证连接。"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters import native_drill
from aiops_agent.adapters.local_drill import LocalDrillBackend
from aiops_agent.application.data_protection_execution import BackupArtifact, DrillRejected, DrillRequest, execute_isolated_drill
from aiops_agent.application.data_protection_jobs import ProtectionWork
from aiops_agent.application.data_protection_restore import restore_preset
from aiops_agent.bootstrap import data_protection, worker
from aiops_agent.config import (
    AIOpsLocalMySQLDrillConfig, AIOpsLocalOracleDrillConfig, AIOpsLocalPostgreSQLDrillConfig, AIOpsSettings,
)
from platform_core.config import load_settings
from platform_core.contracts.aiops.data_protection import ProtectionPlanSettings
from platform_core.identity import uuid7


_MACHINE = "1" * 32
_PG_LABEL = "20261118-094500F"
_GTID = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff:1-42"
_DB_TYPES = {"RMAN": "ORACLE", "PGBACKREST": "POSTGRESQL", "XTRABACKUP": "MYSQL"}


@pytest.fixture
def exercise(tmp_path, monkeypatch):
    def build(tool, version):
        stage, repo, home = (tmp_path / path for path in ("exercise", "repo", "programs"))
        for path in (stage, repo, home, home / "bin"):
            path.mkdir(mode=0o700)
        for name in ("data", *native_drill.workspace_directories(tool)):
            (stage / name).mkdir(mode=0o700)
        data = stage / "data"
        target_id, resource_id, backup_id = uuid7(), uuid7(), uuid7()
        scope = {"domain_id": uuid7()}
        reference = _PG_LABEL if tool == "PGBACKREST" else f"aiops-{backup_id}"
        common = dict(
            source_target_id=target_id, machine_id=_MACHINE, os_user="drill", tool=tool,
            destination_key="exercise", drill_root=str(stage), backup_root=str(repo), required_free_bytes=1,
            verification_objects=("business.control_table",),
        )
        settings = ProtectionPlanSettings(
            tool=tool, restore_template_key=restore_preset(tool, version, version).key,
            drill_destination_key="exercise", storage_key="warehouse",
            backup_directory=None if tool == "PGBACKREST" else str(repo),
        )
        target = SimpleNamespace(
            db_type=_DB_TYPES[tool], version_code=version, status="ENABLED", row_version=1, tenant_id=scope.get("tenant_id"),
        )
        row = SimpleNamespace(status="ACTIVE", credential_id=uuid7(), nonce=b"restore-nonce")
        uow = SimpleNamespace(
            business_systems=SimpleNamespace(
                get_resource_scoped=AsyncMock(return_value=SimpleNamespace(resource_type_key="DATABASE", lifecycle_status="ACTIVE")),
                get_database_profile=AsyncMock(return_value=SimpleNamespace(target_id=target_id)),
            ),
            targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target)),
            managed_credentials=SimpleNamespace(find=AsyncMock(return_value=row)),
        )

        @asynccontextmanager
        async def uow_factory():
            yield uow

        secrets = SimpleNamespace(read=AsyncMock(return_value={"username": "verify_user", "password": "verify-only!"}))
        calls = []
        state = SimpleNamespace(
            settings=settings, target=target, uow=uow, row=row, calls=calls, approved=True, readonly=True, space=True,
            stage=stage, repo=repo, data=data, scope=scope, resource_id=resource_id, reference=reference, run_id=uuid7(),
        )

        def tool_file(name):
            binary = home / "bin" / name
            binary.touch()
            binary.chmod(0o700)
            return str(binary)

        async def run(executable, args, **kwargs):
            assert "verify-only!" not in repr(args)
            calls.append((Path(executable).name, args))
            if tool == "PGBACKREST":
                if args == ("--version",):
                    return (0, f"postgres (PostgreSQL) {version}.5") if executable.endswith("/postgres") else (0, "pgBackRest 2.55.0")
                if args[-1] == "info":
                    return 0, json.dumps([{
                        "name": "warehouse", "status": {"code": 0},
                        "repo": [{"key": 1, "status": {"code": 0}}],
                        "db": [{"id": 1, "repo-key": 1, "system-id": 1345, "version": version}],
                        "backup": [{"label": reference, "type": "full", "error": False,
                                    "database": {"id": 1, "repo-key": 1}, "info": {"size": 100},
                                    "timestamp": {"start": 50, "stop": 90},
                                    "archive": {"start": "00000001000000000000000A",
                                                "stop": "00000001000000000000000B"}},
                                   *getattr(state, "incr_catalog", [])],
                    }])
                assert args[-1] == "restore" and f"--pg1-path={data}" in args and "--target-action=pause" in args
                (data / "recovery.signal").touch()
                (data / "postgresql.auto.conf").write_text("shared_preload_libraries='source_plugin'\n")
                return 0, ""
            if args[-1] == "--version":
                family = {"5.7": "2.4", "8.0": "8.0", "8.4": "8.4"}[version]
                return (0, f"xtrabackup version {family}.1") if executable.endswith("/xtrabackup") else (0, f"mysqld  Ver {version}.12")
            assert tool == "XTRABACKUP"
            copy = stage / "stage" / f"aiops-{state.run_id}"
            assert f"--target-dir={copy}" in args and f"--defaults-file={copy}/backup-my.cnf" in args
            assert not (copy / "auto.cnf").exists()
            if "--copy-back" in args:
                assert f"--datadir={data}" in args
                (data / "ibdata1").write_bytes(b"mock")
            else:
                assert "--prepare" in args
            return 0, ""

        async def run_oracle(config, executable, args, script):
            calls.append((executable, args))
            assert "verify-only!" not in repr(args) and " AS SYSBACKUP" in script
            if executable == "rman":
                assert script.startswith("CONNECT AUXILIARY 'verify_user/\"verify-only!\" AS SYSBACKUP';")
                assert "NOOPEN" in script and "NOFILENAMECHECK" not in script and "CONNECT TARGET" not in script
                if getattr(state, "incremental_artifact", None):
                    work = stage / f"aiops-{state.run_id}"
                    assert str(work) in script
                    assert {p.name for p in work.iterdir()} == {"a.bkp", "incr.bkp"}
                else:
                    assert str(repo / reference) in script
                (data / "system.dbf").write_bytes(b"mock")
                return 0, ""
            if "select 'DRILL|INSTANCE|'" in script:
                return 0, "\n".join((
                    f"DRILL|INSTANCE|drill01|STARTED|{'19.16.0.0.0' if version == '19c' else '23.26.1.0.0'}",
                    "DRILL|db_name|DRILL01", "DRILL|db_unique_name|DRILL01", f"DRILL|db_create_file_dest|{data}",
                    "DRILL|cluster_database|FALSE", "DRILL|dg_broker_start|FALSE", "DRILL|job_queue_processes|0",
                    f"DRILL|control_files|{data}/control.ctl", f"DRILL|diagnostic_dest|{stage}/diag",
                    f"DRILL|audit_file_dest|{stage}/audit",
                ))
            if "open resetlogs" in script:
                assert "alter system enable restricted session" in script
                return 0, ""
            if "from v$database d" in script:
                return 0, "DRILL|DRILL01|READ WRITE|1234|2"
            if "from dba_tables" in script:
                assert "where owner='BUSINESS' and table_name='CONTROL_TABLE'" in script
                return 0, "DRILL|1"
            return 0, f"DRILL|{data}/system.dbf"

        @asynccontextmanager
        async def fake_start(binary, args, socket):
            calls.append((Path(binary).name, args))
            assert socket.parent == stage / "run"
            assert "verify-only!" not in repr(args)
            if tool == "PGBACKREST":
                assert f"config_file={stage}/run/postgresql.conf" in args
                assert (data / "postgresql.auto.conf").read_text() == ""
                assert "listen_addresses=''" in (stage / "run" / "postgresql.conf").read_text()
            else:
                assert "--skip-networking" in args and "--event-scheduler=OFF" in args
                assert "--skip-log-bin" in args and "--gtid-mode=ON" in args
                assert "--skip-replica-start" in args or "--skip-slave-start" in args
            yield

        monkeypatch.setattr(native_drill, "_start_local", fake_start)
        if tool == "RMAN":
            tool_file("sqlplus")
            tool_file("rman")
            config = AIOpsLocalOracleDrillConfig(
                **common, oracle_home=str(home), oracle_sid="drill01", auxiliary_database="DRILL01",
            )
            (stage / "diag").mkdir()
            (stage / "audit").mkdir()
            artifact = repo / reference
            artifact.mkdir()
            (artifact / "a.bkp").write_bytes(b"mock")
        elif tool == "PGBACKREST":
            tool_file("postgres")
            config = AIOpsLocalPostgreSQLDrillConfig(
                **common, postgres_bin=str(home / "bin"), executable=tool_file("pgbackrest"), system_identifier="1345",
            )
            (repo / "backup" / "warehouse" / reference).mkdir(parents=True)
            (repo / "archive" / "warehouse").mkdir(parents=True)
            live = {
                "directory": str(data), "paused": True, "timeline": 1, "lsn": "0/123A",
                "replay_time": datetime.now(UTC) - timedelta(seconds=30), "objects": 3,
            }
            pg_connection = SimpleNamespace(
                fetchrow=AsyncMock(return_value=live), fetchval=AsyncMock(side_effect=["r", 1]), close=AsyncMock(),
            )
            monkeypatch.setattr(native_drill.asyncpg, "connect", AsyncMock(return_value=pg_connection))
        else:
            config = AIOpsLocalMySQLDrillConfig(
                **common, executable=tool_file("xtrabackup"), mysqld_executable=tool_file("mysqld"), server_id=403,
            )
            artifact = repo / reference
            artifact.mkdir()
            (artifact / "ibdata1").write_bytes(b"mock")
            (artifact / "auto.cnf").write_text("original")
            (artifact / "xtrabackup_logfile").write_bytes(b"mock")
            (artifact / "xtrabackup_checkpoints").write_text("backup_type = full-backuped\nfrom_lsn = 0\n")
            (artifact / "xtrabackup_binlog_info").write_text(f"mysql-bin.000004\t789\t{_GTID}\n")
            (artifact / "backup-my.cnf").write_text(
                "[mysqld]\ninnodb_data_file_path=ibdata1:12M:autoextend\ninnodb_page_size=16384\n"
                "innodb_log_checksums=1\ninnodb_log_checksums=ON\nmaster_key_id=0\n",
            )

            @asynccontextmanager
            async def cursor():
                yield SimpleNamespace(
                    execute=AsyncMock(), fetchone=AsyncMock(side_effect=[
                        (str(data) + "/", version + ".12", 403, _GTID, True), (2,), ("BASE TABLE",),
                    ]),
                )

            monkeypatch.setattr(native_drill.aiomysql, "connect", AsyncMock(return_value=SimpleNamespace(cursor=cursor, close=lambda: None)))

        def verify(config, reader):
            if not state.approved or reader() != config.machine_id:
                raise DrillRejected("主机不符", code="ISOLATION_UNVERIFIED")

        state.config = config
        state.backend = LocalDrillBackend(
            bindings={resource_id: config}, uow_factory=uow_factory, credentials=secrets, run=run, run_oracle=run_oracle,
            read_machine=lambda: _MACHINE, verify_user=verify, read_only_mount=lambda path: state.readonly,
        )
        return state

    return build


async def _prepare(state):
    async def approval():
        return await state.backend.preflight(
            scope=state.scope, actor_id="operator", resource_id=state.resource_id, settings=state.settings, operation="DRILL",
        )

    approved = await approval()
    now = datetime.now(UTC)
    artifact = getattr(state, "incremental_artifact", None) or BackupArtifact(
        approved.storage_key, state.reference, now - timedelta(seconds=60), True,
    )
    work = ProtectionWork(
        state.run_id, state.resource_id, state.scope, "DRILL", now, now + timedelta(hours=1),
        {"settings": state.settings.model_dump(mode="json"),
         "approval": {"binding": {**asdict(approved), "target_id": str(approved.target_id)}}},
        artifact,
    )
    runner = await state.backend.drill_runner(work)
    return runner, DrillRequest(state.config.tool, approved.destination, approved.storage_key, artifact, now)


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "version", "fault"), [
    ("RMAN", "26ai", "none"), ("RMAN", "26ai", "broken"), ("RMAN", "26ai", "older_scn"),
    ("PGBACKREST", "18", "none"), ("PGBACKREST", "18", "broken"),
    ("XTRABACKUP", "8.4", "none"), ("XTRABACKUP", "8.4", "broken"),
])
async def test_incremental_drill_uses_complete_readonly_chain(exercise, tool, version, fault):
    state = exercise(tool, version)
    broken = fault == "broken"
    base = state.reference
    now = datetime.now(UTC) - timedelta(minutes=2)
    kind = {"kind": "INCREMENTAL", "dependencies": (base,)}
    if tool == "PGBACKREST":
        label = base + "_20261119-094500I"
        state.incr_catalog = [{
            "label": label, "type": "incr", "prior": "wrong" if broken else base, "reference": [base],
            "database": {"id": 1, "repo-key": 1}, "info": {"size": 110},
            "timestamp": {"start": 100, "stop": 120},
            "archive": {"start": "00000001000000000000000B", "stop": "00000001000000000000000C"},
        }]
        (state.repo / "backup" / state.settings.storage_key / label).mkdir()
    else:
        label = f"aiops-{uuid7()}"
        target = state.repo / label
        target.mkdir()
        if tool == "RMAN":
            (target / "incr.bkp").write_bytes(b"mock")
            kind["minimum_scn"] = 2000 if fault == "older_scn" else 1233
            kind["baseline_fingerprint"] = native_drill.oracle_local_backup.piece_manifest(state.repo / base)
            kind["fingerprint"] = native_drill.oracle_local_backup.piece_manifest(target)
            if broken:
                (target / "incr.bkp").write_bytes(b"truncated")
        else:
            full = state.repo / base
            (full / "xtrabackup_checkpoints").write_text(
                "backup_type=full-backuped\nfrom_lsn=0\nto_lsn=100\nlast_lsn=120\n",
            )
            (target / "xtrabackup_checkpoints").write_text(
                "backup_type=incremental\nfrom_lsn=100\nto_lsn=200\nlast_lsn=220\n",
            )
            (target / "ibdata1.delta").write_bytes(b"mock delta")
            (target / "ibdata1.meta").write_text("page_size=16384\n")
            (target / "xtrabackup_info").write_text("incremental=Y\n")
            (target / "xtrabackup_logfile").write_bytes(b"mock redo")
            (target / "backup-my.cnf").write_bytes((full / "backup-my.cnf").read_bytes())
            (target / "xtrabackup_binlog_info").write_text(f"mysql-bin.000005\t799\t{_GTID}\n")
            kind.update(start_lsn=100, end_lsn=200, fingerprint=native_drill.mysql_local_backup.incremental_fingerprint(target))
            if broken:
                (target / "ibdata1.delta").write_bytes(b"tampered")
    state.incremental_artifact = BackupArtifact(state.settings.storage_key, label, now, True, **kind)
    runner, request = await _prepare(state)
    before_write = AsyncMock()
    if broken:
        with pytest.raises(DrillRejected) as rejected:
            await execute_isolated_drill(request, runner, before_write=before_write)
        assert rejected.value.code == "BACKUP_UNAVAILABLE"
        before_write.assert_not_awaited()
        assert not any(state.data.iterdir())
        return
    result = await execute_isolated_drill(request, runner, before_write=before_write)
    assert result.database_verified == (fault != "older_scn")
    before_write.assert_awaited_once()
    if tool == "PGBACKREST":
        assert any(f"--set={label}" in args for _, args in state.calls)
    if tool == "XTRABACKUP":
        operations = [args for _, args in state.calls if "--prepare" in args or "--copy-back" in args]
        assert len(operations) == 3 and "--apply-log-only" in operations[0]
        assert any(arg.endswith(f"aiops-{state.run_id}-incremental") and arg.startswith("--incremental-dir=")
                   for arg in operations[1])
        assert "--copy-back" in operations[2]
        assert (state.repo / label / "ibdata1.delta").read_bytes() == b"mock delta"


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "version", "coordinate"), [
    ("RMAN", "19c", "1234:2"), ("RMAN", "26ai", "1234:2"),
    ("PGBACKREST", "16", "1:0/123A"), ("PGBACKREST", "17", "1:0/123A"), ("PGBACKREST", "18", "1:0/123A"),
    ("XTRABACKUP", "5.7", _GTID), ("XTRABACKUP", "8.0", _GTID), ("XTRABACKUP", "8.4", _GTID),
])
async def test_local_worker_drill_uses_only_bound_native_tools(exercise, tool, version, coordinate):
    state = exercise(tool, version)
    runner, request = await _prepare(state)
    result = await execute_isolated_drill(request, runner)
    assert result.database_verified and result.recovery_coordinate == coordinate
    assert result.recovered_at is None if tool != "PGBACKREST" else result.recovered_at is not None
    evidence = result.evidence(tool, not_before=request.simulated_failure_at, not_after=datetime.now(UTC))
    assert evidence["database_verified"] and evidence["recovery_coordinate"] == coordinate
    assert (evidence["recovered_at_earliest"] is not None) == (tool == "PGBACKREST")
    assert all(call.kwargs["credential_kind"] == "protection_restore" for call in state.uow.managed_credentials.find.await_args_list)
    assert "verify-only!" not in repr(result)
    if tool == "XTRABACKUP":
        assert (state.repo / state.reference / "auto.cnf").read_text() == "original"
        assert "innodb_log_checksums=1" in (state.repo / state.reference / "backup-my.cnf").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["host", "writeable", "nonempty", "asset", "credential", "directory", "backup_link", "backup_changed"])
async def test_local_worker_refuses_unsafe_or_changed_context(exercise, condition):
    state = exercise("XTRABACKUP", "8.4")
    runner, request = await _prepare(state)
    await runner.probe_drill(request)
    if condition == "host":
        state.approved = False
    elif condition == "writeable":
        state.readonly = False
    elif condition == "nonempty":
        (state.data / "current.ibd").touch()
    elif condition == "asset":
        state.target.row_version += 1
    elif condition == "credential":
        state.row.nonce = b"changed"
    elif condition == "directory":
        state.data.rename(state.stage / "old")
        state.data.mkdir(mode=0o700)
    elif condition == "backup_link":
        (state.repo / state.reference / "link").symlink_to(state.repo, target_is_directory=True)
    else:
        request = replace(request, backup=replace(request.backup, reference=f"aiops-{uuid7()}"))
    with pytest.raises(DrillRejected):
        await runner.restore_and_verify(request)
    assert all(args[-1] == "--version" for _, args in state.calls)
    assert not list((state.stage / "stage").iterdir())


def test_only_registered_worker_assembles_drills(exercise, monkeypatch):
    state = exercise("RMAN", "19c")
    settings = SimpleNamespace(oracle_local_backups={}, file_local_backups={}, local_drills={state.resource_id: state.config})
    monkeypatch.setattr(data_protection, "machine_id", lambda: None)
    assert data_protection.local_protection_backend(settings=settings, uow_factory=None, credentials=None, worker=True) is None
    monkeypatch.setattr(data_protection, "machine_id", lambda: _MACHINE)
    assert data_protection.local_protection_backend(settings=settings, uow_factory=None, credentials=None) is None
    backend = data_protection.local_protection_backend(settings=settings, uow_factory=None, credentials=None, worker=True)
    assert backend.drill_resource_ids == {state.resource_id} and not backend.backup_resource_ids


@pytest.mark.asyncio
async def test_registered_drill_host_does_not_start_general_worker_components(exercise, monkeypatch):
    state = exercise("RMAN", "19c")
    settings = SimpleNamespace(
        oracle_local_backups={}, file_local_backups={}, local_drills={state.resource_id: state.config},
        worker=SimpleNamespace(service_name="test-worker", service_version="test", claim_interval_seconds=2),
    )
    runtime = SimpleNamespace(
        uow_factory=object(), start=AsyncMock(), close=AsyncMock(),
        check_database_connection=AsyncMock(return_value={"database": "ok"}),
    )
    running, stopped = asyncio.Event(), asyncio.Event()

    def unexpected(*args, **kwargs):
        pytest.fail("独立演练机不应装配通用 Worker")

    class FakeProtectionWorker:
        def __init__(self, *, uow_factory, backend, interval_seconds):
            assert uow_factory is runtime.uow_factory
            assert backend.drill_resource_ids == {state.resource_id}
            assert backend.backup_resource_ids == frozenset()
            assert interval_seconds == 2

        async def run_forever(self):
            running.set()
            await stopped.wait()

        def stop(self):
            stopped.set()

    monkeypatch.setattr(worker, "configure_process_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "create_database_runtime", lambda _: SimpleNamespace(session_factory=object()))
    monkeypatch.setattr(worker, "create_aiops_uow_factory", lambda _: runtime.uow_factory)
    monkeypatch.setattr(worker, "AIOpsProcessRuntime", lambda **kwargs: runtime)
    monkeypatch.setattr(worker, "ManagedCredentialCipher", SimpleNamespace(
        from_settings=lambda _: object(), from_environment=lambda: object(),
    ))
    monkeypatch.setattr(worker, "create_cursor_codec", unexpected)
    monkeypatch.setattr(data_protection, "machine_id", lambda: _MACHINE)
    monkeypatch.setattr(data_protection, "DataProtectionWorker", FakeProtectionWorker)
    app = worker.create_aiops_worker_probe(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(running.wait(), timeout=1)
        assert await app.state.ready_check() == {"database": "ok", "component:data-protection": "ok"}
    assert stopped.is_set()
    runtime.start.assert_awaited_once()
    runtime.close.assert_awaited_once()


def test_local_drill_binding_uses_existing_aiops_config(tmp_path):
    resource_id, target_id = uuid7(), uuid7()
    path = tmp_path / "deployment.toml"
    path.write_text(f"""
environment = "development"
[aiops.local_drills."{resource_id}"]
tool = "XTRABACKUP"
source_target_id = "{target_id}"
machine_id = "{_MACHINE}"
os_user = "drill"
destination_key = "mysql-exercise"
drill_root = "/srv/drill/mysql"
backup_root = "/srv/backups/mysql"
required_free_bytes = 100000
executable = "/usr/bin/xtrabackup"
mysqld_executable = "/usr/sbin/mysqld"
server_id = 403
""")
    settings = load_settings(AIOpsSettings, service="aiops_agent", config_file=path)
    assert settings.local_drills[resource_id].source_target_id == target_id
