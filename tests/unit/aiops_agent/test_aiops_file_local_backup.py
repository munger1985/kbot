"""原生备份适配离线测试：仅模拟数据库查询和已安装工具。"""

from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters import mysql_local_backup, postgresql_local_backup
from aiops_agent.adapters.file_local_backup import FileLocalBackupBackend
from aiops_agent.application.data_protection_backup import render_backup
from aiops_agent.application.data_protection_execution import BackupArtifact, DrillRejected, valid_backup_chain
from aiops_agent.application.data_protection_jobs import ProtectionJobs, ProtectionWork
from aiops_agent.bootstrap.data_protection import local_protection_backend
from aiops_agent.config import AIOpsOracleLocalBackupConfig, AIOpsPgBackRestConfig, AIOpsSettings, AIOpsXtraBackupConfig
from platform_core.config import load_settings
from platform_core.contracts.aiops.data_protection import ProtectionPlanSettings
from platform_core.identity import uuid7


_LABEL = "20261118-153010F"
_SECRET = r'fixture\:password "s";#'


@pytest.fixture
def native(tmp_path):
    sockets = []

    def build(tool, version):
        data, root, sockets_dir = (tmp_path / name for name in ("data", "backup", "sock"))
        for directory in (data, root, sockets_dir):
            directory.mkdir()
        executable = tmp_path / ("pgbackrest" if tool == "PGBACKREST" else "xtrabackup")
        checksum = tmp_path / "innochecksum"
        for binary in (executable, checksum):
            binary.touch()
            binary.chmod(0o700)
        db_type = "POSTGRESQL" if tool == "PGBACKREST" else "MYSQL"
        resource_id, target_id, credential_id = uuid7(), uuid7(), uuid7()
        scope = {"domain_id": uuid7()}
        target = SimpleNamespace(
            target_id=target_id, status="ENABLED", db_type=db_type, deployment_type="SINGLE_INSTANCE",
            db_role="PRIMARY", row_version=1, version_code=version, tenant_id=scope.get("tenant_id"),
        )
        row = SimpleNamespace(credential_id=credential_id, status="ACTIVE", nonce=b"initial-test")
        uow = SimpleNamespace(
            business_systems=SimpleNamespace(
                get_resource_scoped=AsyncMock(return_value=SimpleNamespace(resource_type_key="DATABASE", lifecycle_status="ACTIVE")),
                get_database_profile=AsyncMock(return_value=SimpleNamespace(target_id=target_id)),
            ),
            targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target)),
            managed_credentials=SimpleNamespace(find=AsyncMock(return_value=row)),
            protection_plans=SimpleNamespace(due_plan=AsyncMock(return_value=None)),
        )

        @asynccontextmanager
        async def uow_factory():
            yield uow

        run = AsyncMock()
        credentials = SimpleNamespace(read=AsyncMock(return_value={
            "auth_mode": "DATABASE_PASSWORD", "username": "backup_user", "password": _SECRET,
        }))
        common = dict(target_id=target_id, executable=str(executable), data_directory=str(data), backup_root=str(root))
        state = SimpleNamespace(
            tool=tool, run=run, credentials=credentials, resource_id=resource_id, scope=scope, uow=uow,
            uow_factory=uow_factory, data=data, root=root, target=target, row=row, run_id=uuid7(),
        )
        if tool == "PGBACKREST":
            sock = sockets_dir / ".s.PGSQL.5432"
            (data / "global").mkdir()
            (data / "global" / "pg_control").write_bytes(b"mock control file")
            (data / "PG_VERSION").write_text(version)
            config_file = tmp_path / "pgbackrest.conf"
            config_file.write_text(f"[global]\nrepo1-path={root}\n[main]\npg1-path={data}\n")
            config = AIOpsPgBackRestConfig(
                **common, tool=tool, config_file=str(config_file), stanza="main",
                socket_directory=str(sockets_dir), system_identifier="7439954591938921111",
            )
            settings = ProtectionPlanSettings(tool=tool, storage_key="main", backup_parallelism=3)
            probe = AsyncMock(return_value={
                "version": int(version) * 10000 + 4, "directory": str(data), "system_identifier": config.system_identifier,
                "local_socket": True, "is_primary": True, "archive_mode": "on",
                "archive_command": "pgbackrest --stanza=main archive-push %p", "archive_library": "",
                "can_backup": True,
            })
            backend = FileLocalBackupBackend(
                bindings={resource_id: config}, uow_factory=uow_factory, credentials=credentials, run_local=run, pg_probe=probe,
            )
            state.info = [{
                "name": "main", "status": {"code": 2}, "repo": [{"key": 1, "status": {"code": 2}}],
                "db": [{"id": 1, "repo-key": 1, "system-id": int(config.system_identifier), "version": version}], "backup": [],
            }]
            state.pg_conf = config_file
        else:
            sock = sockets_dir / "mysql.sock"
            config = AIOpsXtraBackupConfig(
                **common, tool=tool, checksum_executable=str(checksum), socket=str(sock), server_uuid=uuid7(),
            )
            settings = ProtectionPlanSettings(tool=tool, backup_directory=str(root), backup_parallelism=3)
            (data / "auto.cnf").write_text(f"[auto]\nserver-uuid={config.server_uuid}\n")
            (data / "db").mkdir()
            files = [Path("ibdata1"), Path("db/test.ibd")]
            if version != "5.7":
                files.append(Path("mysql.ibd"))
            for file in files:
                (data / file).write_bytes(b"mock data page")
            state.files = files
            state.mysql_version = f"{version}.44"
            probe = AsyncMock(return_value=(
                (state.mysql_version, str(config.server_uuid), f"{data}/", str(sock), 0, 0, "backup_user@localhost"),
                {"innodb_data_file_path": "ibdata1:12M:autoextend", "innodb_undo_directory": "./"},
                ("./db/test.ibd",),
            ))
            backend = FileLocalBackupBackend(
                bindings={resource_id: config}, uow_factory=uow_factory, credentials=credentials, run_local=run, mysql_probe=probe,
            )
        server = socket.socket(socket.AF_UNIX)
        sockets.append(server)
        server.bind(str(sock))
        state.config, state.settings, state.backend, state.probe = config, settings, backend, probe
        state.run.side_effect = _mock_tools(state)
        return state

    yield build
    for server in sockets:
        server.close()


def _mock_tools(state, result="good"):
    written = False
    auth_path = None

    async def run(executable, args, *, env=None, capture=True):
        nonlocal written, auth_path
        assert _SECRET not in repr((executable, args, env))
        assert not any(arg in {"restore", "check", "--prepare", "--copy-back", "--move-back"} for arg in args)
        if state.tool == "PGBACKREST":
            if args == ("--version",):
                return 0, "pgBackRest 2.54.2" if result == "tool_version" else "pgBackRest 2.55.0"
            assert "--repo=1" in args and "--stanza=main" in args
            assert f"--config={state.pg_conf}" in args
            if args[-1] == "info":
                return 0, json.dumps(state.info)
            if args[-1] == "backup":
                assert "--no-expire-auto" in args and "--archive-check" in args
                assert "--type=full" in args and "--process-max=3" in args
                assert f"--annotation=aiops-run={state.run_id}" in args
                assert f"--pg1-path={state.data}" in args and f"--pg1-socket-path={state.config.socket_directory}" in args
                auth_path = Path(env["PGPASSFILE"])
                assert auth_path.stat().st_mode & 0o777 == 0o600
                assert r"*:5432:postgres:backup_user:fixture\\\:password" in auth_path.read_text()
                assert capture is False
                annotation = {} if result == "unbound" else {"aiops-run": str(state.run_id)}
                state.info[0]["backup"] = [{
                    "label": _LABEL, "type": "full", "error": result == "page_error", "annotation": annotation,
                    "database": {"repo-key": 1, "id": 1},
                    "info": {"size": 100}, "timestamp": {"start": 50, "stop": 90},
                    "archive": {"start": "000000010000000000000001",
                                "stop": "000000020000000000000004" if result == "timeline" else "000000010000000000000004"},
                }]
                return (1, "") if result == "exit" else (0, "")
            assert args[-1] == "verify" and f"--set={_LABEL}" in args
            assert "--output=text" in args and "--verbose" in args and not auth_path.exists()
            valid = {"corrupt": 1, "empty_verify": 0}.get(result, 2)
            count = 0 if result == "empty_verify" else 2
            report = f"stanza: main\nstatus: ok\n  backup: {_LABEL}, status: valid, total files checked: {count}, total valid files: {valid}\n"
            return 0, "" if result == "no_verify" else report
        if args == ("--no-defaults", "--version"):
            family = {"5.7": "2.4", "8.0": "8.0", "8.4": "8.4"}[state.target.version_code]
            if result == "tool_version":
                family = "1.1"
            return 0, f"xtrabackup version {family}.99 based on MySQL server {state.mysql_version} Linux (x86_64)"
        if args == ("--version",):
            checksum_version = "5.1.2" if result == "checksum_version" else state.mysql_version
            return 0, f"innochecksum Ver {checksum_version} for Linux on x86_64"
        if "--backup" in args:
            assert args[0].startswith("--defaults-file=") and f"--datadir={state.data}" in args
            assert "--parallel=3" in args
            auth_path = Path(args[0].split("=", 1)[1])
            assert auth_path.stat().st_mode & 0o777 == 0o600
            assert r'password="fixture\\:password \"s\";#"' in auth_path.read_text()
            assert f'socket="{state.config.socket}"' in auth_path.read_text()
            assert capture is False
            path = state.root / f"aiops-{state.run_id}"
            assert f"--target-dir={path}" in args
            assert path.stat().st_mode & 0o777 == 0o700
            assert not list(path.iterdir())
            if result == "exit":
                return 1, ""
            (path / "db").mkdir()
            for file in state.files:
                if result != "missing_data" or file.name != "test.ibd":
                    (path / file).write_bytes(b"mock data page")
            if result == "alias":
                (path / "db" / "test.ibd").unlink()
                (path / "db" / "test.ibd").symlink_to(state.data / "db" / "test.ibd")
            if result != "missing_redo":
                (path / "xtrabackup_logfile").write_bytes(b"mock redo")
            if result != "missing_meta":
                (path / "xtrabackup_checkpoints").write_text(
                    f"backup_type = full-backuped\nfrom_lsn = 0\nto_lsn = 100\nlast_lsn = {50 if result == 'bad_lsn' else 200}\n",
                )
            (path / "xtrabackup_info").write_text(
                f"server_version = {state.mysql_version}\nstart_time = 2026-11-18 15:30:10\nend_time = 2026-11-18 15:31:10\n"
                f"partial = {'Y' if result == 'partial' else 'N'}\nincremental = N\nformat = file\ncompressed = N\nencrypted = N\n",
            )
            (path / "backup-my.cnf").write_text("[mysqld]\ninnodb_page_size=16384\n")
            written = True
            return 0, ""
        assert written and not auth_path.exists() and capture is False
        assert executable == state.config.checksum_executable
        assert len(args) == 1 and Path(args[0]).is_relative_to(state.root / f"aiops-{state.run_id}")
        return (1 if result == "corrupt" else 0), ""

    return run


async def _preflight(state, operation="BACKUP"):
    return await state.backend.preflight(
        scope=state.scope, actor_id=str(uuid7()), resource_id=state.resource_id, settings=state.settings, operation=operation,
    )


async def _work(state):
    approval = await _preflight(state)
    binding = {**asdict(approval), "target_id": str(approval.target_id)}
    now = datetime.now(UTC)
    return (
        ProtectionWork(
            state.run_id, state.resource_id, state.scope, "BACKUP", now, now + timedelta(hours=1),
            {"settings": state.settings.model_dump(mode="json"), "approval": {"binding": binding}},
        ),
        render_backup(state.settings, state.run_id, db_version=approval.db_version),
    )


def test_pgbackrest_chain_can_span_more_than_one_week():
    refs = (_LABEL, *(f"{_LABEL}_202611{day:02}-011003I" for day in range(19, 29)))
    catalog = []
    for index, label in enumerate(refs):
        item = {
            "label": label, "type": "incr" if index else "full", "database": {"id": 1, "repo-key": 1},
            "timestamp": {"start": 100 + index, "stop": 101 + index}, "info": {"size": 100},
            "archive": {"start": "000000010000000000000001", "stop": "000000010000000000000004"},
        }
        if index:
            item.update(prior=refs[index - 1], reference=list(refs[:index]))
        catalog.append(item)
    artifact = BackupArtifact("main", refs[-1], datetime.now(UTC), True, "INCREMENTAL", refs[:-1])
    assert valid_backup_chain("PGBACKREST", artifact)
    assert postgresql_local_backup.verified_chain(catalog, 1, refs)
    assert not postgresql_local_backup.verified_chain(catalog[:5] + catalog[6:], 1, refs)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["16", "17", "18"])
@pytest.mark.parametrize("fault", ["none", "missing_directory", "other_parent", "full_fallback"])
async def test_pgbackrest_incremental_cannot_attach_to_an_unapproved_backup(native, version, fault):
    state = native("PGBACKREST", version)
    state.settings = ProtectionPlanSettings.model_validate({
        **state.settings.model_dump(), "backup_schedule": {"timezone": "UTC", "cron": "0 2 * * 0"},
        "backup_incremental_schedule": {"timezone": "UTC", "cron": "0 1 * * 1-6"},
    })
    work, invocation = await _work(state)
    full = await state.backend.backup(work, invocation)
    assert full.artifact.reference == _LABEL
    state.run_id = uuid7()
    work, _ = await _work(state)
    work = replace(work, backup=full.artifact, snapshot={**work.snapshot, "backup_kind": "INCREMENTAL"})
    invocation = render_backup(state.settings, state.run_id, db_version=version, kind="INCREMENTAL", baseline=full.artifact)
    directory = state.root / "backup" / "main" / _LABEL
    if fault != "missing_directory":
        directory.mkdir(parents=True)
    next_label = _LABEL + "_20261119-011003I"
    original = state.run.side_effect

    async def command(program, args, **kwargs):
        if args[-1] == "backup":
            assert "--type=incr" in args
            parent = state.info[0]["backup"][0]
            state.info[0]["backup"].append({
                **parent, "label": next_label if fault != "full_fallback" else "20261119-011003F",
                "type": "incr" if fault != "full_fallback" else "full", "reference": [_LABEL],
                "prior": "20261117-011003F" if fault == "other_parent" else _LABEL,
                "annotation": {"aiops-run": str(state.run_id)},
            })
            return 0, ""
        if args[-1] == "verify":
            assert f"--set={next_label}" in args
            return 0, f"stanza: main\nstatus: ok\nbackup: {next_label}, status: valid, total files checked: 2, total valid files: 2\n"
        return await original(program, args, **kwargs)

    state.run.side_effect = command
    state.run.reset_mock()
    result = await state.backend.backup(work, invocation)
    if fault == "none":
        assert result.artifact.kind == "INCREMENTAL" and result.artifact.dependencies == (_LABEL,)
    else:
        assert result.artifact is None
        if fault == "missing_directory":
            assert not any(call.args[1][-1] == "backup" for call in state.run.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["5.7", "8.0", "8.4"])
@pytest.mark.parametrize("fault", ["none", "missing_meta", "wrong_lsn", "changed_base"])
async def test_xtrabackup_incremental_checks_native_checkpoints_and_artifacts(native, version, fault):
    state = native("XTRABACKUP", version)
    state.settings = ProtectionPlanSettings.model_validate({
        **state.settings.model_dump(), "backup_schedule": {"timezone": "UTC", "cron": "0 2 * * 0"},
        "backup_incremental_schedule": {"timezone": "UTC", "cron": "0 1 * * 1-6"},
    })
    work, invocation = await _work(state)
    full = await state.backend.backup(work, invocation)
    assert full.artifact.end_lsn == 100
    state.run_id = uuid7()
    work, _ = await _work(state)
    baseline = replace(full.artifact, end_lsn=99) if fault == "changed_base" else full.artifact
    work = replace(work, backup=baseline, snapshot={**work.snapshot, "backup_kind": "INCREMENTAL"})
    invocation = render_backup(state.settings, state.run_id, db_version=version, kind="INCREMENTAL", baseline=baseline)
    original = state.run.side_effect
    directory = Path(invocation.output_directory)

    async def command(program, args, **kwargs):
        if "--backup" not in args:
            return await original(program, args, **kwargs)
        assert f"--incremental-basedir={state.root / full.artifact.reference}" in args
        (directory / "xtrabackup_checkpoints").write_text(
            f"backup_type = incremental\nfrom_lsn = {101 if fault == 'wrong_lsn' else 100}\nto_lsn = 200\nlast_lsn = 210\n",
        )
        (directory / "xtrabackup_logfile").write_bytes(b"redo")
        (directory / "backup-my.cnf").write_text("[mysqld]\ninnodb_page_size=16384\n")
        (directory / "xtrabackup_info").write_text(
            f"server_version = {state.mysql_version}\nincremental = Y\nformat = file\npartial = N\ncompressed = N\n",
        )
        (directory / "ibdata1.delta").write_bytes(b"mock native delta")
        if fault != "missing_meta":
            (directory / "ibdata1.meta").write_text("page_size = 16384\nzip_size = 0\nspace_id = 0\n")
        return 0, ""

    state.run.side_effect = command
    result = await state.backend.backup(work, invocation)
    if fault == "none":
        assert result.artifact.kind == "INCREMENTAL" and result.artifact.dependencies == (baseline.reference,)
        assert (result.artifact.start_lsn, result.artifact.end_lsn) == (100, 200)
        assert result.artifact.fingerprint == mysql_local_backup.incremental_fingerprint(directory)
    else:
        assert result.artifact is None
        if fault == "changed_base":
            assert not directory.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "version"), [
    ("PGBACKREST", "16"), ("PGBACKREST", "17"), ("PGBACKREST", "18"),
    ("XTRABACKUP", "5.7"), ("XTRABACKUP", "8.0"), ("XTRABACKUP", "8.4"),
])
async def test_native_version_presets_verify_backup_without_restore(native, tool, version):
    state = native(tool, version)
    work, invocation = await _work(state)
    result = await state.backend.backup(work, invocation)
    assert result.exit_code == 0 and result.artifact.checksum_verified is True
    assert result.artifact.reference == (_LABEL if tool == "PGBACKREST" else f"aiops-{state.run_id}")
    assert result.continuous_logs_verified is None
    assert state.uow.managed_credentials.find.await_args.kwargs == {
        **state.scope, "namespace": "aiops", "credential_kind": "protection_backup", "external_key": str(state.resource_id),
    }
    assert _SECRET not in repr(work.snapshot)
    with pytest.raises(DrillRejected):
        await state.backend.drill_runner(work)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["exit", "unbound", "timeline", "page_error", "corrupt", "empty_verify", "no_verify"])
async def test_pg_rejects_unverified_output(native, fault):
    state = native("PGBACKREST", "18")
    state.run.side_effect = _mock_tools(state, fault)
    result = await state.backend.backup(*await _work(state))
    assert result.exit_code != 0 and result.artifact is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["exit", "missing_data", "missing_redo", "missing_meta", "bad_lsn", "corrupt", "alias", "partial"])
async def test_mysql_rejects_unverified_output(native, fault):
    state = native("XTRABACKUP", "8.4")
    state.run.side_effect = _mock_tools(state, fault)
    result = await state.backend.backup(*await _work(state))
    assert result.exit_code != 0 and result.artifact is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [
    "version", "identity", "role", "privileges", "archive", "plan_stanza", "repo_path", "repo_host", "repo_id", "tool_version",
])
async def test_pg_preflight_has_no_backup_side_effect(native, fault):
    state = native("PGBACKREST", "18")
    live = state.probe.return_value
    if fault == "version":
        live["version"] = 170006
    elif fault == "identity":
        live["system_identifier"] = "1234"
    elif fault == "role":
        live["is_primary"] = False
    elif fault == "privileges":
        live["can_backup"] = False
    elif fault == "archive":
        live["archive_mode"] = "off"
    elif fault == "plan_stanza":
        state.settings = state.settings.model_copy(update={"storage_key": "other"})
    elif fault == "repo_path":
        state.pg_conf.write_text("[main]\nrepo1-path=/other/repo\n")
    elif fault == "repo_host":
        state.pg_conf.write_text(f"[main]\nrepo1-path={state.root}\n[global:backup]\nrepo1-host=remote\n")
    elif fault == "repo_id":
        state.info[0]["db"][0]["system-id"] = 1234
    else:
        state.run.side_effect = _mock_tools(state, fault)
    with pytest.raises(DrillRejected):
        await _preflight(state)
    assert all(call.args[1][-1] in {"--version", "info"} for call in state.run.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [
    "version", "identity", "socket", "role", "outside_backup", "outside_tablespace", "wrong_layout",
    "auto_uuid", "tool_version", "checksum_version",
])
async def test_mysql_preflight_has_no_backup_side_effect(native, fault, tmp_path):
    state = native("XTRABACKUP", "8.0")
    live, layout, paths = state.probe.return_value
    live = list(live)
    if fault == "version":
        live[0] = "8.4.0"
    elif fault == "identity":
        live[1] = str(uuid7())
    elif fault == "socket":
        live[3] = "/srv/different/mysql.sock"
    elif fault == "role":
        live[5] = 1
    elif fault == "outside_backup":
        state.settings = state.settings.model_copy(update={"backup_directory": str(tmp_path)})
    elif fault == "outside_tablespace":
        paths = ("/srv/another/db/test.ibd",)
    elif fault == "wrong_layout":
        layout["innodb_undo_directory"] = "/srv/another/undo"
    elif fault == "auto_uuid":
        (state.data / "auto.cnf").write_text(f"[auto]\nserver-uuid={uuid7()}\n")
    else:
        state.run.side_effect = _mock_tools(state, fault)
    state.probe.return_value = (live, layout, paths)
    with pytest.raises(DrillRejected):
        await _preflight(state)
    assert all(call.args[1][-1] == "--version" for call in state.run.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["rotated", "revoked", "revision", "invocation", "wrong_scope", "drill"])
async def test_job_rechecks_frozen_source_before_writing(native, fault):
    state = native("PGBACKREST", "17")
    work, invocation = await _work(state)
    if fault == "rotated":
        state.row.nonce = b"rotated-cred"
    elif fault == "revoked":
        state.row.status = "REVOKED"
    elif fault == "revision":
        state.target.row_version += 1
    elif fault == "invocation":
        invocation = replace(invocation, arguments=("restore",))
    elif fault == "wrong_scope":
        state.uow.targets.get_scoped.return_value = None
    else:
        work = replace(work, operation="DRILL")
    with pytest.raises(DrillRejected):
        await state.backend.backup(work, invocation)
    assert all(call.args[1][-1] in {"--version", "info"} for call in state.run.await_args_list)


def test_deployment_config_opt_in_without_new_process(tmp_path):
    path = tmp_path / "deployment.toml"
    base = "environment = 'development'\n"
    path.write_text(base)
    settings = load_settings(AIOpsSettings, service="aiops_agent", config_file=path)
    assert local_protection_backend(settings=settings, uow_factory=None, credentials=None) is None
    postgres, mysql = uuid7(), uuid7()
    path.write_text(base + f"""
[aiops.file_local_backups."{postgres}"]
tool = "PGBACKREST"
target_id = "{uuid7()}"
executable = "/usr/bin/pgbackrest"
data_directory = "/var/lib/pgsql/data"
backup_root = "/srv/backup/pg"
config_file = "/etc/pgbackrest/pgbackrest.conf"
stanza = "main"
socket_directory = "/run/postgresql"
system_identifier = "7439954591938921111"

[aiops.file_local_backups."{mysql}"]
tool = "XTRABACKUP"
target_id = "{uuid7()}"
executable = "/usr/bin/xtrabackup"
data_directory = "/var/lib/mysql"
backup_root = "/srv/backup/mysql"
checksum_executable = "/usr/bin/innochecksum"
socket = "/run/mysqld/mysqld.sock"
server_uuid = "{uuid7()}"
""")
    settings = load_settings(AIOpsSettings, service="aiops_agent", config_file=path)
    assert isinstance(settings.file_local_backups[postgres], AIOpsPgBackRestConfig)
    assert isinstance(settings.file_local_backups[mysql], AIOpsXtraBackupConfig)


@pytest.mark.asyncio
async def test_shared_worker_routes_only_explicit_backup_resources(native):
    state = native("PGBACKREST", "18")
    oracle, mysql = uuid7(), uuid7()
    oracle_config = AIOpsOracleLocalBackupConfig(
        target_id=uuid7(), oracle_home="/opt/oracle", oracle_sid="orcl", dbid=12345,
        db_unique_name="orcl", backup_root="/srv/oracle",
    )
    mysql_config = AIOpsXtraBackupConfig(
        target_id=uuid7(), tool="XTRABACKUP", executable="/usr/bin/xtrabackup", data_directory="/var/lib/mysql",
        backup_root="/srv/mysql", checksum_executable="/usr/bin/innochecksum", socket="/run/mysql.sock", server_uuid=uuid7(),
    )
    settings = SimpleNamespace(
        oracle_local_backups={oracle: oracle_config},
        file_local_backups={state.resource_id: state.config, mysql: mysql_config},
    )
    merged = local_protection_backend(settings=settings, uow_factory=state.uow_factory, credentials=state.credentials)
    assert merged.backup_resource_ids == {oracle, mysql, state.resource_id}
    work, invocation = await _work(state)
    oracle_runner = merged._routes[oracle]
    file_runner = merged._routes[state.resource_id]
    oracle_runner.backup = AsyncMock(return_value="oracle")
    file_runner.backup = AsyncMock(side_effect=lambda work, _: work.resource_id)
    assert await merged.backup(replace(work, resource_id=oracle), invocation) == "oracle"
    assert await merged.backup(work, invocation) == state.resource_id
    assert await merged.backup(replace(work, resource_id=mysql), invocation) == mysql
    oracle_runner.backup.assert_awaited_once()
    assert file_runner.backup.await_count == 2
    with pytest.raises(DrillRejected):
        await merged.backup(replace(work, resource_id=uuid7()), invocation)
    with pytest.raises(DrillRejected):
        await merged.backup(replace(work, operation="DRILL"), invocation)
    with pytest.raises(DrillRejected):
        await merged.drill_runner(work)
    jobs = ProtectionJobs(uow_factory=state.uow_factory, backend=merged)
    assert await jobs.claim("DRILL") is None
    state.uow.protection_plans.due_plan.assert_not_awaited()
    assert await jobs.claim("BACKUP") is None
    assert state.uow.protection_plans.due_plan.await_args.kwargs["resource_ids"] == merged.backup_resource_ids
    settings.oracle_local_backups[state.resource_id] = oracle_config
    with pytest.raises(ValueError, match="不能登记多个"):
        local_protection_backend(settings=settings, uow_factory=None, credentials=None)


@pytest.mark.asyncio
async def test_pg_probe_uses_bound_socket_without_calling_backup_functions(native, monkeypatch):
    state = native("PGBACKREST", "17")
    connection = SimpleNamespace(fetchrow=AsyncMock(return_value=state.probe.return_value), close=AsyncMock())
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(postgresql_local_backup.asyncpg, "connect", connect)
    identity = SimpleNamespace(username="backup_user", password=_SECRET, db_version="17")
    assert await postgresql_local_backup.probe(state.config, identity) == state.probe.return_value
    assert connect.await_args.kwargs["host"] == state.config.socket_directory
    assert connect.await_args.kwargs["database"] == "postgres"
    query = connection.fetchrow.await_args.args[0]
    assert "pg_control_system()" in query and "has_function_privilege" in query
    connection.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["5.7", "8.0", "8.4"])
@pytest.mark.parametrize("allowed", [True, False])
async def test_mysql_probe_only_reads_version_specific_permissions_and_role(native, monkeypatch, version, allowed):
    state = native("XTRABACKUP", version)
    live, layout, files = state.probe.return_value
    grants = "GRANT PROCESS, RELOAD, LOCK TABLES, REPLICATION CLIENT ON *.* TO `backup_user`@`localhost`"
    if not allowed:
        grants = grants.replace("PROCESS, ", "")
    responses = { "SHOW GRANTS": [
        (grants,),
        ("GRANT BACKUP_ADMIN ON *.* TO `backup_user`@`localhost`",),
    ]}
    queries = []

    class Cursor:
        async def execute(self, sql, *_):
            self.sql = sql
            queries.append(sql)

        async def fetchone(self):
            return live if self.sql.startswith("SELECT @@version") else None

        async def fetchall(self):
            if self.sql.startswith("SHOW GLOBAL VARIABLES"):
                return tuple(layout.items())
            if self.sql.startswith("SELECT PATH"):
                return tuple((path,) for path in files)
            return responses[self.sql]

    @asynccontextmanager
    async def cursor():
        yield Cursor()

    connection = SimpleNamespace(cursor=cursor, close=lambda: None)
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(mysql_local_backup.aiomysql, "connect", connect)
    identity = SimpleNamespace(username="backup_user", password=_SECRET, db_version=version)
    if allowed:
        assert await mysql_local_backup.probe(state.config, identity) == (live, layout, files)
        assert ("SHOW REPLICA STATUS" if version == "8.4" else "SHOW SLAVE STATUS") in queries
        schema = "INNODB_SYS_DATAFILES" if version == "5.7" else "INNODB_DATAFILES"
        assert f"SELECT PATH FROM information_schema.{schema}" in queries
        assert ("SELECT 1 FROM performance_schema.log_status LIMIT 0" in queries) == (version != "5.7")
    else:
        with pytest.raises(DrillRejected, match="缺少"):
            await mysql_local_backup.probe(state.config, identity)
    assert connect.await_args.kwargs["unix_socket"] == state.config.socket
    assert all(sql.startswith(("SELECT ", "SHOW ")) for sql in queries)
