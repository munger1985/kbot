"""本机 Oracle 备份执行：用假 SQL*Plus/RMAN 进程验证，无数据库操作。"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.adapters.oracle_local_backup import OracleLocalBackupBackend, piece_manifest
from aiops_agent.application.data_protection_backup import backup_storage_key, render_backup
from aiops_agent.application.data_protection_execution import BackupArtifact, DrillRejected
from aiops_agent.application.data_protection_jobs import ProtectionJobs, ProtectionWork
from aiops_agent.config import AIOpsOracleLocalBackupConfig, AIOpsSettings
from platform_core.config import load_settings
from platform_core.contracts.aiops.data_protection import ProtectionBackupIdentityInput, ProtectionPlanSettings
from platform_core.identity import uuid7

READY = "AIOPS_FACTS|12345|TESTCDB|ARCHIVELOG|PRIMARY|READ WRITE|19.3.0.0.0|YES|CDB$ROOT|YES|FALSE"
INVENTORY = (
    "AIOPS_SET|21|D|NO|NO|YES\n"
    "AIOPS_SET|22|L|NO|NO|NO\n"
    "AIOPS_SET|23|D|YES|NO|NO\n"
    "AIOPS_SET|24|D|NO|YES|NO\n"
    "AIOPS_DATAFILES|YES\n"
)


@pytest.fixture
def local_backup(tmp_path):
    home, directory = tmp_path / "oracle_home", tmp_path / "backup"
    directory.mkdir()
    (home / "bin").mkdir(parents=True)
    for name in ("sqlplus", "rman"):
        binary = home / "bin" / name
        binary.touch()
        binary.chmod(0o700)
    resource_id, target_id, credential_id = uuid7(), uuid7(), uuid7()
    scope = {"domain_id": 18}
    config = AIOpsOracleLocalBackupConfig(
        target_id=target_id, oracle_home=str(home), oracle_sid="TESTCDB", dbid=12345,
        db_unique_name="TESTCDB", backup_root=str(directory),
    )
    source = SimpleNamespace(
        target_id=target_id, status="ENABLED", db_type="ORACLE", deployment_type="SINGLE_INSTANCE",
        db_role="PRIMARY", row_version=1, oracle_container_scope="CDB_ROOT", version_code="19c",
    )
    uow = SimpleNamespace(
        business_systems=SimpleNamespace(
            get_resource_scoped=AsyncMock(return_value=SimpleNamespace(resource_type_key="DATABASE", lifecycle_status="ACTIVE")),
            get_database_profile=AsyncMock(return_value=SimpleNamespace(target_id=target_id)),
        ),
        targets=SimpleNamespace(get_scoped=AsyncMock(return_value=source)),
        managed_credentials=SimpleNamespace(find=AsyncMock(return_value=SimpleNamespace(credential_id=credential_id, status="ACTIVE"))),
        protection_plans=SimpleNamespace(due_plan=AsyncMock(return_value=None)),
    )

    @asynccontextmanager
    async def uow_factory():
        yield uow

    tool = AsyncMock(return_value=(0, READY))
    credentials = SimpleNamespace(read=AsyncMock(return_value={"auth_mode": "ORACLE_OS", "username": "oracle"}))
    backend = OracleLocalBackupBackend(
        bindings={resource_id: config}, uow_factory=uow_factory, credentials=credentials, run_local=tool,
        os_user=lambda: "oracle",
    )
    return SimpleNamespace(
        backend=backend, tool=tool, credentials=credentials, resource_id=resource_id, scope=scope, uow=uow,
        uow_factory=uow_factory, settings=ProtectionPlanSettings(tool="RMAN", backup_directory=str(directory)),
        target=source, root=directory,
    )


async def _preflight(fixture):
    return await fixture.backend.preflight(
        scope=fixture.scope, actor_id=str(uuid7()), resource_id=fixture.resource_id, settings=fixture.settings, operation="BACKUP",
    )


def test_backup_os_auth_requires_only_explicit_username():
    assert ProtectionBackupIdentityInput(auth_mode="ORACLE_OS", username="oracle").password is None
    with pytest.raises(ValidationError):
        ProtectionBackupIdentityInput(auth_mode="ORACLE_OS", username="oracle", password="unused")
    with pytest.raises(ValidationError):
        ProtectionBackupIdentityInput(username="oracle")


def test_oracle_bindings_are_opt_in_via_existing_deployment_config(tmp_path):
    resource_id, target_id = uuid7(), uuid7()
    path = tmp_path / "deployment.toml"
    base = "environment = 'development'\n"
    path.write_text(base, encoding="utf-8")
    assert load_settings(AIOpsSettings, service="aiops_agent", config_file=path).oracle_local_backups == {}
    path.write_text(
        base + f"""
[aiops.oracle_local_backups."{resource_id}"]
target_id = "{target_id}"
oracle_home = "/opt/oracle/dbhome"
oracle_sid = "TESTCDB"
dbid = 12345
db_unique_name = "TESTCDB"
backup_root = "/srv/db_backups/testcdb"
""",
        encoding="utf-8",
    )
    settings = load_settings(AIOpsSettings, service="aiops_agent", config_file=path)
    assert settings.oracle_local_backups[resource_id].target_id == target_id
    assert settings.oracle_local_backups[resource_id].backup_root == "/srv/db_backups/testcdb"


@pytest.mark.asyncio
@pytest.mark.parametrize(("probe", "expected_code"), [
    (READY.replace("|ARCHIVELOG|", "|NOARCHIVELOG|"), "ORACLE_ARCHIVELOG_REQUIRED"),
    (READY.replace("|12345|", "|54321|"), "UNVERIFIED"),
    (READY.replace("CDB$ROOT", "PDB01"), "UNVERIFIED"),
    (READY.replace("|PRIMARY|", "|PHYSICAL STANDBY|"), "UNVERIFIED"),
    (READY.replace("|FALSE", "|TRUE"), "UNVERIFIED"),
    (READY.replace("|19.3.0.0.0|", "|23.26.0.0.0|"), "BACKUP_VERSION_MISMATCH"),
])
async def test_preflight_stops_before_rman_on_unsupported_database(local_backup, probe, expected_code):
    local_backup.tool.return_value = (0, probe)
    with pytest.raises(DrillRejected) as rejected:
        await _preflight(local_backup)
    assert rejected.value.code == expected_code
    local_backup.tool.assert_awaited_once()
    assert local_backup.tool.await_args.args[1] == "sqlplus"


@pytest.mark.asyncio
async def test_configured_oracle_26ai_requires_full_release_and_uses_26ai_preset(local_backup):
    local_backup.target.version_code = "26ai"
    for version in ("19.3.0.0.0", "23.9.0.0.0", "23.0.0.0.0"):
        local_backup.tool.return_value = (0, READY.replace("19.3.0.0.0", version))
        with pytest.raises(DrillRejected) as rejected:
            await _preflight(local_backup)
        assert rejected.value.code == "BACKUP_VERSION_MISMATCH"
    local_backup.tool.return_value = (0, READY.replace("19.3.0.0.0", "23.26.1.0.0"))
    binding = await _preflight(local_backup)
    assert binding.db_version == "26ai" and binding.template_key == "rman.oracle26ai.backup"
    assert all("i.version_full" in call.args[3] for call in local_backup.tool.await_args_list)


@pytest.mark.asyncio
async def test_unknown_oracle_version_never_runs_tools(local_backup):
    local_backup.target.version_code = "23ai"
    with pytest.raises(DrillRejected) as rejected:
        await _preflight(local_backup)
    assert rejected.value.code == "BACKUP_PRESET_UNSUPPORTED"
    local_backup.tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_preflight_does_not_use_password_or_fallback_account(local_backup):
    local_backup.credentials.read.return_value = {"auth_mode": "DATABASE_PASSWORD", "username": "oracle", "password": "secret"}
    with pytest.raises(DrillRejected):
        await _preflight(local_backup)
    local_backup.tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_preflight_refuses_symlink_and_unapproved_directory(local_backup, tmp_path):
    different = tmp_path / "other"
    different.mkdir()
    alias = local_backup.root / "alias"
    alias.symlink_to(different, target_is_directory=True)
    for path in (alias, different):
        local_backup.settings = ProtectionPlanSettings(tool="RMAN", backup_directory=str(path))
        with pytest.raises(DrillRejected):
            await _preflight(local_backup)
    local_backup.tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_this_worker_never_claims_drills_or_other_execution_hosts(local_backup):
    jobs = ProtectionJobs(uow_factory=local_backup.uow_factory, backend=local_backup.backend)
    assert await jobs.claim("DRILL") is None
    local_backup.uow.protection_plans.due_plan.assert_not_awaited()
    assert await jobs.claim("BACKUP") is None
    assert local_backup.uow.protection_plans.due_plan.await_args.kwargs["resource_ids"] == {local_backup.resource_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["19c", "26ai"])
@pytest.mark.parametrize("fault", ["none", "base_missing", "scn_mismatch"])
async def test_oracle_cumulative_incremental_requires_matching_level_zero(local_backup, version, fault):
    case = local_backup
    case.settings = ProtectionPlanSettings.model_validate({
        **case.settings.model_dump(), "backup_schedule": {"timezone": "UTC", "cron": "0 2 * * 0"},
        "backup_incremental_schedule": {"timezone": "UTC", "cron": "0 1 * * 1-6"},
    })
    now, run_id = datetime.now(UTC), uuid7()
    base = case.root / f"aiops-{uuid7()}"
    assert "INCREMENTAL LEVEL 0 DATABASE" in render_backup(case.settings, uuid7(), db_version=version).command_file
    base.mkdir()
    piece = base / "base.bkp"
    piece.write_bytes(b"baseline")
    artifact = BackupArtifact(
        backup_storage_key(case.settings), base.name, now - timedelta(days=1), True,
        fingerprint=piece_manifest(base),
    )
    if fault == "base_missing":
        piece.unlink()
    work = ProtectionWork(
        run_id, case.resource_id, case.scope, "BACKUP", now, now + timedelta(hours=1),
        {"settings": case.settings.model_dump(mode="json"), "backup_kind": "INCREMENTAL",
         "approval": {"binding": {"db_version": version}}}, artifact,
    )
    invocation = render_backup(case.settings, run_id, db_version=version, kind="INCREMENTAL", baseline=artifact)

    async def command(config, executable, args, script):
        if executable == "sqlplus":
            if "b.checkpoint_change#=d.incremental_change#" in script:
                assert str(base) in script and "s.incremental_level=1" in script
                output = INVENTORY.replace("AIOPS_SET|21|D", "AIOPS_SET|21|I") + "\nAIOPS_RECOVERY_SCN|1234\n"
                return 0, output.replace("AIOPS_DATAFILES|YES", "AIOPS_DATAFILES|NO") if fault == "scn_mismatch" else output
            assert str(base) in script and "s.incremental_level=0" in script
            return 0, INVENTORY
        if script.startswith("RUN"):
            assert "INCREMENTAL LEVEL 1 CUMULATIVE DATABASE" in script
            (Path(invocation.output_directory) / "changes.bkp").write_bytes(b"mock changes")
            return 0, ""
        return 0, "validation complete\n" * 4

    case.tool.side_effect = command
    result = await case.backend.backup(work, invocation)
    if fault == "none":
        assert result.artifact.kind == "INCREMENTAL" and result.artifact.dependencies == (base.name,)
        assert result.artifact.baseline_fingerprint == artifact.fingerprint
        assert result.artifact.minimum_scn == 1234
    else:
        assert result.artifact is None
        if fault == "base_missing":
            case.tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("db_version", ["19c", "26ai"])
@pytest.mark.parametrize("result", [
    "good", "metadata_only", "missing_datafile", "missing_archive", "missing_controlfile",
    "missing_spfile", "missing_files", "bad_validation", "missing_validation", "rman_error",
])
async def test_backup_requires_native_files_and_readback_before_success(local_backup, result, db_version):
    local_backup.target.version_code = db_version
    if db_version == "26ai":
        local_backup.tool.return_value = (0, READY.replace("19.3.0.0.0", "23.26.1.0.0"))
    binding = await _preflight(local_backup)
    assert binding.execution_identity and binding.destination is None
    run_id, now = uuid7(), datetime.now(UTC)
    work = ProtectionWork(
        run_id, local_backup.resource_id, local_backup.scope, "BACKUP", now, now + timedelta(hours=1),
        {"settings": local_backup.settings.model_dump(mode="json"), "approval": {"binding": {"db_version": binding.db_version}}},
    )
    invocation = render_backup(local_backup.settings, run_id, db_version=db_version)

    async def command(config, executable, args, script):
        if executable == "sqlplus":
            assert str(invocation.output_directory) in script
            substitutions = {
                "metadata_only": ("AIOPS_SET|21|D|NO|NO|YES\n", ""),
                "missing_datafile": ("AIOPS_DATAFILES|YES", "AIOPS_DATAFILES|NO"),
                "missing_archive": ("AIOPS_SET|22|L|NO|NO|NO\n", ""),
                "missing_controlfile": ("AIOPS_SET|23|D|YES|NO|NO\n", ""),
                "missing_spfile": ("AIOPS_SET|24|D|NO|YES|NO\n", ""),
            }
            return (0, INVENTORY.replace(*substitutions[result]) if result in substitutions else INVENTORY)
        assert args == ("target", '"/ as sysbackup"', "nocatalog")
        if script.startswith("RUN {"):
            assert script == invocation.command_file
            if result != "missing_files":
                (Path(invocation.output_directory) / "fixture.bkp").write_bytes(b"mock backup")
            return (0, "RMAN-00569: ERROR" if result == "rman_error" else "backup completed")
        assert script.startswith("VALIDATE BACKUPSET 21, 22, 23, 24;")
        if result == "missing_validation":
            return (0, 3 * "validation complete\n")
        return (1, "corrupted") if result == "bad_validation" else (0, 4 * "validation complete\n")

    local_backup.tool.side_effect = command
    actual = await local_backup.backend.backup(work, invocation)
    if result == "good":
        assert actual.exit_code == 0 and actual.artifact.checksum_verified
        assert actual.artifact.reference == f"aiops-{run_id}"
        assert actual.continuous_logs_verified is None
    else:
        assert actual.exit_code != 0 and actual.artifact is None

