"""内置备份模板与入参回归（不运行数据库工具）。"""

import pytest
from pydantic import ValidationError

from aiops_agent.application.data_protection_backup import backup_storage_key, backup_template, render_backup
from platform_core.contracts.aiops.data_protection import ProtectionPlanSettings
from platform_core.identity import uuid7


@pytest.mark.parametrize("path", ["/", "relative/path", "/srv/../prod", "/srv/./backup", "//srv/backup",
                                  "/srv/backup/", "/srv/b%U", "/srv/b'backup", "/srv/b\n", "/srv/back;up"])
def test_untrusted_directory_cannot_become_a_command(path):
    with pytest.raises(ValidationError):
        ProtectionPlanSettings(tool="RMAN", backup_directory=path)


@pytest.mark.parametrize("parallelism", [0, 33, True, 1.5, "4"])
def test_parallelism_must_be_an_integer_in_range(parallelism):
    with pytest.raises(ValidationError):
        ProtectionPlanSettings(tool="RMAN", backup_parallelism=parallelism)


def test_rman_backup_only_uses_its_run_directory_parallel_channels_and_no_persistent_config():
    run_id = uuid7()
    settings = ProtectionPlanSettings(tool="RMAN", backup_directory="/srv/backup/oracle", backup_parallelism=3)
    invocation = render_backup(settings, run_id, db_version="19c")
    assert invocation.preset.key == "rman.oracle19c.backup"
    assert invocation.output_directory == f"/srv/backup/oracle/aiops-{run_id}"
    assert invocation.command_file.count("ALLOCATE CHANNEL ") == 3
    assert invocation.command_file.count(f"FORMAT '{invocation.output_directory}/%U.bkp'") == 3
    for marker in ("PLUS ARCHIVELOG", "BACKUP CURRENT CONTROLFILE", "BACKUP SPFILE"):
        assert marker in invocation.command_file
    for forbidden in ("CONFIGURE", "DELETE", "RESTORE", "CONNECT", "RECOVER"):
        assert forbidden not in invocation.command_file
    assert backup_storage_key(settings).startswith("directory-")
    assert render_backup(settings, uuid7(), db_version="19c").output_directory != invocation.output_directory


def test_pgbackrest_preserves_stanza_repository_and_archive_setup():
    settings = ProtectionPlanSettings(tool="PGBACKREST", storage_key="pg-prod", backup_parallelism=4)
    for version in ("16", "17", "18"):
        invocation = render_backup(settings, uuid7(), db_version=version)
        assert invocation.preset.key == "pgbackrest.postgresql.backup"
        assert invocation.preset.tool_requirement == f"支持 PostgreSQL {version} 的 pgBackRest"
        assert invocation.arguments == ("--stanza=pg-prod", "--repo=1", "--process-max=4", "--type=full", "backup")
        assert invocation.command_file is None and invocation.output_directory is None
    with pytest.raises(ValueError, match="不能由备份计划覆盖"):
        render_backup(ProtectionPlanSettings(
            tool="PGBACKREST", storage_key="pg-prod", backup_directory="/srv/other",
        ), uuid7(), db_version="16")


def test_xtrabackup_only_generates_a_backup_not_prepare_or_copy_back():
    run_id = uuid7()
    settings = ProtectionPlanSettings(tool="XTRABACKUP", backup_directory="/srv/mysql", backup_parallelism=2)
    v57 = render_backup(settings, run_id, db_version="5.7")
    v80 = render_backup(settings, run_id, db_version="8.0")
    v84 = render_backup(settings, run_id, db_version="8.4")
    assert v57.arguments == v80.arguments == v84.arguments == (
        "--backup", f"--target-dir=/srv/mysql/aiops-{run_id}", "--parallel=2",
    )
    assert v57.command_file is v80.command_file is v84.command_file is None
    assert len({v57.preset.key, v80.preset.key, v84.preset.key}) == 3
    assert v57.preset.tool_requirement == "Percona XtraBackup 2.4"
    assert v80.preset.tool_requirement == "Percona XtraBackup 8.0"
    assert v84.preset.tool_requirement == "Percona XtraBackup 8.4"


@pytest.mark.parametrize(("tool", "db_version"), [("RMAN", "19c"), ("PGBACKREST", "16"), ("XTRABACKUP", "5.7")])
def test_missing_storage_or_directory_rejects_build(tool, db_version):
    with pytest.raises(ValueError):
        render_backup(ProtectionPlanSettings(tool=tool), uuid7(), db_version=db_version)


@pytest.mark.parametrize(("tool", "db_version"), [
    ("RMAN", "23ai"), ("RMAN", None), ("RMAN", "8.4"), ("PGBACKREST", "19"), ("XTRABACKUP", "8"),
])
def test_no_fallback_for_unlisted_database_version(tool, db_version):
    assert backup_template(tool, db_version) is None
    with pytest.raises(ValueError, match="没有内置备份预设"):
        render_backup(ProtectionPlanSettings(tool=tool), uuid7(), db_version=db_version)

