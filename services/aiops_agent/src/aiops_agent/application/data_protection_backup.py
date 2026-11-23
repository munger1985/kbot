"""代码内置的源端备份模板。只生成命令，不读取凭据、不执行工具。"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from re import fullmatch
from uuid import UUID

from aiops_agent.application.data_protection_execution import BackupArtifact, FILE_BACKUP_REFERENCE
from platform_core.contracts.aiops.data_protection import ProtectionBackupKind, ProtectionPlanSettings, ProtectionTool


@dataclass(frozen=True)
class BackupPreset:
    key: str
    revision: str
    tool_requirement: str


# 只按业务系统登记的精确版本选预设。命令相同可共用生成逻辑，工具版本不能互换。
_PRESETS = {
    ("RMAN", "19c"): BackupPreset("rman.oracle19c.backup", "1", "Oracle RMAN 19c"),
    ("RMAN", "26ai"): BackupPreset("rman.oracle26ai.backup", "1", "Oracle RMAN 26ai"),
    ("XTRABACKUP", "5.7"): BackupPreset("xtrabackup.mysql57.backup", "1", "Percona XtraBackup 2.4"),
    ("XTRABACKUP", "8.0"): BackupPreset("xtrabackup.mysql80.backup", "1", "Percona XtraBackup 8.0"),
    ("XTRABACKUP", "8.4"): BackupPreset("xtrabackup.mysql84.backup", "1", "Percona XtraBackup 8.4"),
    **{
        ("PGBACKREST", version): BackupPreset(
            "pgbackrest.postgresql.backup", "1", f"支持 PostgreSQL {version} 的 pgBackRest",
        )
        for version in ("16", "17", "18")
    },
}


@dataclass(frozen=True)
class BackupInvocation:
    tool: ProtectionTool
    preset: BackupPreset
    arguments: tuple[str, ...]
    # RMAN 连接由受控适配负责，命令文件中不包含认证材料。
    command_file: str | None
    # 对 RMAN 和 XtraBackup 必须使用每次运行的专属目录。
    output_directory: str | None
    kind: ProtectionBackupKind = "FULL"


def backup_template(tool: ProtectionTool, db_version: str | None) -> BackupPreset | None:
    return _PRESETS.get((tool, db_version))


def backup_storage_key(settings: ProtectionPlanSettings) -> str | None:
    if settings.storage_key:
        return settings.storage_key
    if settings.backup_directory:
        return "directory-" + sha256(settings.backup_directory.encode()).hexdigest()
    return None


def validate_backup_parameters(settings: ProtectionPlanSettings) -> None:
    if settings.tool == "PGBACKREST":
        if not settings.storage_key:
            raise ValueError("pgBackRest 需要已配置的 stanza（使用其 repo1）")
        if settings.backup_directory:
            raise ValueError("pgBackRest 的目录由 stanza 的 repo1 配置管理，不能由备份计划覆盖")
    elif not settings.backup_directory:
        raise ValueError("RMAN 和 XtraBackup 需要填写备份目录")


def render_work_backup(settings, work, *, db_version):
    return render_backup(
        settings, work.run_id, db_version=db_version, kind=work.snapshot.get("backup_kind", "FULL"),
        baseline=work.backup if work.operation == "BACKUP" else None,
    )


def render_backup(
    settings: ProtectionPlanSettings, run_id: UUID, *, db_version: str | None,
    kind: ProtectionBackupKind = "FULL", baseline: BackupArtifact | None = None,
) -> BackupInvocation:
    """输入仅为验证后的字段。程序／身份／实际仓库绑定由预检确定。"""
    preset = backup_template(settings.tool, db_version)
    if preset is None:
        raise ValueError("当前数据库版本没有内置备份预设")
    validate_backup_parameters(settings)
    incremental = kind == "INCREMENTAL"
    if incremental and (
        not settings.incremental_enabled or baseline is None or not baseline.checksum_verified
        or baseline.storage_key != backup_storage_key(settings)
        or (settings.tool != "PGBACKREST" and (
            baseline.kind != "FULL" or not fullmatch(FILE_BACKUP_REFERENCE, baseline.reference)
        ))
    ):
        raise ValueError("缺少已核验的增量基线")
    parallelism = settings.backup_parallelism
    if settings.tool == "PGBACKREST":
        # 首版取已登记的 stanza/repo1。不得临时覆盖仓库路径，否则可能脱离归档链。
        return BackupInvocation(
            settings.tool, preset,
            (f"--stanza={settings.storage_key}", "--repo=1", f"--process-max={parallelism}",
             "--type=incr" if incremental else "--type=full", "backup"),
            None, None, kind,
        )

    directory = str(PurePosixPath(settings.backup_directory) / f"aiops-{run_id}")
    if settings.tool == "XTRABACKUP":
        extra = (f"--incremental-basedir={PurePosixPath(settings.backup_directory) / baseline.reference}",) if incremental else ()
        return BackupInvocation(
            settings.tool, preset, ("--backup", f"--target-dir={directory}", f"--parallel={parallelism}", *extra),
            None, directory, kind,
        )

    channels = [f"  ALLOCATE CHANNEL dp{i} DEVICE TYPE DISK FORMAT '{directory}/%U.bkp';"
                for i in range(1, parallelism + 1)]
    tag = "AIOPS_" + run_id.hex[-24:].upper()
    level = ""
    if settings.incremental_enabled:
        level = " INCREMENTAL LEVEL 1 CUMULATIVE" if incremental else " INCREMENTAL LEVEL 0"
    command_file = "\n".join((
        "RUN {",
        f"  SET CONTROLFILE AUTOBACKUP FORMAT FOR DEVICE TYPE DISK TO '{directory}/%F';",
        *channels,
        f"  BACKUP AS BACKUPSET{level} DATABASE TAG '{tag}' PLUS ARCHIVELOG;",
        f"  BACKUP CURRENT CONTROLFILE TAG '{tag}';",
        f"  BACKUP SPFILE TAG '{tag}';",
        "}",
        "EXIT;",
        "",
    ))
    return BackupInvocation(settings.tool, preset, (), command_file, directory, kind)

