"""离线生成固定恢复步骤。不会打开实例、运行工具或绕过空库预检。"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from re import fullmatch
from uuid import UUID

from aiops_agent.application.data_protection_backup import backup_storage_key, backup_template
from aiops_agent.application.data_protection_execution import (
    DrillRejected, DrillRequest, FILE_BACKUP_REFERENCE, valid_backup_chain,
)
from platform_core.contracts.aiops.data_protection import ProtectionPlanSettings, ProtectionTool


@dataclass(frozen=True)
class RestorePreset:
    key: str
    revision: str = "1"


@dataclass(frozen=True)
class OracleMedia:
    # 辅助实例已核验为 NOMOUNT；增量时在演练端另建仅含所选链的工作目录。
    backup_directory: str
    auxiliary_database: str
    staging_directory: str | None = None


@dataclass(frozen=True)
class PostgreSQLMedia:
    # 呼叫端须校验该配置只有批准的只读仓库及本机路径，没有链接或恢复选项覆盖。
    repository_config: str
    repository_directory: str
    tablespace_directory: str


@dataclass(frozen=True)
class MySQLMedia:
    # 已核验的可写工作副本，包括安全的备份参数文件；禁止在原始仓库执行 prepare。
    backup_directory: str
    staging_directory: str
    incremental_directory: str | None = None
    incremental_staging_directory: str | None = None


@dataclass(frozen=True)
class RestoreStep:
    arguments: tuple[str, ...]
    command_file: str | None = None


@dataclass(frozen=True)
class RestoreInvocation:
    tool: ProtectionTool
    preset: RestorePreset
    steps: tuple[RestoreStep, ...]


def restore_preset(tool: ProtectionTool, db_version: str | None, target_version: str | None) -> RestorePreset | None:
    """仅同代目标，具体工具补丁兼容性仍由实际执行端核验。"""
    if db_version != target_version or (backup := backup_template(tool, db_version)) is None:
        return None
    return RestorePreset(backup.key.removesuffix(".backup") + ".restore")


def _path(value: str) -> PurePosixPath:
    if not fullmatch(r"(?:/[a-zA-Z0-9_.-]+)+", value):
        raise DrillRejected("恢复位置必须是已核验的规范绝对路径", code="DESTINATION_CHANGED")
    path = PurePosixPath(value)
    if ".." in path.parts or str(path) != value:
        raise DrillRejected("恢复位置包含非规范路径", code="DESTINATION_CHANGED")
    return path


def _separate(*paths: PurePosixPath) -> None:
    for index, path in enumerate(paths):
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths[:index]):
            raise DrillRejected("演练写入路径与仓库或其它目录重合", code="DESTINATION_CHANGED")


def _backup_directory(raw: str, reference: str, settings: ProtectionPlanSettings) -> PurePosixPath:
    if not fullmatch(FILE_BACKUP_REFERENCE, reference):
        raise DrillRejected("没有本任务可用的原生备份引用", code="BACKUP_UNAVAILABLE")
    path = _path(raw)
    if not settings.backup_directory or path != _path(settings.backup_directory) / reference:
        raise DrillRejected("备份目录与已选产物不符", code="BACKUP_UNAVAILABLE")
    return path


def render_restore(
    settings: ProtectionPlanSettings, request: DrillRequest, *,
    db_version: str, target_version: str, run_id: UUID,
    media: OracleMedia | PostgreSQLMedia | MySQLMedia,
) -> RestoreInvocation:
    """仅在既有演练门禁之后消费渲染结果；不能以参数校验替代现场核验。"""
    tool, backup = request.tool, request.backup
    preset = restore_preset(tool, db_version, target_version)
    if (tool != settings.tool or preset is None or settings.restore_template_key != preset.key
        or settings.drill_destination_key != request.destination.key):
        raise DrillRejected("演练模板、源或目标版本与批准配置不符", code="BINDING_CHANGED")
    if (not request.storage_key or request.storage_key != backup_storage_key(settings)
        or request.storage_key != backup.storage_key or backup.checksum_verified is not True
        or request.simulated_failure_at.utcoffset() is None or backup.available_at.utcoffset() is None
        or backup.available_at > request.simulated_failure_at):
        raise DrillRejected("没有 T0 时已存在且匹配仓库的备份", code="BACKUP_UNAVAILABLE")
    destination = _path(request.destination.data_path)
    if not valid_backup_chain(tool, backup):
        raise DrillRejected("原生备份链不完整或引用无效", code="BACKUP_UNAVAILABLE")

    if tool == "RMAN" and isinstance(media, OracleMedia):
        directory = _backup_directory(media.backup_directory, backup.reference, settings)
        _separate(destination, directory.parent)
        if backup.kind == "INCREMENTAL":
            if not media.staging_directory:
                raise DrillRejected("没有已批准的 RMAN 链工作目录", code="BACKUP_UNAVAILABLE")
            stage = _path(media.staging_directory)
            _separate(stage, destination, directory.parent)
            if stage != destination.parent / f"aiops-{run_id}":
                raise DrillRejected("RMAN 链工作目录与本次演练不符", code="DESTINATION_CHANGED")
            directory = stage
        if not fullmatch(r"[A-Z][A-Z0-9_]{0,7}", media.auxiliary_database):
            raise DrillRejected("辅助数据库标识不符合批准模板", code="DESTINATION_CHANGED")
        # 只连辅助实例；不读源库、不恢复源 SPFILE，文件路径由已核验的辅助实例 OMF 产生。
        command = f"DUPLICATE DATABASE TO {media.auxiliary_database} BACKUP LOCATION '{directory}' NOOPEN;\nEXIT;\n"
        return RestoreInvocation(tool, preset, (RestoreStep(("auxiliary",), command),))

    if tool == "PGBACKREST" and isinstance(media, PostgreSQLMedia):
        if settings.backup_directory or not settings.storage_key:
            raise DrillRejected("没有匹配 stanza 的 pgBackRest 备份链", code="BACKUP_UNAVAILABLE")
        config = _path(media.repository_config)
        tablespaces = _path(media.tablespace_directory)
        _separate(destination, tablespaces, config, _path(media.repository_directory))
        # 恢复至所选备份能达到的一致点；不承诺 WAL 覆盖到 T0。
        args = (
            f"--config={config}", f"--stanza={request.storage_key}", "--repo=1",
            f"--set={backup.reference}", f"--pg1-path={destination}", f"--tablespace-map-all={tablespaces}",
            "--type=immediate", "--target-action=pause", "--no-link-all", "--no-delta", "restore",
        )
        return RestoreInvocation(tool, preset, (RestoreStep(args),))

    if tool == "XTRABACKUP" and isinstance(media, MySQLMedia):
        incremental = backup.kind == "INCREMENTAL"
        base = backup.dependencies[0] if incremental else backup.reference
        directory = _backup_directory(media.backup_directory, base, settings)
        copy = _path(media.staging_directory)
        _separate(destination, directory.parent, copy)
        if copy.name != f"aiops-{run_id}":
            raise DrillRejected("XtraBackup 工作副本与本次演练不符", code="DESTINATION_CHANGED")
        defaults = f"--defaults-file={copy / 'backup-my.cnf'}"
        prepare = (defaults, "--prepare", f"--target-dir={copy}")
        if incremental:
            if not (media.incremental_directory and media.incremental_staging_directory):
                raise DrillRejected("缺少已核验的 XtraBackup 增量工作副本", code="BACKUP_UNAVAILABLE")
            _backup_directory(media.incremental_directory, backup.reference, settings)
            delta = _path(media.incremental_staging_directory)
            _separate(destination, directory.parent, copy, delta)
            if delta != copy.with_name(f"{copy.name}-incremental"):
                raise DrillRejected("增量工作副本与本次演练不符", code="DESTINATION_CHANGED")
            steps = (
                RestoreStep((*prepare, "--apply-log-only")),
                RestoreStep((*prepare, f"--incremental-dir={delta}")),
            )
        else:
            steps = (RestoreStep(prepare),)
        return RestoreInvocation(tool, preset, (
            *steps,
            RestoreStep((defaults, "--copy-back", f"--target-dir={copy}", f"--datadir={destination}")),
        ))

    raise DrillRejected("备份介质类型与已批准模板不符", code="BACKUP_UNAVAILABLE")
