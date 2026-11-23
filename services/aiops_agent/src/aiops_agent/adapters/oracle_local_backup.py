"""受控的本机 Oracle 19c / 26ai 源端备份。不依赖 Docker，也不提供恢复入口。"""

import asyncio
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import re
from uuid import UUID

from aiops_agent.application.data_protection_backup import (
    BackupInvocation, backup_storage_key, backup_template, render_work_backup, validate_backup_parameters,
)
from aiops_agent.application.data_protection_execution import BackupArtifact, DrillRejected
from aiops_agent.application.data_protection_jobs import ApprovedProtection, BackupResult, ProtectionWork
from aiops_agent.application.managed_credentials import AIOpsManagedCredentialError
from aiops_agent.config import AIOpsOracleLocalBackupConfig
from platform_core.contracts.aiops.configuration import DATABASE_VERSION_PREFIXES
from platform_core.contracts.aiops.data_protection import ProtectionPlanSettings


_SQL_PREFIX = """whenever oserror exit failure
whenever sqlerror exit failure
set heading off feedback off pagesize 0 linesize 32767 trimout on verify off define off
"""
ORACLE_BACKUP_PROBE = """select 'AIOPS_FACTS|' || d.dbid || '|' || d.db_unique_name || '|' ||
       d.log_mode || '|' || d.database_role || '|' || d.open_mode || '|' ||
       i.version_full || '|' || d.cdb || '|' || sys_context('USERENV','CON_NAME') || '|' ||
       (select case when value is null then 'NO' else 'YES' end from v$parameter where name='spfile') || '|' ||
       (select upper(value) from v$parameter where name='cluster_database')
from v$database d cross join v$instance i;"""
_ORACLE_ERROR = re.compile(r"(?m)^[ \t]*(?:RMAN-00569|ORA-\d{5}|SP2-\d{4}):")
_VALIDATION_COMPLETE = "validation complete"
_CREDENTIAL_KIND = "protection_backup"


def _current_os_user() -> str:
    import pwd

    return pwd.getpwuid(os.geteuid()).pw_name


async def _run_local(config: AIOpsOracleLocalBackupConfig, executable: str, args: tuple[str, ...], script: str):
    environment = {
        "ORACLE_HOME": config.oracle_home,
        "ORACLE_SID": config.oracle_sid,
        "PATH": f"{config.oracle_home}/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"{config.oracle_home}/lib",
        "NLS_LANG": "AMERICAN_AMERICA.AL32UTF8",
    }
    process = await asyncio.create_subprocess_exec(
        str(Path(config.oracle_home) / "bin" / executable), *args, env=environment, cwd=config.oracle_home,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await process.communicate(script.encode())
    except asyncio.CancelledError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        raise
    # 只交给固定结构的解析函数；不输出或持久化原始日志。
    return process.returncode, output.decode(errors="replace")


def _checked_directory(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or not path.name or path.resolve(strict=True) != path or not path.is_dir():
        raise DrillRejected("不是已登记的规范本机目录")
    return path


def oracle_backup_inventory(directory: Path, *, level: int | None = None, baseline: Path | None = None) -> str:
    # 目录来自产品参数和本次 UUID，插入 SQL 前仍严格检查。使用前缀等值而非 LIKE。
    if not re.fullmatch(r"(/[a-zA-Z0-9_.-]+)+", str(directory)) or ".." in directory.parts:
        raise ValueError("备份产物目录格式无效")
    prefix = f"{directory}/"
    level_filter = {
        None: "s.backup_type='D'", 0: "s.backup_type in ('D','I') and s.incremental_level=0",
        1: "s.backup_type='I' and s.incremental_level=1",
    }[level]
    parent_filter = ""
    if baseline is not None:
        if level != 1 or not re.fullmatch(r"(/[a-zA-Z0-9_.-]+)+", str(baseline)) or ".." in baseline.parts:
            raise ValueError("增量基线目录格式无效")
        base = f"{baseline}/"
        # 累计 Level 1 必须从该 Level 0 的检查点开始，且属于同一文件与数据库 incarnation。
        parent_filter = f"""and exists (
              select 1 from v$backup_datafile b join v$backup_piece bp
                on b.set_stamp=bp.set_stamp and b.set_count=bp.set_count
              where b.incremental_level=0 and b.file#=d.file#
                and b.creation_change#=d.creation_change# and b.resetlogs_change#=d.resetlogs_change#
                and b.checkpoint_change#=d.incremental_change#
                and bp.status='A' and bp.deleted='NO' and substr(bp.handle, 1, length('{base}'))='{base}'
            )"""
    scn_query = ""
    if level == 1:
        scn_query = f"""
select 'AIOPS_RECOVERY_SCN|' || max(d.checkpoint_change#)
from v$backup_datafile d join v$backup_set s on d.set_stamp=s.set_stamp and d.set_count=s.set_count
join v$backup_piece p on s.set_stamp=p.set_stamp and s.set_count=p.set_count
where d.file#>0 and ({level_filter}) and p.status='A' and p.deleted='NO'
  and substr(p.handle, 1, length('{prefix}'))='{prefix}';"""
    return f"""select distinct 'AIOPS_SET|' || s.recid || '|' || s.backup_type || '|' ||
       s.controlfile_included || '|' ||
       case when exists (select 1 from v$backup_spfile f where f.set_stamp=s.set_stamp and f.set_count=s.set_count)
            then 'YES' else 'NO' end || '|' ||
       case when exists (select 1 from v$backup_datafile d
                         where d.set_stamp=s.set_stamp and d.set_count=s.set_count and d.file#>0 and ({level_filter}))
            then 'YES' else 'NO' end
from v$backup_set s join v$backup_piece p on s.set_stamp=p.set_stamp and s.set_count=p.set_count
where p.status='A' and p.deleted='NO' and substr(p.handle, 1, length('{prefix}'))='{prefix}';
select 'AIOPS_DATAFILES|' || case
       when exists (select 1 from v$datafile f)
        and not exists (
          select 1 from v$datafile f
          where not exists (
            select 1 from v$backup_datafile d
            join v$backup_set s on s.set_stamp=d.set_stamp and s.set_count=d.set_count
            join v$backup_piece p on p.set_stamp=s.set_stamp and p.set_count=s.set_count
            where d.file#=f.file# and d.creation_change#=f.creation_change# and ({level_filter})
              and d.resetlogs_change#=(select resetlogs_change# from v$database)
              {parent_filter}
              and p.status='A' and p.deleted='NO' and substr(p.handle, 1, length('{prefix}'))='{prefix}'
          )
        ) then 'YES' else 'NO' end from dual;{scn_query}"""


def _backupsets(output: str, *, level: int | None = None) -> tuple[int, ...]:
    coverage = [line.strip() for line in output.splitlines() if line.strip().startswith("AIOPS_DATAFILES|")]
    if coverage != ["AIOPS_DATAFILES|YES"]:
        return ()
    sets = [match.groups() for line in output.splitlines()
            if (match := re.fullmatch(r"AIOPS_SET\|(\d+)\|([DIL])\|(YES|NO)\|(YES|NO)\|(YES|NO)", line.strip()))]
    if (not any(kind == "L" for _, kind, _, _, _ in sets)
        or not any(ctrl == "YES" for _, _, ctrl, _, _ in sets)
        or not any(spfile == "YES" for _, _, _, spfile, _ in sets)
        or not any(kind in ({None: {"D"}, 0: {"D", "I"}, 1: {"I"}}[level]) and data == "YES"
                   for _, kind, _, _, data in sets)):
        return ()
    return tuple(sorted({int(key) for key, _, _, _, _ in sets if int(key) > 0}))


def piece_manifest(directory: Path) -> str:
    """RMAN 已执行原生 VALIDATE；清单用于后续发现备份片缺失或大小变化。"""
    if not directory.is_dir() or directory.resolve(strict=True) != directory:
        raise ValueError("RMAN 备份目录不符")
    entries = []
    for piece in sorted(directory.iterdir()):
        if not piece.is_file() or piece.resolve(strict=True) != piece or piece.stat().st_size <= 0:
            raise ValueError("RMAN 备份片不可读取")
        entries.append((piece.name, piece.stat().st_size))
    if not entries:
        raise ValueError("RMAN 备份片缺失")
    return hashlib.sha256(repr(entries).encode()).hexdigest()


class OracleLocalBackupBackend:
    def __init__(self, *, bindings, uow_factory, credentials, run_local=_run_local, os_user=_current_os_user):
        self.backup_resource_ids = frozenset(bindings)
        self._bindings = bindings
        self._uow_factory = uow_factory
        self._credentials = credentials
        self._run = run_local
        self._os_user = os_user

    async def _sqlplus(self, config, query: str) -> str:
        code, out = await self._run(config, "sqlplus", ("-L", "-s", "/ as sysbackup"),
                                    _SQL_PREFIX + query + "\nexit success\n")
        if code or _ORACLE_ERROR.search(out):
            raise DrillRejected("本机 SYSBACKUP 预检或备份目录校验失败")
        return out

    def _parameters(self, resource_id: UUID, settings: ProtectionPlanSettings):
        config = self._bindings.get(resource_id)
        if config is None or settings.tool != "RMAN":
            raise DrillRejected("未登记本机 Oracle 执行位置")
        try:
            validate_backup_parameters(settings)
            home = _checked_directory(config.oracle_home)
            root = _checked_directory(config.backup_root)
            directory = _checked_directory(settings.backup_directory)
        except (OSError, ValueError) as exc:
            raise DrillRejected("Oracle Home 或备份目录不可用") from exc
        if root.is_relative_to(home) or home.is_relative_to(root) or not directory.is_relative_to(root):
            raise DrillRejected("备份目录不在 DBA 登记的独立存储范围内")
        if not os.access(directory, os.W_OK | os.X_OK):
            raise DrillRejected("本机 Oracle 执行身份无法写入备份目录")
        if not all((binary := home / "bin" / name).is_file() and os.access(binary, os.X_OK)
                   for name in ("sqlplus", "rman")):
            raise DrillRejected("指定 Oracle Home 缺少可执行工具")
        return config, directory

    async def _scoped_source(self, *, scope: dict, resource_id: UUID, target_id: UUID) -> tuple[int, str, str, str]:
        async with self._uow_factory() as uow:
            resource = await uow.business_systems.get_resource_scoped(resource_id=resource_id, **scope)
            if resource is None or resource.resource_type_key != "DATABASE" or resource.lifecycle_status != "ACTIVE":
                raise DrillRejected("备份源数据库资源不再有效")
            profile = await uow.business_systems.get_database_profile(resource_id)
            target = await uow.targets.get_scoped(target_id=target_id, domain_id=scope["domain_id"])
            if (not profile or profile.target_id != target_id or not target or target.status != "ENABLED"
                or target.db_type != "ORACLE" or target.deployment_type != "SINGLE_INSTANCE"
                or target.db_role != "PRIMARY" or target.oracle_container_scope not in ("CDB_ROOT", "NON_CDB")
                or ("tenant_id" in scope and target.tenant_id != scope["tenant_id"])):
                raise DrillRejected("登记实例与当前单实例 Oracle 主库配置不符")
            row = await uow.managed_credentials.find(
                **scope, namespace="aiops", credential_kind=_CREDENTIAL_KIND, external_key=str(resource_id),
            )
            if row is None or row.status != "ACTIVE":
                raise DrillRejected("专用备份身份不可用")
            try:
                identity = await self._credentials.read(
                    **scope, uow=uow, credential_id=row.credential_id, credential_kind=_CREDENTIAL_KIND,
                    external_key=resource_id,
                )
            except AIOpsManagedCredentialError:
                raise DrillRejected("专用备份身份不可用") from None
            username = identity.get("username")
            if identity.get("auth_mode") != "ORACLE_OS" or not username or username != self._os_user():
                raise DrillRejected("必须使用登记的 OS SYSBACKUP 用户运行 AIOps")
            return int(target.row_version), target.oracle_container_scope, username, target.version_code

    async def preflight(self, *, scope, actor_id, resource_id, settings, operation) -> ApprovedProtection:
        if operation != "BACKUP":
            raise DrillRejected("本机 Oracle 备份适配没有恢复权限")
        config, directory = self._parameters(resource_id, settings)
        revision, container_scope, username, db_version = await self._scoped_source(
            scope=scope, resource_id=resource_id, target_id=config.target_id,
        )
        preset = backup_template(settings.tool, db_version)
        if preset is None:
            raise DrillRejected("当前 Oracle 版本没有内置 RMAN 预设", code="BACKUP_PRESET_UNSUPPORTED")
        rows = [line.strip().split("|")[1:] for line in (await self._sqlplus(config, ORACLE_BACKUP_PROBE)).splitlines()
                if line.strip().startswith("AIOPS_FACTS|")]
        if len(rows) != 1 or len(rows[0]) != 10:
            raise DrillRejected("无法确认 Oracle 实例身份")
        dbid, unique_name, archives, role, open_mode, version, cdb, container, spfile, clustered = rows[0]
        if dbid != str(config.dbid) or unique_name.casefold() != config.db_unique_name.casefold():
            raise DrillRejected("当前实例身份与 DBA 登记不符")
        if not version.startswith(DATABASE_VERSION_PREFIXES[("ORACLE", db_version)]):
            raise DrillRejected("实际 Oracle 版本与业务系统登记不符", code="BACKUP_VERSION_MISMATCH")
        if archives != "ARCHIVELOG":
            raise DrillRejected("整库在线备份需要 ARCHIVELOG", code="ORACLE_ARCHIVELOG_REQUIRED")
        if (role != "PRIMARY" or open_mode != "READ WRITE" or spfile != "YES"
            or clustered != "FALSE"
            or (container_scope == "CDB_ROOT" and (cdb, container) != ("YES", "CDB$ROOT"))
            or (container_scope == "NON_CDB" and cdb != "NO")):
            raise DrillRejected("当前实例不属于本机 Oracle 在线备份范围")
        identity = hashlib.sha256(
            (config.model_dump_json() + str(directory) + version + username).encode(),
        ).hexdigest()
        return ApprovedProtection(
            config.target_id, revision, settings.tool, preset.key, preset.revision,
            backup_storage_key(settings), db_version, execution_identity=identity,
        )

    async def backup(self, work: ProtectionWork, invocation: BackupInvocation) -> BackupResult:
        settings = ProtectionPlanSettings.model_validate(work.snapshot["settings"])
        config, parent = self._parameters(work.resource_id, settings)
        db_version = work.snapshot["approval"]["binding"].get("db_version")
        if work.operation != "BACKUP" or settings.tool != "RMAN" or backup_template(settings.tool, db_version) is None:
            raise DrillRejected("拒绝未确认 Oracle 版本的备份任务")
        expected = render_work_backup(settings, work, db_version=db_version)
        if invocation != expected or not invocation.command_file:
            raise DrillRejected("拒绝未从内置 Oracle 模板生成的调用")
        directory = parent / f"aiops-{work.run_id}"
        if str(directory) != invocation.output_directory:
            raise DrillRejected("备份目录与计划不一致")
        baseline = None
        level = None
        if settings.incremental_enabled:
            level = 1 if invocation.kind == "INCREMENTAL" else 0
        if level == 1:
            baseline = parent / work.backup.reference
            try:
                if (piece_manifest(baseline) != work.backup.fingerprint
                    or not _backupsets(await self._sqlplus(config, oracle_backup_inventory(baseline, level=0)), level=0)):
                    raise ValueError
            except (OSError, ValueError, DrillRejected):
                return BackupResult(1, None, message="RMAN 无法核验本周 Level 0 基线及数据文件覆盖")
        directory.mkdir(mode=0o700, exist_ok=False)
        args = ("target", '"/ as sysbackup"', "nocatalog")
        code, output = await self._run(config, "rman", args, invocation.command_file)
        if code or _ORACLE_ERROR.search(output):
            return BackupResult(code or 1, None)

        try:
            inventory = await self._sqlplus(
                config, oracle_backup_inventory(directory, level=level, baseline=baseline),
            )
            keys = _backupsets(inventory, level=level)
        except DrillRejected:
            return BackupResult(1, None)
        minimum_scn = None
        if level == 1:
            scns = re.findall(r"(?m)^\s*AIOPS_RECOVERY_SCN\|([1-9][0-9]{0,19})\s*$", inventory)
            if len(scns) != 1:
                return BackupResult(1, None)
            minimum_scn = int(scns[0])
        if not keys or not any(item.is_file() and item.stat().st_size > 0 for item in directory.iterdir()):
            return BackupResult(1, None)
        validate = f"VALIDATE BACKUPSET {', '.join(map(str, keys))};\nEXIT;\n"
        code, output = await self._run(config, "rman", args, validate)
        if code or _ORACLE_ERROR.search(output) or output.count(_VALIDATION_COMPLETE) < len(keys):
            return BackupResult(code or 1, None)
        try:
            manifest = piece_manifest(directory) if level is not None else None
        except (OSError, ValueError):
            return BackupResult(1, None)
        return BackupResult(0, BackupArtifact(
            storage_key=backup_storage_key(settings), reference=directory.name, available_at=datetime.now(UTC),
            checksum_verified=True, kind=invocation.kind, dependencies=(baseline.name,) if baseline else (),
            fingerprint=manifest, baseline_fingerprint=work.backup.fingerprint if baseline else None,
            minimum_scn=minimum_scn,
        ))

    async def drill_runner(self, work):
        raise DrillRejected("本机 Oracle 备份适配没有恢复入口")
