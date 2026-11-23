"""XtraBackup 本机全量与基于计划全量的增量。不在源端 prepare。"""

import asyncio
from configparser import ConfigParser, Error as IniError
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import file_digest, sha256
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import aiomysql

from aiops_agent.application.data_protection_execution import BackupArtifact, DrillRejected
from aiops_agent.application.data_protection_jobs import BackupResult
from platform_core.contracts.aiops.configuration import DATABASE_VERSION_PREFIXES


_TOOL_FAMILY = {"5.7": "2.4", "8.0": "8.0", "8.4": "8.4"}
_REQUIRED_GRANTS = {"PROCESS", "RELOAD", "LOCK TABLES", "REPLICATION CLIENT"}
_LAYOUT_VARIABLES = (
    "innodb_data_file_path", "innodb_data_home_dir", "innodb_log_group_home_dir", "innodb_undo_directory",
)
_INFO_FILES = ("xtrabackup_checkpoints", "xtrabackup_info", "backup-my.cnf", "xtrabackup_logfile")


def _globals(grants):
    result = set()
    for (row,) in grants:
        match = re.match(r"GRANT (.+?) ON \*\.\* TO ", row, re.I)
        if match:
            result.update(privilege.strip().upper() for privilege in match[1].split(","))
    return result


async def probe(config, source):
    try:
        async with asyncio.timeout(10):
            connection = await aiomysql.connect(
                unix_socket=config.socket, user=source.username, password=source.password,
                connect_timeout=8, autocommit=True,
            )
            try:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT @@version, @@server_uuid, @@datadir, @@socket, @@global.read_only, "
                        "@@global.super_read_only, CURRENT_USER()",
                    )
                    facts = await cursor.fetchone()
                    version, _, _, _, _, _, current_user = facts
                    if not version.startswith(DATABASE_VERSION_PREFIXES[("MYSQL", source.db_version)]):
                        raise DrillRejected("实际 MySQL 版本与登记不符", code="BACKUP_VERSION_MISMATCH")
                    if current_user.rsplit("@", 1)[0] != source.username:
                        raise DrillRejected("MySQL 未使用登记的专用备份身份")
                    modern = source.db_version != "5.7"
                    await cursor.execute("SHOW GRANTS")
                    grants = _globals(await cursor.fetchall())
                    required = _REQUIRED_GRANTS | ({"BACKUP_ADMIN"} if modern else set())
                    if "ALL PRIVILEGES" in grants:
                        grants |= _REQUIRED_GRANTS
                    if not required <= grants:
                        raise DrillRejected("MySQL 专用账号缺少此预设所需的直接全局备份授权")
                    await cursor.execute("SHOW REPLICA STATUS" if source.db_version == "8.4" else "SHOW SLAVE STATUS")
                    if await cursor.fetchone() is not None:
                        raise DrillRejected("本机备份预设不包含 MySQL 副本")
                    await cursor.execute(
                        "SELECT 1 FROM information_schema.PLUGINS "
                        "WHERE PLUGIN_NAME = 'group_replication' AND PLUGIN_STATUS = 'ACTIVE'",
                    )
                    if await cursor.fetchone() is not None:
                        raise DrillRejected("本机备份预设不包含 MySQL 组复制实例")
                    if modern:
                        await cursor.execute("SELECT 1 FROM performance_schema.log_status LIMIT 0")
                        await cursor.execute("SELECT 1 FROM performance_schema.replication_group_members LIMIT 0")
                        if source.db_version == "8.4":
                            await cursor.execute("SELECT 1 FROM performance_schema.keyring_component_status LIMIT 0")
                    await cursor.execute("SHOW GLOBAL VARIABLES WHERE Variable_name IN %s", (_LAYOUT_VARIABLES,))
                    layout = dict(await cursor.fetchall())
                    if modern:
                        await cursor.execute("SELECT PATH FROM information_schema.INNODB_DATAFILES")
                    else:
                        await cursor.execute("SELECT PATH FROM information_schema.INNODB_SYS_DATAFILES")
                    datafiles = tuple(path for (path,) in await cursor.fetchall())
                    return facts, layout, datafiles
            finally:
                connection.close()
    except (aiomysql.Error, OSError, TimeoutError, ValueError):
        raise DrillRejected("MySQL 专用身份连接失败，或无法读取备份预检所需信息") from None


def _datafiles(directory, version, layout, paths):
    if (layout.get("innodb_data_file_path", "").split(":")[0] != "ibdata1"
        or ";" in layout["innodb_data_file_path"]):
        raise DrillRejected("本机 XtraBackup 仅支持单文件系统表空间 ibdata1")
    allowed_locations = {"", ".", "./", str(directory), f"{directory}/"}
    if any(layout.get(key, "") not in allowed_locations for key in _LAYOUT_VARIABLES[1:]):
        raise DrillRejected("本机 XtraBackup 不支持 datadir 外的 InnoDB 数据、redo 或 undo")
    names = {Path("ibdata1")}
    if version != "5.7":
        names.add(Path("mysql.ibd"))
    for raw in paths:
        if not isinstance(raw, str) or not raw:
            raise DrillRejected("无法确认 InnoDB 表空间文件")
        path = Path(raw)
        if not path.is_absolute():
            path = directory / path
        if (not path.is_relative_to(directory) or path.resolve(strict=True) != path
            or not path.is_file() or not os.access(path, os.R_OK)):
            raise DrillRejected("InnoDB 表空间不在可读取的已登记 datadir 内")
        names.add(path.relative_to(directory))
    for name in names:
        path = directory / name
        if path.resolve(strict=True) != path or not path.is_file() or not os.access(path, os.R_OK):
            raise DrillRejected("本机身份无法读取登记 datadir 的 InnoDB 文件")
    return frozenset(names)


def _settings_file(config, source):
    def quoted(value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t") + '"'

    return ("[xtrabackup]\n"
            f"user={quoted(source.username)}\npassword={quoted(source.password)}\n"
            f"socket={quoted(config.socket)}\n")


def _read_metadata(path):
    result = {}
    # 此文件中的原始工具参数永不进入日志或证据。
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            key, sep, value = line.partition("=")
            key = key.strip()
            if sep and key:
                if key in result:
                    raise ValueError("XtraBackup 元数据格式无效")
                result[key] = value.strip()
    return result


def _required_artifacts(directory, facts):
    def regular(path):
        return path.resolve(strict=True) == path and path.is_file() and path.stat().st_size > 0

    if not all(regular(directory / name) for name in _INFO_FILES):
        return ()
    details = _read_metadata(directory / "xtrabackup_info")
    if (details.get("server_version") != facts.server_version
        or details.get("format") != "file" or details.get("partial") != "N"
        or details.get("incremental") != "N" or details.get("compressed") != "N"
        or details.get("encrypted", "N") != "N"
        or datetime.fromisoformat(details["end_time"]) < datetime.fromisoformat(details["start_time"])):
        return ()
    pages = {directory / name for name in facts.datafiles}
    # 同时覆盖备份中新增的 InnoDB 表文件；不跟随任何符号链接。
    pages.update(directory.rglob("*.ibd"))
    pages.update(directory.rglob("*.ibu"))
    if not pages or not all(regular(path) for path in pages):
        return ()
    return tuple(sorted(pages))


def checkpoints(directory, *, incremental):
    path = directory / "xtrabackup_checkpoints"
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("XtraBackup 检查点不是仓库中的普通文件")
    metadata = _read_metadata(path)
    values = [metadata[key] for key in ("from_lsn", "to_lsn", "last_lsn")]
    if metadata.get("backup_type") != ("incremental" if incremental else "full-backuped") or not all(
        re.fullmatch(r"[0-9]{1,20}", value) for value in values
    ):
        raise ValueError("XtraBackup 检查点格式不符")
    start, end, last = map(int, values)
    if not (0 <= start < end <= last and (start > 0 if incremental else start == 0)):
        raise ValueError("XtraBackup 检查点范围不符")
    return start, end


def incremental_fingerprint(directory):
    """读回所有增量文件。delta 是 XtraBackup 容器，不能当数据文件执行 innochecksum。"""
    digest = sha256()
    deltas = set()
    metadata = set()
    for path in sorted(directory.rglob("*")):
        if path.resolve(strict=True) != path or (not path.is_dir() and not path.is_file()):
            raise ValueError("增量介质含外部链接或非普通文件")
        if path.is_dir():
            continue
        name = path.relative_to(directory).as_posix()
        if path.suffix == ".delta":
            if path.stat().st_size == 0:
                raise ValueError("XtraBackup delta 为空")
            deltas.add(name.removesuffix(".delta"))
        if path.suffix == ".meta":
            values = _read_metadata(path)
            if "page_size" in values:
                if not re.fullmatch(r"[0-9]+", values["page_size"]) or int(values["page_size"]) not in (
                    4096, 8192, 16384, 32768, 65536,
                ):
                    raise ValueError("XtraBackup delta 页大小无效")
                metadata.add(name.removesuffix(".meta"))
        with path.open("rb") as stream:
            digest.update(name.encode("utf-8") + b"\0" + file_digest(stream, "sha256").digest())
    if not deltas or deltas != metadata or not all(
        (directory / name).is_file() and (directory / name).stat().st_size > 0 for name in _INFO_FILES
    ):
        raise ValueError("XtraBackup 增量缺少成对的 delta、meta 或原生备份元数据")
    return digest.hexdigest()


@dataclass(frozen=True)
class _Facts:
    identity: tuple
    server_version: str
    datafiles: frozenset[Path]


class XtraBackupLocal:
    def __init__(self, config, run, read_probe):
        self.config, self.run, self.read_probe = config, run, read_probe

    async def inspect(self, source):
        config = self.config
        code, output = await self.run(config.executable, ("--no-defaults", "--version"))
        tool = re.search(
            r"(?m)^(?:\S+/)?xtrabackup version (\d+\.\d+\.\d+\S*) based on MySQL server (\d+\.\d+\.\d+\S*) ",
            output,
        )
        if (code or not tool or not tool[1].startswith(_TOOL_FAMILY[source.db_version] + ".")
            or not tool[2].startswith(source.db_version + ".")):
            raise DrillRejected("本机 XtraBackup 工具版本与登记的 MySQL 预设不兼容")
        code, output = await self.run(config.checksum_executable, ("--version",))
        checksum = re.search(r"\binnochecksum\s+(?:Ver|version)\s+(\d+\.\d+\.\d+\S*)", output, re.I)
        if code or not checksum or not checksum[1].startswith(source.db_version + "."):
            raise DrillRejected("缺少与登记版本匹配的 MySQL innochecksum")
        live, layout, paths = await self.read_probe(config, source)
        version, server_uuid, datadir, socket, read_only, super_read_only, _ = live
        if not version.startswith(DATABASE_VERSION_PREFIXES[("MYSQL", source.db_version)]) or "mariadb" in version.lower():
            raise DrillRejected("实际 MySQL 版本与业务系统登记不符", code="BACKUP_VERSION_MISMATCH")
        if (server_uuid.casefold() != str(config.server_uuid) or datadir.rstrip("/") != config.data_directory
            or socket != config.socket or read_only != 0 or super_read_only != 0):
            raise DrillRejected("MySQL 实例身份、目录或主库状态与登记不符")
        directory = Path(config.data_directory)
        try:
            ini = ConfigParser(interpolation=None)
            with (directory / "auto.cnf").open(encoding="utf-8") as info:
                ini.read_file(info)
            if ini.get("auto", "server-uuid").casefold() != str(config.server_uuid):
                raise ValueError
        except (IniError, ValueError):
            raise DrillRejected("无法通过 datadir 的 auto.cnf 核验 MySQL 实例身份") from None
        datafiles = _datafiles(directory, source.db_version, layout, paths)
        return _Facts((tool[1], tool[2], checksum[1], version, sorted(layout.items())), version, datafiles)

    async def backup(self, work, invocation, source, facts, storage_key):
        config = self.config
        directory = Path(invocation.output_directory)
        incremental = invocation.kind == "INCREMENTAL"
        baseline = None
        if incremental:
            baseline = directory.parent / work.backup.reference
            try:
                if baseline.resolve(strict=True) != baseline:
                    raise ValueError
                _, base_end = checkpoints(baseline, incremental=False)
                if (base_end != work.backup.end_lsn
                    or not all((baseline / name).is_file() and (baseline / name).resolve(strict=True) == baseline / name
                               and (baseline / name).stat().st_size > 0 for name in (*_INFO_FILES, "ibdata1"))
                    or _read_metadata(baseline / "xtrabackup_info")["server_version"] != facts.server_version):
                    raise ValueError
            except (OSError, ValueError, KeyError):
                return BackupResult(1, None, message="XtraBackup 全量基线介质缺失或检查点不一致")
        directory.mkdir(mode=0o700, exist_ok=False)
        with NamedTemporaryFile(mode="w", encoding="utf-8") as defaults:
            defaults.write(_settings_file(config, source))
            defaults.flush()
            code, _ = await self.run(
                config.executable, (
                    f"--defaults-file={defaults.name}", f"--datadir={config.data_directory}",
                    *invocation.arguments,
                ),
                env={"MYSQL_TEST_LOGIN_FILE": defaults.name + ".no-login"}, capture=False,
            )
        if code:
            return BackupResult(code, None)
        if incremental:
            try:
                start, end = checkpoints(directory, incremental=True)
                if start != base_end:
                    raise ValueError
                digest = await asyncio.to_thread(incremental_fingerprint, directory)
                details = _read_metadata(directory / "xtrabackup_info")
                if (details.get("server_version") != facts.server_version or details.get("incremental") != "Y"
                    or details.get("format") != "file" or details.get("partial") != "N"
                    or details.get("compressed") != "N" or details.get("encrypted", "N") != "N"):
                    raise ValueError
            except (OSError, KeyError, ValueError):
                return BackupResult(1, None, message="XtraBackup 增量文件或 LSN 未通过核验")
            return BackupResult(0, BackupArtifact(
                storage_key, directory.name, datetime.now(UTC), True, kind=invocation.kind,
                dependencies=(baseline.name,), fingerprint=digest, end_lsn=end, start_lsn=start,
            ))
        try:
            pages = _required_artifacts(directory, facts)
            _, end = checkpoints(directory, incremental=False)
        except (OSError, KeyError, ValueError):
            return BackupResult(1, None)
        if not pages:
            return BackupResult(1, None)
        for path in pages:
            code, _ = await self.run(config.checksum_executable, (str(path),), capture=False)
            if code:
                return BackupResult(code, None)
        return BackupResult(0, BackupArtifact(
            storage_key=storage_key, reference=directory.name, available_at=datetime.now(UTC),
            checksum_verified=True, end_lsn=end,
        ))
