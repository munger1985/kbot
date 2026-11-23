"""pgBackRest 本机备份与仓库读回。只使用已登记 stanza 的 repo1。"""

import asyncio
from configparser import ConfigParser, Error as IniError
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import asyncpg

from aiops_agent.application.data_protection_execution import BackupArtifact, DrillRejected, PG_FULL_LABEL, PG_INCREMENTAL_LABEL
from aiops_agent.application.data_protection_jobs import BackupResult


_PROBE = """
select current_setting('server_version_num')::int as version,
       current_setting('data_directory') as directory,
       current_setting('archive_mode') as archive_mode,
       current_setting('archive_command') as archive_command,
       current_setting('archive_library') as archive_library,
       inet_server_addr() is null as local_socket,
       not pg_is_in_recovery() as is_primary,
       (select system_identifier::text from pg_control_system()) as system_identifier,
       has_function_privilege('pg_catalog.pg_backup_start(text,boolean)', 'EXECUTE')
       and has_function_privilege('pg_catalog.pg_backup_stop(boolean)', 'EXECUTE')
       and has_function_privilege('pg_catalog.pg_switch_wal()', 'EXECUTE')
       and has_function_privilege('pg_catalog.pg_create_restore_point(text)', 'EXECUTE') as can_backup
"""
_WAL_SEGMENT = re.compile(r"[0-9A-F]{24}")
_REPO_STATUS_INITIALIZED = {0, 2}  # 2: stanza 已建但尚无备份。


async def probe(config, source):
    try:
        async with asyncio.timeout(10):
            connection = await asyncpg.connect(
                host=config.socket_directory, port=config.port, database="postgres",
                user=source.username, password=source.password, ssl=False, timeout=8, command_timeout=8,
                server_settings={"search_path": "pg_catalog"},
            )
            try:
                return await connection.fetchrow(_PROBE)
            finally:
                await connection.close(timeout=1)
    except (asyncpg.PostgresError, OSError, TimeoutError, ValueError):
        raise DrillRejected("PostgreSQL 专用身份连接失败，或无法读取备份所需系统信息") from None


def _registered_repo(config) -> str:
    """显式 --config 不读取默认 include；只接受本机单 PG、POSIX repo1。"""
    raw = Path(config.config_file).read_bytes()
    parser = ConfigParser(interpolation=None, delimiters=("=",))
    try:
        parser.read_string(raw.decode())
    except (IniError, UnicodeError):
        raise DrillRejected("无法解析登记的 pgBackRest 配置") from None
    if parser.defaults() or not parser.has_section(config.stanza):
        raise DrillRejected("登记的 pgBackRest stanza 配置不存在")
    repo_path_set = False
    for section in parser.sections():
        if section.split(":", 1)[0] not in {"global", config.stanza}:
            continue
        for option, value in parser.items(section):
            if (re.match(r"(?:repo\d*|pg\d*|db\d*)-host(?:$|-)", option)
                or re.match(r"(?:pg|db)(?!1-)\d+-", option)
                or option in {"config-path", "config-include-path"}):
                raise DrillRejected("仅支持本机 PostgreSQL 与本机 pgBackRest 仓库")
            if option in {"repo1-path", "repo-path"}:
                if value != config.backup_root:
                    raise DrillRejected("pgBackRest repo1 与登记目录不符")
                if section in {"global", config.stanza}:
                    repo_path_set = True
            if option in {"repo1-type", "repo-type"} and value != "posix":
                raise DrillRejected("只支持本机 POSIX 格式 pgBackRest 仓库")
    if not repo_path_set:
        raise DrillRejected("pgBackRest 配置必须显式指定已登记的 repo1 路径")
    return sha256(raw).hexdigest()


def _repository(output, config, db_version):
    try:
        stanzas = json.loads(output)
        if not isinstance(stanzas, list) or len(stanzas) != 1:
            raise ValueError
        stanza = stanzas[0]
        repos = stanza["repo"]
        if (stanza["name"] != config.stanza or stanza["status"]["code"] not in _REPO_STATUS_INITIALIZED
            or len(repos) != 1 or repos[0]["key"] != 1
            or repos[0]["status"]["code"] not in _REPO_STATUS_INITIALIZED):
            raise ValueError
        # info 按旧到新列出仓库数据库历史；只接受当前一代。
        db = [item for item in stanza["db"] if item["repo-key"] == 1][-1]
        if str(db["system-id"]) != config.system_identifier or str(db["version"]) != db_version:
            raise ValueError
        if not isinstance(stanza["backup"], list):
            raise ValueError
        return db["id"], stanza["backup"]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        raise DrillRejected("无法核验已登记 stanza/repo1 的数据库身份及备份清单") from None


def _valid_backup(backup, db_id, kind):
    try:
        label, stamp, archive = backup["label"], backup["timestamp"], backup["archive"]
        return bool((re.fullmatch(PG_FULL_LABEL if kind == "full" else PG_INCREMENTAL_LABEL, label)
            and backup["type"] == kind and backup.get("error") is not True
            and backup["database"] == {"id": db_id, "repo-key": 1}
            and backup["info"]["size"] > 0 and 0 < stamp["start"] <= stamp["stop"]
            and _WAL_SEGMENT.fullmatch(archive["start"]) and _WAL_SEGMENT.fullmatch(archive["stop"])
            and archive["start"][:8] == archive["stop"][:8] and archive["start"] <= archive["stop"]))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def verified_chain(backups, db_id, references):
    catalog = {}
    for item in backups:
        if not isinstance(item, dict):
            return False
        label = item.get("label")
        if not isinstance(label, str) or label in catalog:
            return False
        catalog[label] = item
    seen = set()
    for index, label in enumerate(references):
        item = catalog.get(label)
        if item is None or not _valid_backup(item, db_id, "incr" if index else "full"):
            return False
        if index:
            refs = item.get("reference")
            if (item.get("prior") != references[index - 1] or not isinstance(refs, list)
                or not all(isinstance(ref, str) and ref in seen for ref in refs)):
                return False
        seen.add(label)
    return bool(references)


def _backup_label(backups, run_id, db_id, dependencies=()):
    matches = [item.get("label") for item in backups if (item.get("annotation") or {}).get("aiops-run") == str(run_id)]
    if len(matches) != 1 or not isinstance(matches[0], str):
        return None
    label = matches[0]
    return label if verified_chain(backups, db_id, (*dependencies, label)) else None


def _verified(output, stanza, label):
    lines = [line.strip() for line in output.splitlines()]
    matches = [line for line in lines if line.startswith(f"backup: {label},")]
    if lines[:2] != [f"stanza: {stanza}", "status: ok"] or len(matches) != 1:
        return False
    counts = re.fullmatch(
        rf"backup: {re.escape(label)}, status: valid, total files checked: ([1-9][0-9]*), total valid files: \1",
        matches[0],
    )
    return counts is not None


def _passfile(config, source):
    def escape(value):
        return value.replace("\\", "\\\\").replace(":", "\\:").replace("*", "\\*")

    # libpq 对默认 socket 目录可能使用 localhost 匹配。实际连接由固定 pg1 socket/port 参数决定。
    return f"*:{config.port}:postgres:{escape(source.username)}:{escape(source.password)}\n"


@dataclass(frozen=True)
class _Facts:
    identity: tuple
    db_id: int
    backups: list = field(repr=False)


class PgBackRestLocal:
    def __init__(self, config, run, read_probe):
        self.config, self.run, self.read_probe = config, run, read_probe
        self.common = (
            f"--config={config.config_file}", "--log-level-file=off",
            "--log-level-console=off", "--log-level-stderr=off",
        )
        self.repository_args = (f"--stanza={config.stanza}", "--repo=1")

    async def _info(self, db_version):
        code, output = await self.run(
            self.config.executable, self.common + self.repository_args + ("--output=json", "info"),
        )
        if code:
            raise DrillRejected("pgBackRest repo1 无法读取")
        return _repository(output, self.config, db_version)

    async def inspect(self, source):
        config = self.config
        digest = _registered_repo(config)
        data = Path(config.data_directory)
        if (not (Path(config.socket_directory) / f".s.PGSQL.{config.port}").is_socket()
            or not os.access(data / "global" / "pg_control", os.R_OK)
            or (data / "PG_VERSION").read_text().strip() != source.db_version):
            raise DrillRejected("PostgreSQL socket、数据目录或控制文件不符合登记")
        code, output = await self.run(
            config.executable, ("--version",),
        )
        version = re.fullmatch(r"pgBackRest (\d+)\.(\d+)\.(\d+)", output.strip())
        if code or not version or tuple(map(int, version.groups())) < (2, 55, 0):
            raise DrillRejected("本机 pgBackRest 需为支持当前预设及定向 verify 的 2.55.0 或更新版本")
        live = await self.read_probe(config, source)
        if not live or not isinstance(live["version"], int) or str(live["version"] // 10000) != source.db_version:
            raise DrillRejected("实际 PostgreSQL 版本与登记不符", code="BACKUP_VERSION_MISMATCH")
        if (live["directory"] != config.data_directory or live["system_identifier"] != config.system_identifier
            or live["local_socket"] is not True or live["is_primary"] is not True):
            raise DrillRejected("PostgreSQL 实例身份、角色或数据目录与登记不符")
        if (live["archive_mode"] != "on" or not (live["archive_command"] or live["archive_library"])
            or live["can_backup"] is not True):
            raise DrillRejected("未确认 PostgreSQL 归档设置或专用账号的备份权限")
        db_id, backups = await self._info(source.db_version)
        return _Facts((digest, version.group(0), live["version"], db_id), db_id, backups)

    async def backup(self, work, invocation, source, facts, storage_key):
        config = self.config
        if any(isinstance(item, dict) and (item.get("annotation") or {}).get("aiops-run") == str(work.run_id)
               for item in facts.backups):
            raise DrillRejected("仓库中已存在本次任务标记，拒绝重复备份")
        dependencies = ()
        if invocation.kind == "INCREMENTAL":
            dependencies = (*work.backup.dependencies, work.backup.reference)
            current_labels = [item.get("label") for item in facts.backups
                              if item.get("database") == {"id": facts.db_id, "repo-key": 1}]
            if (not current_labels or current_labels[-1] != work.backup.reference
                or not verified_chain(facts.backups, facts.db_id, dependencies)
                or not all((directory := Path(config.backup_root) / "backup" / config.stanza / ref).is_dir()
                           and directory.resolve(strict=True) == directory for ref in dependencies)):
                return BackupResult(1, None, message="pgBackRest 仓库的最新备份不是本计划已核验的链尾")
        args = (
            f"--pg1-path={config.data_directory}", f"--pg1-socket-path={config.socket_directory}",
            f"--pg1-port={config.port}", "--pg1-database=postgres", f"--pg1-user={source.username}",
            "--online", "--backup-standby=n", "--archive-check", "--archive-mode-check", "--no-expire-auto",
            f"--annotation=aiops-run={work.run_id}",
        )
        with NamedTemporaryFile(mode="w", encoding="utf-8") as passfile:
            passfile.write(_passfile(config, source))
            passfile.flush()
            code, _ = await self.run(
                config.executable, self.common + args + invocation.arguments,
                env={"PGPASSFILE": passfile.name}, capture=False,
            )
        if code:
            return BackupResult(code, None)
        try:
            db_id, backups = await self._info(source.db_version)
        except DrillRejected:
            return BackupResult(1, None)
        label = _backup_label(backups, work.run_id, db_id, dependencies) if db_id == facts.db_id else None
        if label is None:
            return BackupResult(1, None)
        code, output = await self.run(
            config.executable,
            self.common + self.repository_args + (f"--set={label}", "--output=text", "--verbose", "verify"),
        )
        if code or not _verified(output, config.stanza, label):
            return BackupResult(code or 1, None)
        return BackupResult(0, BackupArtifact(
            storage_key=storage_key, reference=label, available_at=datetime.now(UTC), checksum_verified=True,
            kind=invocation.kind, dependencies=dependencies,
        ))
