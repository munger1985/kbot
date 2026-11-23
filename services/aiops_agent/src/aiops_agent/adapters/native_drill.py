"""独立主机上固定的三库恢复步骤。只使用已绑定的本机实例与只读介质。"""

import asyncio
from configparser import ConfigParser
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
import re
import shutil
from time import monotonic
from types import SimpleNamespace

import aiomysql
import asyncpg

from aiops_agent.adapters import mysql_local_backup, oracle_local_backup, postgresql_local_backup
from aiops_agent.application.data_protection_execution import (
    DrillRejected, DrillVerification, FILE_BACKUP_REFERENCE, valid_backup_chain,
)
from aiops_agent.application.data_protection_restore import MySQLMedia, OracleMedia, PostgreSQLMedia
from platform_core.contracts.aiops.configuration import DATABASE_VERSION_PREFIXES


_ENGINE_DIRECTORY = {"RMAN": (), "PGBACKREST": ("run", "tablespaces"), "XTRABACKUP": ("run", "stage")}
_BACKUP_REFERENCE = re.compile(FILE_BACKUP_REFERENCE)
_ORACLE_ERRORS = re.compile(r"(?m)(?:ORA-\d{5}|SP2-\d{4}|RMAN-00569|RMAN-03002):")


def workspace_directories(tool):
    return _ENGINE_DIRECTORY[tool]


def _executable(raw):
    path = Path(raw)
    if not path.is_file() or path.resolve(strict=True) != path or not os.access(path, os.X_OK):
        raise DrillRejected("登记的本机演练工具不可执行")
    return str(path)


def _media_directory(root, reference, read_only_mount):
    if not _BACKUP_REFERENCE.fullmatch(reference):
        raise DrillRejected("备份引用无效", code="BACKUP_UNAVAILABLE")
    directory = root / reference
    if (not directory.is_dir() or directory.resolve(strict=True) != directory
        or not read_only_mount(directory)):
        raise DrillRejected("本次备份目录不在只读挂载上", code="BACKUP_UNAVAILABLE")
    return directory


def _safe_file(path):
    return path.is_file() and path.resolve(strict=True) == path and path.stat().st_size > 0


def _check_stage_space(stage, data, copied_bytes, required_free_bytes):
    if stage.exists() or stage.is_symlink():
        raise DrillRejected("本次演练工作副本已经存在", code="NOT_EMPTY")
    space = os.statvfs(stage.parent)
    same_volume = stage.parent.stat().st_dev == data.stat().st_dev
    if space.f_bavail * space.f_frsize < copied_bytes + (required_free_bytes if same_volume else 0):
        raise DrillRejected("演练工作副本空间不足")


def _known_failure(restore_started, retrieval_seconds=None):
    return DrillVerification(
        False, None, None, None, retrieval_seconds=retrieval_seconds,
        restore_seconds=int(monotonic() - restore_started),
    )


@asynccontextmanager
async def _start_local(executable, args, socket):
    """只终止本次直接启动的实例，绝不根据外部 PID 停库。"""
    process = await asyncio.create_subprocess_exec(
        executable, *args, env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(40):
            while not socket.is_socket():
                if process.returncode is not None:
                    raise DrillRejected("演练实例启动未成功")
                await asyncio.sleep(0.2)
        yield
    finally:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            async with asyncio.timeout(15):
                await process.wait()


class _Engine:
    def __init__(self, config, site, credential, db_version, settings, run, run_oracle):
        self.config, self.site, self.credential = config, site, credential
        self.db_version, self.settings = db_version, settings
        self.run, self.run_oracle = run, run_oracle


class _Oracle(_Engine):
    def _connect_string(self):
        user, password = self.credential.username, self.credential.password
        if (not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_$#]*", user)
            or not re.fullmatch(r"""[a-zA-Z0-9_!#$%*+=,.@:/?-]{1,128}""", password)):
            raise DrillRejected("演练身份字符不符合 Oracle 本机工具认证规则")
        return f'{user}/"{password}" AS SYSBACKUP'

    async def _sql(self, query):
        code, output = await self.run_oracle(self.config, "sqlplus", ("-L", "-s", "/nolog"), (
            "whenever oserror exit failure\nwhenever sqlerror exit failure\n"
            "set heading off feedback off pagesize 0 linesize 32767 verify off define off\n"
            + "CONNECT " + self._connect_string() + "\n" + query + "\nexit success\n"
        ))
        if code or _ORACLE_ERRORS.search(output):
            raise DrillRejected("辅助实例认证、状态或验证不符合要求")
        return [line.strip() for line in output.splitlines() if line.strip().startswith("DRILL|")]

    async def inspect(self):
        c, data = self.config, self.site.data
        for tool in ("sqlplus", "rman"):
            _executable(Path(c.oracle_home) / "bin" / tool)
        rows = await self._sql(
            "select 'DRILL|INSTANCE|'||instance_name||'|'||status||'|'||version_full from v$instance;\n"
            "select 'DRILL|'||name||'|'||value from v$parameter where name in "
            "('db_name','db_unique_name','db_create_file_dest','control_files','db_recovery_file_dest',"
            "'db_file_name_convert','log_file_name_convert','cluster_database','job_queue_processes',"
            "'dg_broker_start','local_listener','remote_listener','log_archive_dest','diagnostic_dest','audit_file_dest')"
            " or name like 'db_create_online_log_dest_%' or name like 'log_archive_dest_%';"
        )
        info = {}
        for row in rows:
            _, key, value = row.split("|", 2)
            if key in info:
                raise DrillRejected("辅助实例参数存在歧义")
            info[key] = value
        instance, status, version = info["INSTANCE"].split("|")
        if (instance.casefold() != c.oracle_sid.casefold() or status != "STARTED"
            or not version.startswith(DATABASE_VERSION_PREFIXES[("ORACLE", self.db_version)])
            or info.get("db_name", "").upper() != c.auxiliary_database
            or info.get("db_unique_name", "").upper() != c.auxiliary_database
            or info.get("db_create_file_dest") != str(data)
            or info.get("cluster_database") != "FALSE" or info.get("dg_broker_start") != "FALSE"
            or info.get("job_queue_processes") != "0"):
            raise DrillRejected("Oracle 不是登记的独立 NOMOUNT 辅助实例", code="DESTINATION_CHANGED")
        for key, value in info.items():
            if key in {"db_file_name_convert", "log_file_name_convert", "remote_listener", "log_archive_dest"} \
                    or re.fullmatch(r"log_archive_dest_\d+", key):
                if value:
                    raise DrillRejected("辅助实例不得继承源库路径、远程监听或归档目的地")
            if key == "db_recovery_file_dest" or key.startswith("db_create_online_log_dest_"):
                if value not in {"", str(data)}:
                    raise DrillRejected("辅助实例的联机日志或 FRA 不在专用目录")
        for key in ("diagnostic_dest", "audit_file_dest"):
            folder = Path(info.get(key, ""))
            if (not folder.is_absolute() or folder.resolve(strict=True) != folder or not folder.is_dir()
                or not folder.is_relative_to(self.site.root) or folder.is_relative_to(data)):
                raise DrillRejected("Oracle 诊断与审计目录必须在独立演练工作区")
        controls = info.get("control_files", "").split(",")
        if not controls or not all(
            re.fullmatch(re.escape(str(data)) + r"/[a-zA-Z0-9_-]+\.ctl", item.strip()) for item in controls
        ):
            raise DrillRejected("控制文件不在已核验的空目录")
        return instance, version, tuple(sorted(info.items()))

    async def media(self, artifact, run_id, read_only_mount):
        directory = _media_directory(self.site.backup_root, artifact.reference, read_only_mount)
        if not valid_backup_chain("RMAN", artifact):
            raise DrillRejected("RMAN 备份链无效", code="BACKUP_UNAVAILABLE")
        copied_bytes, names = 0, set()
        for ref in (*artifact.dependencies, artifact.reference):
            source = _media_directory(self.site.backup_root, ref, read_only_mount)
            pieces = tuple(source.iterdir())
            if not pieces or any(not _safe_file(piece) or not read_only_mount(piece) or piece.name in names
                                 for piece in pieces):
                raise DrillRejected("RMAN 备份片缺失、不可读或文件名冲突", code="BACKUP_UNAVAILABLE")
            names.update(piece.name for piece in pieces)
            copied_bytes += sum(piece.stat().st_size for piece in pieces)
        stage = None
        if artifact.dependencies:
            if (oracle_local_backup.piece_manifest(directory) != artifact.fingerprint
                or oracle_local_backup.piece_manifest(self.site.backup_root / artifact.dependencies[0])
                != artifact.baseline_fingerprint):
                raise DrillRejected("RMAN 基线或增量备份片与已验证的清单不一致", code="BACKUP_UNAVAILABLE")
            stage = self.site.root / f"aiops-{run_id}"
            _check_stage_space(stage, self.site.data, copied_bytes, self.config.required_free_bytes)
        return OracleMedia(str(directory), self.config.auxiliary_database, str(stage) if stage else None)

    async def restore(self, invocation, media, request):
        retrieval = None
        if media.staging_directory:
            retrieved_at = monotonic()
            stage = Path(media.staging_directory)
            stage.mkdir(mode=0o700, exist_ok=False)

            def copy_pieces():
                for ref in (*request.backup.dependencies, request.backup.reference):
                    for piece in (self.site.backup_root / ref).iterdir():
                        with piece.open("rb") as src, (stage / piece.name).open("xb") as dst:
                            shutil.copyfileobj(src, dst)

            await asyncio.to_thread(copy_pieces)
            retrieval = int(monotonic() - retrieved_at)
        started = monotonic()
        command = f"CONNECT AUXILIARY '{self._connect_string()}';\n" + invocation.steps[0].command_file
        code, output = await self.run_oracle(self.config, "rman", (), command)
        if code or _ORACLE_ERRORS.search(output):
            return _known_failure(started, retrieval)
        data = self.site.data
        files = await self._sql(
            "select 'DRILL|'||name from v$datafile union all select 'DRILL|'||name from v$tempfile "
            "union all select 'DRILL|'||member from v$logfile union all select 'DRILL|'||name from v$controlfile;"
        )
        if not files or any(
            not Path(line.split("|", 1)[1]).resolve().is_relative_to(data) for line in files
        ):
            raise DrillRejected("已恢复的 Oracle 文件不在演练目录，拒绝打开")
        await self._sql(
            "alter system enable restricted session;\n"
            "alter database open resetlogs;"
        )
        verify_started = monotonic()
        elapsed = int(verify_started - started)
        rows = await self._sql(
            "select 'DRILL|'||d.db_unique_name||'|'||d.open_mode||'|'||d.resetlogs_change#||'|'||i.incarnation# "
            "from v$database d cross join v$database_incarnation i where i.status='CURRENT' and "
            "exists (select 1 from v$datafile_header) and not exists "
            "(select 1 from v$datafile_header where error is not null or recover='YES');"
        )
        if len(rows) != 1:
            return _known_failure(started)
        _, name, mode, scn, incarnation = rows[0].split("|")
        if name.upper() != self.config.auxiliary_database or mode != "READ WRITE" or not (
            scn.isdecimal() and incarnation.isdecimal() and min(int(scn), int(incarnation)) > 0
        ):
            return _known_failure(started, retrieval)
        if request.backup.kind == "INCREMENTAL" and int(scn) < request.backup.minimum_scn:
            return _known_failure(started, retrieval)
        for label in self.config.verification_objects:
            owner, table = label.upper().split(".")
            checks = await self._sql(
                "select 'DRILL|'||count(*) from dba_tables "
                f"where owner='{owner}' and table_name='{table}';",
            )
            if checks != ["DRILL|1"]:
                return _known_failure(started)
        return DrillVerification(
            True, datetime.now(UTC), f"{scn}:{incarnation}", None, recovery_coordinate_kind="SCN",
            retrieval_seconds=retrieval, restore_seconds=elapsed,
            verification_seconds=int(monotonic() - verify_started),
        )


class _PostgreSQL(_Engine):
    def _base_args(self):
        return (
            "--no-config", f"--stanza={self.settings.storage_key}", "--repo=1",
            f"--repo1-path={self.site.backup_root}", "--log-level-file=off",
            "--log-level-console=off", "--log-level-stderr=off",
        )

    async def _info(self):
        code, output = await self.run(self.config.executable, (*self._base_args(), "--output=json", "info"))
        if code:
            raise DrillRejected("只读 pgBackRest 仓库不可用", code="BACKUP_UNAVAILABLE")
        registered = SimpleNamespace(stanza=self.settings.storage_key, system_identifier=self.config.system_identifier)
        return postgresql_local_backup._repository(output, registered, self.db_version)

    async def inspect(self):
        c = self.config
        _executable(c.executable)
        postgres = _executable(Path(c.postgres_bin) / "postgres")
        code, version = await self.run(postgres, ("--version",))
        if code or not re.search(rf"\(PostgreSQL\) {self.db_version}(?:\.|\s|$)", version):
            raise DrillRejected("演练 PostgreSQL 与备份版本不一致", code="BACKUP_VERSION_MISMATCH")
        code, backrest = await self.run(c.executable, ("--version",))
        release = re.fullmatch(r"pgBackRest (\d+)\.(\d+)\.(\d+)", backrest.strip())
        if code or not release or tuple(map(int, release.groups())) < (2, 55, 0):
            raise DrillRejected("演练需要兼容的 pgBackRest")
        database, _ = await self._info()
        return version.strip(), backrest.strip(), database

    async def media(self, artifact, run_id, read_only_mount):
        repo, stanza = self.site.backup_root, self.settings.storage_key
        if not valid_backup_chain("PGBACKREST", artifact):
            raise DrillRejected("没有匹配的 pgBackRest 备份链", code="BACKUP_UNAVAILABLE")
        database, backups = await self._info()
        references = (*artifact.dependencies, artifact.reference)
        for part in (repo / "archive" / stanza, *(repo / "backup" / stanza / ref for ref in references)):
            if not part.is_dir() or part.resolve(strict=True) != part or not read_only_mount(part):
                raise DrillRejected("仓库文件路径解析不符或不是只读", code="BACKUP_UNAVAILABLE")
        if not postgresql_local_backup.verified_chain(backups, database, references):
            raise DrillRejected("pgBackRest 仓库缺少本次演练所需的备份链", code="BACKUP_UNAVAILABLE")
        return PostgreSQLMedia(
            str(self.site.root / "run" / "pgbackrest.conf"), str(repo), str(self.site.root / "tablespaces"),
        )

    async def _verify(self, socket, request):
        connection = None
        try:
            async with asyncio.timeout(40):
                while True:
                    try:
                        connection = await asyncpg.connect(
                            host=str(socket.parent), port=5432, database=self.config.verification_database,
                            user=self.credential.username, password=self.credential.password,
                            ssl=False, timeout=3, command_timeout=5, server_settings={"search_path": "pg_catalog"},
                        )
                        started = monotonic()
                        result = await connection.fetchrow(
                            "select current_setting('data_directory') as directory, "
                            "pg_is_in_recovery() and pg_get_wal_replay_pause_state()='paused' as paused, "
                            "(select timeline_id from pg_control_checkpoint()) as timeline, "
                            "pg_last_wal_replay_lsn()::text as lsn, pg_last_xact_replay_timestamp() as replay_time, "
                            "(select count(*) from pg_catalog.pg_class) as objects",
                        )
                        if result["paused"]:
                            for label in self.config.verification_objects:
                                schema, table = label.split(".")
                                table_kind = await connection.fetchval(
                                    "select c.relkind::text from pg_catalog.pg_class c "
                                    "join pg_catalog.pg_namespace n on n.oid=c.relnamespace "
                                    "where n.nspname=$1 and c.relname=$2", schema, table,
                                )
                                if table_kind not in {"r", "p"}:
                                    return None
                                await connection.fetchval(f'SELECT 1 FROM "{schema}"."{table}" LIMIT 1')
                            break
                    except (asyncpg.CannotConnectNowError, OSError):
                        pass
                    finally:
                        if connection is not None:
                            await connection.close(timeout=1)
                            connection = None
                    await asyncio.sleep(0.25)
        except (asyncpg.PostgresError, OSError, TimeoutError):
            return None
        if (result["directory"] != str(self.site.data) or not result["objects"]
            or not result["timeline"] or not re.fullmatch(r"[A-F0-9]+/[A-F0-9]+", result["lsn"] or "")):
            return None
        recovered = result["replay_time"]
        if recovered and recovered > request.simulated_failure_at:
            recovered = None
        return result, recovered, started, int(monotonic() - started)

    async def restore(self, invocation, media, request):
        started = monotonic()
        config = Path(media.repository_config)
        config.write_text(
            f"[global]\nrepo1-path={self.site.backup_root}\nrepo1-type=posix\n"
            f"log-level-file=off\nlock-path={config.parent}\n[{self.settings.storage_key}]\npg1-path={self.site.data}\n",
            encoding="ascii",
        )
        code, _ = await self.run(self.config.executable, invocation.steps[0].arguments, capture=False)
        if code:
            return _known_failure(started)
        data, working = self.site.data, config.parent
        auto_conf = data / "postgresql.auto.conf"
        if (auto_conf.is_symlink() or (data / "standby.signal").exists()
            or not (data / "recovery.signal").is_file()):
            raise DrillRejected("恢复文件含未经批准的备用模式或配置链接")
        # 原备份配置不参与启动；用本次生成的单机配置恢复至所选备份的一致点。
        auto_conf.write_text("", encoding="ascii")
        hba = working / "pg_hba.conf"
        hba.write_text("local all all scram-sha-256\n", encoding="ascii")
        pg_config = working / "postgresql.conf"
        archive_get = " ".join([self.config.executable, f"--config={config}",
                                f"--stanza={self.settings.storage_key}", "--repo=1", 'archive-get %f "%p"'])
        pg_config.write_text(
            f"listen_addresses=''\nunix_socket_directories='{working}'\nport=5432\n"
            f"hba_file='{hba}'\narchive_mode=off\nhot_standby=on\nmax_logical_replication_workers=0\n"
            f"restore_command='{archive_get}'\nrecovery_target='immediate'\n"
            "recovery_target_timeline='current'\nrecovery_target_action='pause'\n",
            encoding="ascii",
        )
        socket = working / ".s.PGSQL.5432"
        async with _start_local(
            str(Path(self.config.postgres_bin) / "postgres"),
            ("-D", str(data), "-c", f"config_file={pg_config}"), socket,
        ):
            results = await self._verify(socket, request)
            checked_at = datetime.now(UTC)
        if results is None:
            return _known_failure(started)
        result, recovered, verify_at, verification = results
        return DrillVerification(
            True, checked_at, f"{result['timeline']}:{result['lsn']}", recovered,
            recovery_coordinate_kind="TIMELINE_LSN", restore_seconds=int(verify_at - started),
            verification_seconds=verification,
        )


_MYSQL_OPTIONS = {
    "innodb_checksum_algorithm", "innodb_data_file_path", "innodb_log_files_in_group",
    "innodb_log_file_size", "innodb_page_size", "innodb_undo_tablespaces", "innodb_undo_directory",
    "innodb_log_checksum_algorithm", "innodb_data_home_dir", "innodb_log_group_home_dir",
    "server_id", "server_uuid", "master_key_id", "redo_log_version", "innodb_fast_checksum",
    "innodb_log_block_size", "innodb_log_checksums", "innodb_redo_log_encrypt", "innodb_undo_log_encrypt",
    "innodb_buffer_pool_filename", "innodb_redo_log_capacity",
}


def _mysql_backup_options(path):
    # XtraBackup 8 的原生文件会重复打印 innodb_log_checksums（最后一项生效）。
    # 仅取准许选项重写到工作副本，原始参数文件不直接交给工具。
    parser = ConfigParser(interpolation=None, delimiters=("=",), strict=False)
    if not _safe_file(path):
        raise DrillRejected("找不到 XtraBackup 原生参数文件")
    parser.read_string(path.read_text(encoding="utf-8"))
    if parser.sections() != ["mysqld"] or parser.defaults():
        raise DrillRejected("备份配置不匹配独立实例")
    options = dict(parser.items("mysqld"))
    if (set(options) - _MYSQL_OPTIONS
        or any(not re.fullmatch(r"[a-zA-Z0-9_.:+/-]*", value) for value in options.values())
        or any(options.get(key, "") not in {"", ".", "./"} for key in
               ("innodb_data_home_dir", "innodb_log_group_home_dir", "innodb_undo_directory"))
        or not re.fullmatch(r"ibdata1:\d+[MK](?::autoextend)?", options.get("innodb_data_file_path", ""))
        or options.get("innodb_buffer_pool_filename", "ib_buffer_pool") != "ib_buffer_pool"
        or options.get("master_key_id", "0") != "0"
        or any(options.get(key, "OFF").upper() not in {"OFF", "0"} for key in
               ("innodb_redo_log_encrypt", "innodb_undo_log_encrypt"))):
        raise DrillRejected("XtraBackup 参数含未批准的路径或选项")
    return options


class _MySQL(_Engine):
    async def inspect(self):
        c = self.config
        mysql = _executable(c.mysqld_executable)
        _executable(c.executable)
        code, version = await self.run(mysql, ("--no-defaults", "--version"))
        if code or not re.search(rf"\bVer\s+{re.escape(self.db_version)}\.\d+", version) or "MariaDB" in version:
            raise DrillRejected("演练 MySQL 与备份版本不匹配", code="BACKUP_VERSION_MISMATCH")
        code, xtra = await self.run(c.executable, ("--no-defaults", "--version"))
        family = mysql_local_backup._TOOL_FAMILY[self.db_version]
        if code or not re.search(r"xtrabackup version " + re.escape(family) + r"\.\d+", xtra):
            raise DrillRejected("演练 XtraBackup 与备份版本不匹配", code="BACKUP_VERSION_MISMATCH")
        return version.strip(), xtra.strip()

    async def media(self, artifact, run_id, read_only_mount):
        if not valid_backup_chain("XTRABACKUP", artifact):
            raise DrillRejected("XtraBackup 备份链无效", code="BACKUP_UNAVAILABLE")
        incremental = artifact.kind == "INCREMENTAL"
        base = artifact.dependencies[0] if incremental else artifact.reference
        directory = _media_directory(self.site.backup_root, base, read_only_mount)
        root = self.site.root / "stage" / f"aiops-{run_id}"
        delta = _media_directory(self.site.backup_root, artifact.reference, read_only_mount) if incremental else None
        delta_stage = root.with_name(f"{root.name}-incremental") if delta else None
        copied_bytes = 0
        for source in (directory, delta) if delta else (directory,):
            for parent, directories, files in os.walk(source, followlinks=False):
                for name in (*directories, *files):
                    child = Path(parent) / name
                    if (child.resolve(strict=True) != child or not read_only_mount(child)
                        or (not child.is_dir() and not child.is_file())):
                        raise DrillRejected("备份含外部链接、非普通文件或可写挂载", code="BACKUP_UNAVAILABLE")
                    if child.is_file():
                        copied_bytes += child.stat().st_size
        _check_stage_space(root, self.site.data, copied_bytes, self.config.required_free_bytes)
        if not all(_safe_file(directory / name) for name in ("xtrabackup_checkpoints", "xtrabackup_logfile", "ibdata1")):
            raise DrillRejected("XtraBackup 全备介质不完整", code="BACKUP_UNAVAILABLE")
        metadata = mysql_local_backup._read_metadata(directory / "xtrabackup_checkpoints")
        if metadata.get("backup_type") != "full-backuped" or metadata.get("from_lsn") != "0":
            raise DrillRejected("只接受原生完整未 prepare 的全备", code="BACKUP_UNAVAILABLE")
        _mysql_backup_options(directory / "backup-my.cnf")
        if delta:
            _check_stage_space(delta_stage, self.site.data, copied_bytes, self.config.required_free_bytes)
            _, base_end = mysql_local_backup.checkpoints(directory, incremental=False)
            start, end = mysql_local_backup.checkpoints(delta, incremental=True)
            if (base_end != start or start != artifact.start_lsn or end != artifact.end_lsn
                or await asyncio.to_thread(mysql_local_backup.incremental_fingerprint, delta) != artifact.fingerprint):
                raise DrillRejected("XtraBackup 备份链的 LSN 或文件摘要不一致", code="BACKUP_UNAVAILABLE")
            _mysql_backup_options(delta / "backup-my.cnf")
        return MySQLMedia(str(directory), str(root), str(delta) if delta else None,
                          str(delta_stage) if delta_stage else None)

    async def _verify(self, socket, coordinate):
        try:
            connection = await aiomysql.connect(
                unix_socket=str(socket), user=self.credential.username, password=self.credential.password,
                db=self.config.verification_database, connect_timeout=5, autocommit=True,
            )
            try:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT @@datadir, @@version, @@server_id, @@global.gtid_executed, @@skip_networking",
                    )
                    data, version, server_id, gtid, no_network = await cursor.fetchone()
                    await cursor.execute(
                        "SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE()",
                    )
                    count, = await cursor.fetchone()
                    for label in self.config.verification_objects:
                        schema, table = label.split(".")
                        await cursor.execute(
                            "SELECT TABLE_TYPE FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
                            (schema, table),
                        )
                        if await cursor.fetchone() != ("BASE TABLE",):
                            return None
                        await cursor.execute(f"SELECT 1 FROM `{schema}`.`{table}` LIMIT 1")
            finally:
                connection.close()
        except (OSError, aiomysql.Error):
            return None
        if (data.rstrip("/") != str(self.site.data) or not version.startswith(self.db_version + ".")
            or server_id != self.config.server_id or not no_network or count <= 0 or not coordinate):
            return None
        kind, value, expected_gtid = coordinate
        if expected_gtid and re.sub(r"\s+", "", gtid or "").casefold() != expected_gtid.casefold():
            return None
        return kind, value

    async def restore(self, invocation, media, request):
        retrieval_started = monotonic()
        source, stage = Path(media.backup_directory), Path(media.staging_directory)
        # 不复制原实例 UUID 与持久化启动参数；原仓库始终只读。
        await asyncio.to_thread(
            shutil.copytree, source, stage, copy_function=shutil.copyfile,
            ignore=shutil.ignore_patterns("auto.cnf", "mysqld-auto.cnf"),
        )
        if media.incremental_directory:
            await asyncio.to_thread(
                shutil.copytree, media.incremental_directory, media.incremental_staging_directory,
                copy_function=shutil.copyfile, ignore=shutil.ignore_patterns("auto.cnf", "mysqld-auto.cnf"),
            )
        retrieval = int(monotonic() - retrieval_started)
        started = monotonic()
        options = _mysql_backup_options(stage / "backup-my.cnf")
        (stage / "backup-my.cnf").write_text(
            "[mysqld]\n" + "".join(f"{key}={value}\n" for key, value in options.items()), encoding="ascii",
        )
        coordinate = None
        info = (Path(media.incremental_directory) if media.incremental_directory else source) / "xtrabackup_binlog_info"
        if _safe_file(info):
            values = info.read_text(encoding="ascii").strip().split(maxsplit=2)
            if len(values) >= 2 and re.fullmatch(r"[a-zA-Z0-9_.-]+", values[0]) and values[1].isdecimal():
                gtid = re.sub(r"\s+", "", values[2]) if len(values) == 3 else None
                if not gtid or re.fullmatch(r"[a-fA-F0-9:,-]{1,256}", gtid):
                    coordinate = ("GTID", gtid, gtid) if gtid else ("BINLOG", f"{values[0]}:{values[1]}", None)
        for step in invocation.steps:
            code, _ = await self.run(self.config.executable, step.arguments, capture=False)
            if code:
                return _known_failure(started, retrieval)
        if (self.site.data / "auto.cnf").exists() or (self.site.data / "mysqld-auto.cnf").exists():
            raise DrillRejected("演练目录包含原实例启动状态，不启动")
        run = self.site.root / "run"
        socket = run / "mysqld.sock"
        modern = self.db_version != "5.7"
        args = (
            "--no-defaults", f"--datadir={self.site.data}", f"--socket={socket}", f"--pid-file={run / 'mysqld.pid'}",
            "--skip-networking", "--event-scheduler=OFF", f"--server-id={self.config.server_id}", "--skip-log-bin",
            "--skip-replica-start" if self.db_version == "8.4" else "--skip-slave-start",
            *(("--mysqlx=OFF", "--persisted_globals_load=OFF") if modern else ()),
            *(("--gtid-mode=ON", "--enforce-gtid-consistency=ON") if coordinate and coordinate[0] == "GTID" else ()),
            *(f"--{key.replace('_', '-')}={options[key]}" for key in (
                "innodb_page_size", "innodb_data_file_path", "innodb_undo_tablespaces",
            ) if key in options and not (modern and key == "innodb_undo_tablespaces")),
        )
        verify_started = monotonic()
        async with _start_local(self.config.mysqld_executable, args, socket):
            verified = await self._verify(socket, coordinate)
            checked_at = datetime.now(UTC)
            verification = int(monotonic() - verify_started)
        if verified is None:
            return _known_failure(started, retrieval)
        kind, value = verified
        return DrillVerification(
            True, checked_at, value, None, recovery_coordinate_kind=kind,
            retrieval_seconds=retrieval, restore_seconds=int(verify_started - started),
            verification_seconds=verification,
        )


def make_engine(config, *args):
    return {"RMAN": _Oracle, "PGBACKREST": _PostgreSQL, "XTRABACKUP": _MySQL}[config.tool](config, *args)
