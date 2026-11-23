"""PostgreSQL / MySQL 本机原生备份。仅调用已登记的源端工具。"""

import asyncio
from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path
from uuid import UUID

from aiops_agent.adapters import mysql_local_backup as mysql
from aiops_agent.adapters import postgresql_local_backup as pg
from aiops_agent.application.data_protection_backup import (
    BackupInvocation,
    backup_storage_key,
    backup_template,
    render_work_backup,
    validate_backup_parameters,
)
from aiops_agent.application.data_protection_execution import DrillRejected
from aiops_agent.application.data_protection_jobs import ApprovedProtection, BackupResult, ProtectionWork
from aiops_agent.application.managed_credentials import AIOpsManagedCredentialError
from aiops_agent.config import AIOpsPgBackRestConfig
from platform_core.contracts.aiops.data_protection import ProtectionPlanSettings


# 不继承 PGPASSWORD、PGSERVICE、PGBACKREST_*、MYSQL_* 或其他现场默认参数。
TOOL_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
_CREDENTIAL_KIND = "protection_backup"


async def _run_local(executable, args, *, env=None, capture=True):
    process = await asyncio.create_subprocess_exec(
        executable, *args, env={**TOOL_ENV, **(env or {})},
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.STDOUT if capture else asyncio.subprocess.DEVNULL,
    )
    try:
        output, _ = await process.communicate()
    except asyncio.CancelledError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        raise
    return process.returncode, output.decode(errors="replace") if output else ""


def _local_path(raw: str, *, predicate: str, mode: int = os.R_OK) -> Path:
    path = Path(raw)
    if (not path.is_absolute() or not path.name or path.as_posix() != raw
        or path.resolve(strict=True) != path or not getattr(path, predicate)() or not os.access(path, mode)):
        raise ValueError("不是可用的规范本机路径")
    return path


@dataclass(frozen=True)
class _Source:
    revision: int
    db_version: str
    credential_revision: str
    username: str = field(repr=False)
    password: str = field(repr=False)


@dataclass(frozen=True)
class _Prepared:
    approval: ApprovedProtection
    source: _Source
    engine: object
    facts: object


class FileLocalBackupBackend:
    def __init__(self, *, bindings, uow_factory, credentials, run_local=_run_local,
                 pg_probe=pg.probe, mysql_probe=mysql.probe):
        self._bindings = dict(bindings)
        self.backup_resource_ids = frozenset(self._bindings)
        self._uow_factory = uow_factory
        self._credentials = credentials
        self._run = run_local
        self._pg_probe = pg_probe
        self._mysql_probe = mysql_probe

    async def _scoped_source(self, *, scope: dict, resource_id: UUID, config, db_type: str) -> _Source:
        async with self._uow_factory() as uow:
            resource = await uow.business_systems.get_resource_scoped(resource_id=resource_id, **scope)
            if resource is None or resource.resource_type_key != "DATABASE" or resource.lifecycle_status != "ACTIVE":
                raise DrillRejected("备份源数据库资源不再有效")
            profile = await uow.business_systems.get_database_profile(resource_id)
            target = await uow.targets.get_scoped(target_id=config.target_id, domain_id=scope["domain_id"])
            if (not profile or profile.target_id != config.target_id or not target or target.status != "ENABLED"
                or target.db_type != db_type or target.deployment_type != "SINGLE_INSTANCE"
                or target.db_role != "PRIMARY"
                or ("tenant_id" in scope and target.tenant_id != scope["tenant_id"])):
                raise DrillRejected("执行位置与当前登记的源数据库不符")
            preset = backup_template(config.tool, target.version_code)
            if preset is None:
                raise DrillRejected("当前数据库版本没有内置备份预设", code="BACKUP_PRESET_UNSUPPORTED")
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
            username, password = identity.get("username"), identity.get("password")
            if (identity.get("auth_mode") != "DATABASE_PASSWORD" or not isinstance(username, str) or not username
                or not isinstance(password, str) or not password
                or any(character in username or character in password for character in "\r\n\x00")):
                raise DrillRejected("专用备份身份不可用于本机工具")
            # 更换加密凭据时 nonce 随之变更；绑定不包含明文或可离线猜测密码的摘要。
            return _Source(target.row_version, target.version_code, row.nonce.hex(), username, password)

    async def _prepare(self, *, scope, resource_id, settings, operation) -> _Prepared:
        config = self._bindings.get(resource_id)
        if operation != "BACKUP" or config is None or config.tool != settings.tool:
            raise DrillRejected("未登记此资源和工具的本机备份执行位置")
        try:
            validate_backup_parameters(settings)
            data = _local_path(config.data_directory, predicate="is_dir", mode=os.R_OK | os.X_OK)
            root = _local_path(config.backup_root, predicate="is_dir", mode=os.R_OK | os.W_OK | os.X_OK)
            _local_path(config.executable, predicate="is_file", mode=os.X_OK)
            if root.is_relative_to(data) or data.is_relative_to(root):
                raise ValueError("备份与源数据目录必须独立")
            if isinstance(config, AIOpsPgBackRestConfig):
                _local_path(config.config_file, predicate="is_file")
                _local_path(config.socket_directory, predicate="is_dir", mode=os.X_OK)
                if settings.storage_key != config.stanza:
                    raise ValueError("计划 stanza 与登记不符")
                directory = root
                engine = pg.PgBackRestLocal(config, self._run, self._pg_probe)
                db_type = "POSTGRESQL"
            else:
                _local_path(config.checksum_executable, predicate="is_file", mode=os.X_OK)
                _local_path(config.socket, predicate="is_socket", mode=os.W_OK)
                directory = _local_path(
                    settings.backup_directory, predicate="is_dir", mode=os.R_OK | os.W_OK | os.X_OK,
                )
                if not directory.is_relative_to(root):
                    raise ValueError("计划目录不在登记的存储范围内")
                engine = mysql.XtraBackupLocal(config, self._run, self._mysql_probe)
                db_type = "MYSQL"
        except (OSError, ValueError):
            raise DrillRejected("工具、源目录、套接字或备份仓库不在可用的登记范围内") from None

        source = await self._scoped_source(scope=scope, resource_id=resource_id, config=config, db_type=db_type)
        try:
            facts = await engine.inspect(source)
        except OSError:
            raise DrillRejected("本机备份工具或文件不可用") from None
        preset = backup_template(config.tool, source.db_version)
        identity = sha256(
            repr((config.model_dump_json(), directory, source.credential_revision, facts.identity)).encode(),
        ).hexdigest()
        approval = ApprovedProtection(
            config.target_id, source.revision, config.tool, preset.key, preset.revision,
            backup_storage_key(settings), source.db_version, execution_identity=identity,
        )
        return _Prepared(approval, source, engine, facts)

    async def preflight(self, *, scope, actor_id, resource_id, settings, operation) -> ApprovedProtection:
        prepared = await self._prepare(scope=scope, resource_id=resource_id, settings=settings, operation=operation)
        return prepared.approval

    async def backup(self, work: ProtectionWork, invocation: BackupInvocation) -> BackupResult:
        settings = ProtectionPlanSettings.model_validate(work.snapshot["settings"])
        prepared = await self._prepare(
            scope=work.scope, resource_id=work.resource_id, settings=settings, operation=work.operation,
        )
        binding = work.snapshot["approval"]["binding"]
        approval = prepared.approval
        frozen = {**vars(approval), "target_id": str(approval.target_id)}
        expected = render_work_backup(settings, work, db_version=approval.db_version)
        if binding != frozen or invocation != expected:
            raise DrillRejected("来源、身份、仓库或备份模板已变化", code="BINDING_CHANGED")
        return await prepared.engine.backup(work, invocation, prepared.source, prepared.facts, approval.storage_key)

    async def drill_runner(self, work):
        raise DrillRejected("本机原生备份适配没有恢复入口")
