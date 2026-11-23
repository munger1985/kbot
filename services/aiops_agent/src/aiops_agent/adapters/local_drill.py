"""独立演练主机的本机恢复适配。批准范围由部署配置固定。"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import os
from pathlib import Path
import pwd
import re
from uuid import UUID

from aiops_agent.adapters import native_drill
from aiops_agent.adapters.file_local_backup import _local_path, _run_local
from aiops_agent.adapters.oracle_local_backup import _run_local as _run_oracle
from aiops_agent.application.data_protection_backup import backup_storage_key
from aiops_agent.application.data_protection_execution import DrillDestination, DrillPreflight, DrillRejected
from aiops_agent.application.data_protection_jobs import ApprovedProtection
from aiops_agent.application.data_protection_restore import render_restore, restore_preset
from aiops_agent.application.managed_credentials import AIOpsManagedCredentialError
from platform_core.contracts.aiops.data_protection import ProtectionPlanSettings


_DB_TYPES = {"RMAN": "ORACLE", "PGBACKREST": "POSTGRESQL", "XTRABACKUP": "MYSQL"}


def machine_id():
    try:
        value = Path("/etc/machine-id").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return value if re.fullmatch(r"[a-f0-9]{32}", value) else None


def _isolated_user(config, read_machine):
    # 网络隔离由批准的演练主机提供；应用不通过连接生产库来验证它。
    try:
        username = pwd.getpwuid(os.geteuid()).pw_name
    except KeyError:
        username = None
    if read_machine() != config.machine_id or os.geteuid() == 0 or username != config.os_user:
        raise DrillRejected("当前执行位置不是登记的独立演练主机和专用系统用户", code="ISOLATION_UNVERIFIED")


def _private(path):
    info = path.stat()
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise DrillRejected("演练写入目录须由专用用户独占", code="ISOLATION_UNVERIFIED")
    return (info.st_dev, info.st_ino)


def _read_only_mount(path):
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)


def _directory(value, *, writable=False):
    mode = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
    return _local_path(str(value), predicate="is_dir", mode=mode)


@dataclass(frozen=True)
class _Identity:
    revision: str
    username: str = field(repr=False)
    password: str = field(repr=False)


@dataclass(frozen=True)
class _Site:
    root: Path
    data: Path
    backup_root: Path
    identity: tuple


class LocalDrillBackend:
    def __init__(self, *, bindings, uow_factory, credentials, run=_run_local, run_oracle=_run_oracle,
                 read_machine=machine_id, verify_user=_isolated_user, read_only_mount=_read_only_mount,
                 engine_factory=native_drill.make_engine):
        self._bindings = dict(bindings)
        self.drill_resource_ids = frozenset(bindings)
        self._uow_factory = uow_factory
        self._credentials = credentials
        self._run = run
        self._run_oracle = run_oracle
        self._read_machine = read_machine
        self._verify_user = verify_user
        self._read_only_mount = read_only_mount
        self._engine_factory = engine_factory

    def _site(self, config):
        self._verify_user(config, self._read_machine)
        try:
            root = _directory(config.drill_root, writable=True)
            repo = _directory(config.backup_root)
            if root.is_relative_to(repo) or repo.is_relative_to(root) or not self._read_only_mount(repo):
                raise DrillRejected("仓库必须只读挂载且与演练目录独立", code="ISOLATION_UNVERIFIED")
            data = _directory(root / "data", writable=True)
            root_id, data_id = _private(root), _private(data)
            if any(data.iterdir()):
                raise DrillRejected("演练目标非空，禁止恢复", code="NOT_EMPTY")
            for name in native_drill.workspace_directories(config.tool):
                child = _directory(root / name, writable=True)
                _private(child)
                if name != "stage" and any(child.iterdir()):
                    raise DrillRejected("演练运行或表空间目录非空", code="NOT_EMPTY")
            usage = os.statvfs(data)
            if usage.f_bavail * usage.f_frsize < config.required_free_bytes:
                raise DrillRejected("演练空间低于登记的容量要求")
            info = repo.stat()
            return _Site(root, data, repo, (root_id, data_id, (info.st_dev, info.st_ino)))
        except (OSError, ValueError):
            raise DrillRejected("演练目录或只读备份挂载不可用") from None

    async def _source(self, scope, resource_id, config):
        async with self._uow_factory() as uow:
            resource = await uow.business_systems.get_resource_scoped(resource_id=resource_id, **scope)
            profile = await uow.business_systems.get_database_profile(resource_id)
            target = await uow.targets.get_scoped(target_id=config.source_target_id, domain_id=scope["domain_id"])
            if (not resource or resource.resource_type_key != "DATABASE" or resource.lifecycle_status != "ACTIVE"
                or not profile or profile.target_id != config.source_target_id or not target or target.status != "ENABLED"
                or target.db_type != _DB_TYPES[config.tool]
                or ("tenant_id" in scope and target.tenant_id != scope["tenant_id"])):
                raise DrillRejected("源端保护绑定已变化", code="BINDING_CHANGED")
            row = await uow.managed_credentials.find(
                **scope, namespace="aiops", credential_kind="protection_restore", external_key=str(resource_id),
            )
            if not row or row.status != "ACTIVE":
                raise DrillRejected("未登记有效的演练专用身份")
            try:
                values = await self._credentials.read(
                    **scope, uow=uow, credential_id=row.credential_id, credential_kind="protection_restore",
                    external_key=resource_id,
                )
            except AIOpsManagedCredentialError:
                raise DrillRejected("无法读取演练专用身份") from None
            username, password = values.get("username"), values.get("password")
            if (not isinstance(username, str) or not username or not isinstance(password, str) or not password
                or any(ch in username or ch in password for ch in "\0\r\n")):
                raise DrillRejected("演练专用身份格式无效")
            return target.version_code, target.row_version, _Identity(row.nonce.hex(), username, password)

    async def _binding(self, scope, resource_id, settings):
        config = self._bindings.get(resource_id)
        if (not config or config.tool != settings.tool or config.destination_key != settings.drill_destination_key):
            raise DrillRejected("目标引用与获准演练位置不符", code="DESTINATION_CHANGED")
        site = self._site(config)
        version, revision, credential = await self._source(scope, resource_id, config)
        preset = restore_preset(config.tool, version, version)
        storage = backup_storage_key(settings)
        if not preset or preset.key != settings.restore_template_key or not storage:
            raise DrillRejected("没有适用于当前数据库版本的批准恢复模板", code="BINDING_CHANGED")
        if config.tool == "PGBACKREST":
            if settings.backup_directory is not None or not settings.storage_key:
                raise DrillRejected("演练须使用已批准的 pgBackRest 仓库", code="BACKUP_UNAVAILABLE")
        elif settings.backup_directory != str(site.backup_root):
            raise DrillRejected("演练主机备份目录与选定仓库不符", code="BACKUP_UNAVAILABLE")
        engine = self._engine_factory(config, site, credential, version, settings, self._run, self._run_oracle)
        try:
            version_facts = await engine.inspect()
        except (OSError, ValueError, KeyError, TypeError):
            raise DrillRejected("演练实例、原生工具或仓库预检失败") from None
        destination = DrillDestination(
            config.destination_key, f"{config.machine_id}:{site.data}", str(site.data),
        )
        execution_identity = sha256(repr((
            config.model_dump_json(), site.identity, credential.revision, version_facts,
        )).encode()).hexdigest()
        return ApprovedProtection(
            config.source_target_id, revision, config.tool, preset.key, preset.revision, storage, version,
            destination, execution_identity,
        ), engine

    async def preflight(self, *, scope, actor_id, resource_id, settings, operation):
        if operation != "DRILL":
            raise DrillRejected("独立演练执行位置不执行源端备份")
        approval, _ = await self._binding(scope, resource_id, settings)
        return approval

    async def backup(self, work, invocation):
        raise DrillRejected("独立演练执行位置不执行源端备份")

    async def drill_runner(self, work):
        if work.operation != "DRILL":
            raise DrillRejected("工作类型不是恢复演练")
        return _Runner(self, work)


class _Runner:
    def __init__(self, backend, work):
        self._backend, self._work = backend, work
        self._settings = ProtectionPlanSettings.model_validate(work.snapshot["settings"])

    async def _prepare(self, request):
        work, settings = self._work, self._settings
        current, engine = await self._backend._binding(work.scope, work.resource_id, settings)
        frozen = {**asdict(current), "target_id": str(current.target_id)}
        if (frozen != work.snapshot["approval"]["binding"] or request.destination != current.destination
            or request.backup != work.backup or request.simulated_failure_at != work.scheduled_at):
            raise DrillRejected("演练目标、凭据、版本或产物与冻结计划不符", code="BINDING_CHANGED")
        try:
            media = await engine.media(request.backup, work.run_id, self._backend._read_only_mount)
        except (OSError, ValueError, KeyError):
            raise DrillRejected("不能只读核验本次演练用到的备份", code="BACKUP_UNAVAILABLE") from None
        invocation = render_restore(
            settings, request, db_version=current.db_version, target_version=engine.db_version,
            run_id=work.run_id, media=media,
        )
        return invocation, engine, media

    async def probe_drill(self, request):
        await self._prepare(request)
        return DrillPreflight(
            request.destination, dedicated_drill_environment=True, no_production_overlap=True,
            isolated_from_production=True, empty_and_idle=True, permissions_verified=True,
            tool_compatible=True, capacity_verified=True, backup_readable=True,
        )

    async def restore_and_verify(self, request):
        # 写入前最后一次读取真实环境；不能复用前一次预检后可能已变化的目录、实例或身份。
        invocation, engine, media = await self._prepare(request)
        return await engine.restore(invocation, media, request)
