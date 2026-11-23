"""数据保护在既有 Worker 中的持久化触发、认领和执行。"""

from asyncio import timeout
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import PurePosixPath
from re import fullmatch
from typing import Literal, Protocol
from uuid import UUID

from aiops_agent.application.data_protection_backup import (
    BackupInvocation, backup_storage_key, backup_template, render_work_backup,
    validate_backup_parameters,
)
from aiops_agent.application.data_protection_execution import (
    BackupArtifact, DrillDestination, DrillRejected, DrillRequest, DrillRunner,
    execute_isolated_drill, valid_backup_chain,
)
from aiops_agent.application.data_protection_schedule import next_backup_at, required_full_at, scheduled_backup_kind
from aiops_agent.entities.data_protection import ProtectionRunEntity
from platform_core.contracts.aiops.data_protection import ProtectionPlanSettings
from platform_core.identity import uuid7


Operation = Literal["BACKUP", "DRILL"]
_REJECTION = {
    "BACKUP_PARAMETERS_MISSING": "没有可用的备份仓库、目录或内置模板参数",
    "BACKUP_PRESET_UNSUPPORTED": "当前数据库版本没有内置备份预设",
    "BACKUP_VERSION_MISMATCH": "实际数据库版本与登记版本不符",
    "ORACLE_ARCHIVELOG_REQUIRED": "Oracle 未启用归档，拒绝整库在线备份",
    "DESTINATION_CHANGED": "演练端实例身份或规范路径未核验，或与登记记录不符",
    "BACKUP_UNAVAILABLE": "没有可用且符合故障时点的备份证据",
    "INCREMENTAL_BASE_UNAVAILABLE": "本次增量所需的计划全量未成功、未通过校验或备份链不可用；未自动改跑全量",
    "ISOLATION_UNVERIFIED": "未证实演练环境与生产主库和备库隔离",
    "NOT_EMPTY": "目标非空、已占用或空状态未验证",
    "BINDING_CHANGED": "来源、数据库版本、批准模板、仓库、演练实例或计划已变化",
}


@dataclass(frozen=True)
class ApprovedProtection:
    """由受控执行位置核验的结果。不能由页面、模型或自由命令生成。"""

    target_id: UUID
    target_revision: int
    tool: str
    template_key: str
    template_revision: str
    storage_key: str
    db_version: str
    destination: DrillDestination | None = None
    execution_identity: str | None = None


@dataclass(frozen=True)
class BackupResult:
    exit_code: int
    artifact: BackupArtifact | None
    # None 表示未证明连续，不等于可任意时间点恢复。
    continuous_logs_verified: bool | None = None
    message: str | None = None


@dataclass(frozen=True)
class ProtectionWork:
    run_id: UUID
    resource_id: UUID
    scope: dict
    operation: Operation
    scheduled_at: datetime
    deadline_at: datetime
    snapshot: dict
    backup: BackupArtifact | None = None


class ProtectionBackend(Protocol):
    """既有执行位置的固定入口。部署未确认时不装配。"""

    async def preflight(
        self, *, scope: dict, actor_id: str, resource_id: UUID, settings: ProtectionPlanSettings, operation: Operation,
    ) -> ApprovedProtection:
        """在实际执行位置只读核验授权、源/目标、专用凭据、工具和仓库。

        db_version 来自当前登记的 Target，并须与现场版本核对。BACKUP 还须核验
        与该预设匹配的原生工具及最终的可写目录或 stanza/repo1。
        任一项未证实，不返回批准绑定。
        """

    async def backup(self, work: ProtectionWork, invocation: BackupInvocation) -> BackupResult:
        """执行内置模板生成的内容；不接收演练端身份，不执行用户命令。"""

    async def drill_runner(self, work: ProtectionWork) -> DrillRunner:
        """目标专用脚本；不加载源库备份身份，不从生产补拉数据。"""


def _snapshot_binding(binding: ApprovedProtection) -> dict:
    return {**asdict(binding), "target_id": str(binding.target_id)}


def _next(schedule, now):
    if schedule is None:
        return None
    if schedule.once_at is not None:
        return schedule.once_at if schedule.once_at >= now else None
    # 配置服务本身依赖任务用例，避免加载模块时反向初始化该包。
    from aiops_agent.application.configuration.schedule import next_cron_run

    return next_cron_run(expression=schedule.cron, timezone_name=schedule.timezone, after=now)


def _scope(plan):
    return {key: getattr(plan, key) for key in ("tenant_id", "domain_id") if hasattr(plan, key)}


def _destination_lock(destination: DrillDestination) -> str:
    # identity 是现场核验后的唯一实例键，不是用户可填的目的地名称。实例别名共用锁。
    return sha256(f"drill:{destination.instance_identity}".encode()).hexdigest()


def _artifact_data(artifact: BackupArtifact) -> dict:
    return {**asdict(artifact), "dependencies": list(artifact.dependencies),
            "available_at": artifact.available_at.astimezone(UTC).isoformat()}


def _artifact_from_run(run, *, at):
    try:
        data = run.result_json["artifact"]
        artifact = BackupArtifact(
            data["storage_key"], data["reference"], datetime.fromisoformat(data["available_at"]),
            data["checksum_verified"], kind=data.get("kind", "FULL"),
            dependencies=tuple(data.get("dependencies", ())), fingerprint=data.get("fingerprint"),
            baseline_fingerprint=data.get("baseline_fingerprint"),
            minimum_scn=data.get("minimum_scn"),
            end_lsn=data.get("end_lsn"), start_lsn=data.get("start_lsn"),
        )
        if artifact.checksum_verified is True and artifact.available_at <= at and artifact.kind in ("FULL", "INCREMENTAL"):
            return artifact
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return None


async def _incremental_base(repo, plan, settings, binding, scheduled_at):
    full_at = required_full_at(settings, scheduled_at)
    if full_at is None:
        return None, None

    def usable(run):
        if (run is None or run.resource_id != plan.resource_id or run.operation != "BACKUP"
            or run.status != "SUCCEEDED" or not run.completed_at or run.completed_at > scheduled_at
            or not full_at <= run.scheduled_at <= scheduled_at
            or run.snapshot_json["revision"] != plan.revision
            or run.snapshot_json["approval"]["binding"] != binding):
            return None
        artifact = _artifact_from_run(run, at=scheduled_at)
        return artifact if artifact and valid_backup_chain(settings.tool, artifact) else None

    recent = await repo.latest_backup(plan.resource_id, at=scheduled_at)
    artifact = usable(recent)
    if artifact is None:
        return None, None
    if artifact.kind == "FULL":
        return (recent.run_id, artifact) if recent.scheduled_at == full_at else (None, None)
    try:
        base_run = await repo.get_run(UUID(recent.snapshot_json["baseline_run_id"]))
    except (KeyError, ValueError, TypeError):
        return None, None
    base = usable(base_run)
    if (not base or base.kind != "FULL" or base.reference != artifact.dependencies[0]
        or base_run.scheduled_at != full_at):
        return None, None
    return base_run.run_id, artifact if settings.tool == "PGBACKREST" else base


class ProtectionJobs:
    def __init__(self, *, uow_factory, backend: ProtectionBackend | None, clock=lambda: datetime.now(UTC)):
        self._uow_factory = uow_factory
        self._backend = backend
        self._clock = clock

    def _resource_ids(self, operation: Operation):
        backup_ids = getattr(self._backend, "backup_resource_ids", None)
        if operation == "BACKUP":
            return backup_ids
        return getattr(self._backend, "drill_resource_ids", frozenset() if backup_ids is not None else None)

    async def confirm(
        self, *, scope: dict, resource_id: UUID, revision: int, actor_id, operation: Operation,
        expected_db_version: str | None = None, drill_request_id: str | None = None,
    ):
        """应用层启用用例。没有获准适配时拒绝；预检期间不持有事务。"""
        if self._backend is None:
            raise DrillRejected("未配置批准的执行适配", code="NO_EXECUTION_BACKEND")
        if operation not in ("BACKUP", "DRILL"):
            raise DrillRejected("未知操作类型")
        resource_ids = self._resource_ids(operation)
        if resource_ids is not None and resource_id not in resource_ids:
            raise DrillRejected("此资源没有获准的执行位置", code="NO_EXECUTION_BACKEND")
        async with self._uow_factory() as uow:
            plan = await uow.protection_plans.get(resource_id=resource_id, **scope)
            if plan is None or plan.revision != revision:
                raise DrillRejected("保护配置已变化", code="BINDING_CHANGED")
            prior_approval = (plan.approval_json or {}).get(operation)
            if drill_request_id and not self._is_pending(plan, drill_request_id):
                raise DrillRejected("演练预检请求已变化", code="BINDING_CHANGED")
            settings = ProtectionPlanSettings.model_validate(plan.settings_json)
        schedule_configured = settings.backup_scheduled if operation == "BACKUP" else settings.drill_schedule is not None
        storage = backup_storage_key(settings)
        if operation == "BACKUP":
            try:
                validate_backup_parameters(settings)
            except ValueError as exc:
                raise DrillRejected(str(exc), code="BACKUP_PARAMETERS_MISSING") from None
        if not (schedule_configured and storage) or (operation == "DRILL" and not settings.restore_template_key):
            raise DrillRejected("计划、模板和备份仓库必须齐全")
        if operation == "DRILL" and not settings.drill_destination_key:
            raise DrillRejected("没有登记专用演练目的地")
        async with timeout(10):
            binding = await self._backend.preflight(
                scope=scope, actor_id=str(actor_id), resource_id=resource_id, settings=settings, operation=operation,
            )
        destination = binding.destination
        if expected_db_version is not None and binding.db_version != expected_db_version:
            raise DrillRejected("业务系统的数据库登记版本已变化", code="BINDING_CHANGED")
        if operation == "BACKUP":
            preset = backup_template(settings.tool, binding.db_version)
            if preset is None:
                raise DrillRejected("当前数据库版本没有内置备份预设", code="BACKUP_PRESET_UNSUPPORTED")
            template, required_version = preset.key, preset.revision
        else:
            template, required_version = settings.restore_template_key, None
        if (binding.tool != settings.tool or binding.template_key != template or not binding.template_revision
            or binding.storage_key != storage or binding.target_revision < 1 or not binding.db_version
            or (required_version and binding.template_revision != required_version)
            or (operation == "BACKUP" and destination is not None)):
            raise DrillRejected("工具、模板、源或仓库与配置不符")
        if operation == "DRILL" and (
            destination is None or destination.key != settings.drill_destination_key
            or not destination.instance_identity or not destination.data_path.startswith("/")
            or not PurePosixPath(destination.data_path).name or ".." in PurePosixPath(destination.data_path).parts
        ):
            raise DrillRejected("未核验专用空演练实例及规范路径")
        now = self._clock()
        next_at = next_backup_at(settings, now) if operation == "BACKUP" else _next(settings.drill_schedule, now)
        if next_at is None:
            raise DrillRejected("计划没有可用的下次执行时间")
        async with self._uow_factory() as uow:
            plan = await uow.protection_plans.get(resource_id=resource_id, lock=True, **scope)
            if plan is None or plan.revision != revision:
                raise DrillRejected("保护配置已变化，请重新预检", code="BINDING_CHANGED")
            approved = dict(plan.approval_json or {})
            if approved.get(operation) != prior_approval:
                raise DrillRejected("保护计划的确认状态已变化", code="BINDING_CHANGED")
            if drill_request_id and not self._is_pending(plan, drill_request_id):
                raise DrillRejected("演练预检请求已变化", code="BINDING_CHANGED")
            snapshot = _snapshot_binding(binding)
            if (operation == "BACKUP" and plan.next_backup_at is not None and prior_approval
                and prior_approval.get("binding") == snapshot):
                return
            approved[operation] = {
                "binding": snapshot, "actor_id": str(actor_id),
                "confirmed_at": now.isoformat(),
            }
            if drill_request_id:
                approved.pop("DRILL_REQUEST", None)
            plan.approval_json = approved
            if operation == "BACKUP":
                plan.next_backup_at = next_at
            else:
                plan.next_drill_at = next_at
            await uow.commit()

    @staticmethod
    def _is_pending(plan, request_id):
        pending = (plan.approval_json or {}).get("DRILL_REQUEST", {})
        return pending.get("status") == "PENDING" and pending.get("id") == request_id

    async def request_drill(self, *, scope, resource_id, revision, actor_id, db_version):
        """API 只登记请求；实际演练位置由独立主机上的 Worker 决定。"""
        async with self._uow_factory() as uow:
            plan = await uow.protection_plans.get(resource_id=resource_id, lock=True, **scope)
            if not plan or plan.revision != revision:
                raise DrillRejected("保护配置已变化", code="BINDING_CHANGED")
            settings = ProtectionPlanSettings.model_validate(plan.settings_json)
            if not (settings.drill_schedule and settings.drill_destination_key and settings.restore_template_key
                    and backup_storage_key(settings) and db_version):
                raise DrillRejected("演练计划、目标、版本、仓库或模板未配置")
            if plan.next_drill_at is not None:
                return
            approved = dict(plan.approval_json or {})
            if approved.get("DRILL_REQUEST", {}).get("status") == "PENDING":
                return
            approved["DRILL_REQUEST"] = {
                "id": str(uuid7()), "status": "PENDING", "actor_id": str(actor_id), "db_version": db_version,
            }
            plan.approval_json = approved
            await uow.commit()

    async def confirm_pending_drill(self):
        ids = getattr(self._backend, "drill_resource_ids", ())
        if not ids:
            return
        async with self._uow_factory() as uow:
            plan = await uow.protection_plans.pending_drill(frozenset(ids))
            if plan is None:
                return
            request = dict(plan.approval_json["DRILL_REQUEST"])
            scope, resource_id, revision = _scope(plan), plan.resource_id, plan.revision
        try:
            await self.confirm(
                scope=scope, resource_id=resource_id, revision=revision, actor_id=request["actor_id"],
                operation="DRILL", expected_db_version=request["db_version"], drill_request_id=request["id"],
            )
        except (DrillRejected, TimeoutError):
            async with self._uow_factory() as uow:
                plan = await uow.protection_plans.get(resource_id=resource_id, lock=True, **scope)
                if plan and plan.revision == revision and self._is_pending(plan, request["id"]):
                    plan.approval_json = {
                        **plan.approval_json, "DRILL_REQUEST": {"id": request["id"], "status": "REJECTED"},
                    }
                    await uow.commit()

    async def pause_backup(self, *, scope: dict, resource_id: UUID, revision: int) -> None:
        await self._pause(scope=scope, resource_id=resource_id, revision=revision, operation="BACKUP")

    async def pause_drill(self, *, scope: dict, resource_id: UUID, revision: int) -> None:
        await self._pause(scope=scope, resource_id=resource_id, revision=revision, operation="DRILL")

    async def _pause(self, *, scope: dict, resource_id: UUID, revision: int, operation: Operation) -> None:
        async with self._uow_factory() as uow:
            plan = await uow.protection_plans.get(resource_id=resource_id, lock=True, **scope)
            if plan is None or plan.revision != revision:
                raise DrillRejected("保护配置已变化", code="BINDING_CHANGED")
            approved = dict(plan.approval_json or {})
            approved.pop(operation, None)
            if operation == "DRILL":
                approved.pop("DRILL_REQUEST", None)
            plan.approval_json = approved or None
            setattr(plan, "next_backup_at" if operation == "BACKUP" else "next_drill_at", None)
            await uow.commit()

    async def claim(self, operation: Operation) -> ProtectionWork | None:
        if self._backend is None:
            return None
        resource_ids = self._resource_ids(operation)
        if resource_ids is not None and not resource_ids:
            return None
        now = self._clock()
        async with self._uow_factory() as uow:
            repo = uow.protection_plans
            filters = {} if resource_ids is None else {"resource_ids": resource_ids}
            plan = await repo.due_plan(operation=operation, now=now, **filters)
            if not plan:
                return None
            settings = ProtectionPlanSettings.model_validate(plan.settings_json)
            approval = (plan.approval_json or {}).get(operation)
            due_field = "next_backup_at" if operation == "BACKUP" else "next_drill_at"
            scheduled_at = getattr(plan, due_field).astimezone(UTC)
            schedule = settings.drill_schedule
            schedule_configured = settings.backup_scheduled if operation == "BACKUP" else schedule is not None
            next_at = None
            if approval and schedule_configured:
                if operation == "BACKUP":
                    next_at = next_backup_at(settings, now)
                elif schedule.cron:
                    next_at = _next(schedule, now)
            setattr(plan, due_field, next_at)
            if not (approval and schedule_configured):
                await uow.commit()
                return None

            scope = _scope(plan)
            binding = approval["binding"]
            snapshot = {
                "settings": plan.settings_json, "revision": int(plan.revision), "approval": approval,
                "queued_at": scheduled_at.isoformat(), "claimed_at": now.isoformat(),
            }
            if operation == "BACKUP":
                snapshot["backup_kind"] = scheduled_backup_kind(settings, scheduled_at)
            run_id = uuid7()
            run = ProtectionRunEntity(
                run_id=run_id, resource_id=plan.resource_id, operation=operation, scheduled_at=scheduled_at,
                status="RUNNING", snapshot_json=snapshot, deadline_at=now + timedelta(seconds=settings.timeout_seconds),
            )
            work = None
            if now - scheduled_at > timedelta(seconds=settings.schedule_window_seconds):
                result = ("SKIPPED", "已错过计划执行窗口")
            elif operation == "BACKUP":
                run.active_key = sha256(f"backup:{plan.resource_id}".encode()).hexdigest()
                result = None
                backup = None
                if snapshot["backup_kind"] is None:
                    result = ("REJECTED", "认领时点与保存的备份计划不匹配")
                elif snapshot["backup_kind"] == "INCREMENTAL":
                    base_run_id, backup = await _incremental_base(repo, plan, settings, binding, scheduled_at)
                    if backup is None:
                        result = ("REJECTED", _REJECTION["INCREMENTAL_BASE_UNAVAILABLE"])
                    else:
                        snapshot["baseline_run_id"] = str(base_run_id)
                        snapshot["baseline"] = _artifact_data(backup)
            else:
                destination = DrillDestination(**binding["destination"])
                run.active_key = _destination_lock(destination)
                candidate = await repo.latest_backup(plan.resource_id, at=scheduled_at)
                backup = None
                if candidate and candidate.result_json and candidate.result_json.get("artifact"):
                    original = candidate.snapshot_json["approval"]["binding"]
                    if (binding.get("db_version")
                        and all(original.get(key) == binding.get(key) for key in
                                ("target_id", "target_revision", "db_version", "tool", "storage_key"))
                        and candidate.snapshot_json["settings"].get("backup_directory") == settings.backup_directory):
                        artifact = candidate.result_json["artifact"]
                        backup = _artifact_from_run(candidate, at=scheduled_at)
                        if backup and backup.kind == "INCREMENTAL" and not valid_backup_chain(settings.tool, backup):
                            backup = None
                        if backup:
                            snapshot["backup"] = {"run_id": str(candidate.run_id), **artifact,
                                                  "continuous_logs_verified": candidate.result_json.get("continuous_logs_verified")}
                result = None if backup else ("REJECTED", "没有 T0 时已存在且与当前来源一致的有效备份")
            if result is None:
                if await repo.reserve(run):
                    work = ProtectionWork(run_id, plan.resource_id, scope, operation, scheduled_at,
                                          run.deadline_at, snapshot, backup)
                else:
                    result = ("SKIPPED", "已有运行占用资源，或上次执行状态未确认")
            if result is not None:
                status, message = result
                await repo.add_run(ProtectionRunEntity(
                    run_id=run_id, resource_id=plan.resource_id, operation=operation, scheduled_at=scheduled_at,
                    deadline_at=run.deadline_at, completed_at=now, status=status,
                    snapshot_json=snapshot, result_json={"message": message},
                ))
            await uow.commit()
            return work

    async def _assert_current(self, work: ProtectionWork) -> None:
        async with self._uow_factory() as uow:
            plan = await uow.protection_plans.get(resource_id=work.resource_id, **work.scope)
            run = await uow.protection_plans.get_run(work.run_id)
            if (not plan or plan.revision != work.snapshot["revision"]
                or (plan.approval_json or {}).get(work.operation) != work.snapshot["approval"]
                or not run or run.status != "RUNNING" or self._clock() >= work.deadline_at):
                raise DrillRejected("保护配置、授权或认领状态已变化", code="BINDING_CHANGED")

    async def execute(self, work: ProtectionWork) -> None:
        backend = self._backend
        if backend is None:
            return
        settings = ProtectionPlanSettings.model_validate(work.snapshot["settings"])
        binding = work.snapshot["approval"]["binding"]
        write_requested = False

        async def before_write():
            nonlocal write_requested
            await self._assert_current(work)
            write_requested = True

        try:
            remaining = (work.deadline_at - self._clock()).total_seconds()
            if remaining <= 0:
                raise TimeoutError
            async with timeout(remaining):
                await self._assert_current(work)
                current = await backend.preflight(
                    scope=work.scope, actor_id=work.snapshot["approval"]["actor_id"], resource_id=work.resource_id,
                    settings=settings, operation=work.operation,
                )
                if _snapshot_binding(current) != binding:
                    raise DrillRejected("来源、批准脚本、仓库或演练实例已变化", code="BINDING_CHANGED")
                if work.operation == "BACKUP":
                    preset = backup_template(settings.tool, current.db_version)
                    if preset is None or (binding["template_key"], binding["template_revision"]) != (preset.key, preset.revision):
                        raise DrillRejected("内置模板已变化，需重新确认计划", code="BINDING_CHANGED")
                    try:
                        invocation = render_work_backup(settings, work, db_version=current.db_version)
                    except ValueError:
                        raise DrillRejected("备份参数不完整", code="BACKUP_PARAMETERS_MISSING") from None
                    await before_write()
                    result = await backend.backup(work, invocation)
                    artifact = result.artifact
                    expected_deps = ()
                    if invocation.kind == "INCREMENTAL":
                        expected_deps = (*work.backup.dependencies, work.backup.reference)
                    if (result.exit_code != 0 or artifact is None or artifact.checksum_verified is not True
                        or artifact.kind != invocation.kind or artifact.dependencies != expected_deps
                        or (settings.incremental_enabled and not valid_backup_chain(settings.tool, artifact))
                        or not fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", artifact.reference)
                        or artifact.storage_key != backup_storage_key(settings)
                        or artifact.available_at.utcoffset() is None or artifact.available_at > self._clock()):
                        await self._finish(work.run_id, "FAILED", {
                            "exit_code": result.exit_code,
                            "message": result.message or "工具失败或未返回已校验、可读取的备份产物",
                        })
                        return
                    payload = {"exit_code": 0, "artifact": _artifact_data(artifact),
                               "continuous_logs_verified": result.continuous_logs_verified}
                    succeeded = True
                else:
                    request = DrillRequest(settings.tool, DrillDestination(**binding["destination"]),
                                           binding["storage_key"], work.backup, work.scheduled_at)
                    runner = await backend.drill_runner(work)
                    verification = await execute_isolated_drill(request, runner, before_write=before_write)
                    payload = verification.evidence(settings.tool, not_before=work.scheduled_at, not_after=self._clock())
                    succeeded = payload["database_verified"]
        except DrillRejected as exc:
            if write_requested:
                await self._finish(work.run_id, "UNKNOWN", {"message": "写入状态未确认，保留现场，禁止自动重试"})
            else:
                message = _REJECTION.get(exc.code, "配置、权限、仓库或空库安全检查未通过")
                await self._finish(work.run_id, "REJECTED", {"message": message})
        except Exception:
            # 不能从异常/超时判断外部进程已退出。不要存储可能含认证信息的异常全文。
            await self._finish(work.run_id, "UNKNOWN", {"message": "外部执行状态未确认，保留现场，禁止自动重试"})
        else:
            await self._finish(work.run_id, "SUCCEEDED" if succeeded else "FAILED", payload)

    async def _finish(self, run_id: UUID, status: str, result: dict) -> None:
        async with self._uow_factory() as uow:
            run = await uow.protection_plans.get_run(run_id, lock=True)
            if run is None or run.status != "RUNNING":
                return
            run.completed_at, run.status, run.result_json = self._clock(), status, result
            if status != "UNKNOWN":
                run.active_key = None
            await uow.commit()

    async def reconcile(self) -> None:
        async with self._uow_factory() as uow:
            now = self._clock()
            run = await uow.protection_plans.expiring_run(now=now)
            if run is not None:
                run.completed_at, run.status = now, "UNKNOWN"
                run.result_json = {"message": "认领已过期，外部进程状态未确认，禁止自动重试"}
                await uow.commit()

