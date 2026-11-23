"""数据保护配置：单独保存计划与凭据，不修改业务系统。"""

from datetime import UTC
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiops_agent.application.configuration.business_system_service import _scope_args
from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.configuration.schedule import next_cron_run
from aiops_agent.application.data_protection_backup import backup_template
from aiops_agent.application.data_protection_execution import DrillRejected
from aiops_agent.application.data_protection_jobs import ProtectionJobs
from aiops_agent.application.errors import state_conflict, validation_failed
from aiops_agent.application.configuration.data_protection_help import TOOLS, configuration_checks, protection_advice
from aiops_agent.entities.data_protection import ProtectionPlanEntity
from platform_core.contracts.aiops.data_protection import (
    ProtectionBackupScheduleChange, ProtectionDrillScheduleChange, ProtectionPlanSave,
    ProtectionPlanSettings, ProtectionPlanView, ProtectionRunSummary,
)


_IDENTITY_KINDS = {"backup": "protection_backup", "restore": "protection_restore"}


async def _identity_present(uow, scope, resource_id, purpose):
    row = await uow.managed_credentials.find(
        **_scope_args(scope), namespace="aiops", credential_kind=_IDENTITY_KINDS[purpose],
        external_key=str(resource_id),
    )
    return row is not None and row.status == "ACTIVE"


async def _view(uow, scope, resource_id, target, plan):
    tool = TOOLS.get(target.db_type)
    if not tool:
        raise validation_failed("不支持此数据库类型的数据保护")
    settings = (ProtectionPlanSettings.model_validate(plan.settings_json)
                if plan else ProtectionPlanSettings(tool=tool))
    has_backup = await _identity_present(uow, scope, resource_id, "backup") if plan else False
    has_restore = await _identity_present(uow, scope, resource_id, "restore") if plan else False
    next_backup = plan.next_backup_at if plan else None
    next_drill = plan.next_drill_at if plan else None
    preset = backup_template(settings.tool, target.version_code)
    history = await uow.protection_plans.history(resource_id) if plan else ()
    return ProtectionPlanView(
        resource_id=resource_id, db_type=target.db_type, db_version=target.version_code,
        db_role=target.db_role, status="SCHEDULED" if next_backup or next_drill else "DRAFT" if plan else "NOT_ENROLLED",
        revision=plan.revision if plan else 0, settings=settings,
        has_backup_identity=has_backup, has_restore_identity=has_restore,
        readiness=configuration_checks(settings, has_backup_identity=has_backup,
                                       has_restore_identity=has_restore),
        advice=protection_advice(settings),
        backup_template_key=preset.key if preset else None,
        backup_template_version=preset.revision if preset else None,
        backup_tool_requirement=preset.tool_requirement if preset else None,
        next_backup_at=next_backup.astimezone(UTC) if next_backup else None,
        next_drill_at=next_drill.astimezone(UTC) if next_drill else None,
        drill_preflight_status=((plan.approval_json or {}).get("DRILL_REQUEST", {}).get("status") if plan else None),
        recent_runs=tuple(ProtectionRunSummary(
            run_id=row.run_id, operation=row.operation, status=row.status,
            backup_kind=row.snapshot_json.get("backup_kind", "FULL") if row.operation == "BACKUP" else None,
            message=(row.result_json or {}).get("message"),
            scheduled_at=row.scheduled_at.astimezone(UTC),
            completed_at=row.completed_at.astimezone(UTC) if row.completed_at else None,
            report_id=row.report_id,
        ) for row in history),
    )


class DataProtectionConfigurationMixin:
    async def get_data_protection(self, *, scope: ConfigurationScope, resource_id: UUID):
        async with self._uow_factory() as uow:
            _, target = await self._get_database_resource_and_target(uow=uow, scope=scope, resource_id=resource_id)
            plan = await uow.protection_plans.get(resource_id=resource_id, **_scope_args(scope))
            return await _view(uow, scope, resource_id, target, plan)

    async def save_data_protection(self, *, scope: ConfigurationScope, resource_id: UUID, request: ProtectionPlanSave):
        settings = request.settings
        if settings.backup_rotation:
            try:
                ZoneInfo(settings.backup_rotation.timezone)
            except (ValueError, ZoneInfoNotFoundError) as exc:
                raise validation_failed("需要有效的 IANA 时区") from exc
        for schedule in (settings.backup_schedule, settings.backup_incremental_schedule, settings.drill_schedule):
            if not schedule:
                continue
            try:
                ZoneInfo(schedule.timezone)
            except (ValueError, ZoneInfoNotFoundError) as exc:
                raise validation_failed("需要有效的 IANA 时区") from exc
            if schedule.cron is not None:
                next_cron_run(expression=schedule.cron, timezone_name=schedule.timezone)

        async with self._uow_factory() as uow:
            # 数据库资源行负责串行化首次保存，配置版本防止覆盖已更新的身份。
            _, target = await self._get_database_resource_and_target(
                uow=uow, scope=scope, resource_id=resource_id, lock=True,
            )
            if TOOLS.get(target.db_type) != settings.tool:
                raise validation_failed("工具与源数据库类型不匹配")
            plan = await uow.protection_plans.get(resource_id=resource_id, lock=True, **_scope_args(scope))
            revision = plan.revision if plan else 0
            if revision != request.expected_revision:
                raise state_conflict("保护配置已变化，请刷新后重新确认")
            for purpose, identity in (("backup", request.backup_identity), ("restore", request.restore_identity)):
                if identity is not None:
                    if purpose == "backup" and identity.auth_mode == "ORACLE_OS" and settings.tool != "RMAN":
                        raise validation_failed("OS SYSBACKUP 只适用于本机 Oracle")
                    await self._managed_credentials.put(
                        uow=uow, **_scope_args(scope), actor_id=scope.actor_id,
                        external_key=resource_id, credential_kind=_IDENTITY_KINDS[purpose],
                        values=identity.model_dump(exclude_none=True),
                    )
            settings_json = settings.model_dump(mode="json")
            if plan is None:
                plan = await uow.protection_plans.add(ProtectionPlanEntity(
                    resource_id=resource_id, **_scope_args(scope), revision=1, settings_json=settings_json,
                ))
            else:
                plan.settings_json = settings_json
                plan.revision += 1
                plan.approval_json = None
                plan.next_backup_at = plan.next_drill_at = None
            result = await _view(uow, scope, resource_id, target, plan)
            await uow.commit()
            return result

    async def change_data_protection_backup_schedule(
        self, *, scope: ConfigurationScope, resource_id: UUID, request: ProtectionBackupScheduleChange,
    ) -> ProtectionPlanView:
        async with self._uow_factory() as uow:
            _, target = await self._get_database_resource_and_target(uow=uow, scope=scope, resource_id=resource_id)
            plan = await uow.protection_plans.get(resource_id=resource_id, **_scope_args(scope))
            if plan is None or plan.revision != request.expected_revision:
                raise state_conflict("保护配置已变化，请刷新后重新确认")
            tool, db_version = TOOLS.get(target.db_type), target.version_code
            if request.enabled and (tool is None or backup_template(tool, db_version) is None):
                raise validation_failed("业务系统登记的数据库版本没有内置备份预设")
            if request.enabled and not await _identity_present(uow, scope, resource_id, "backup"):
                raise validation_failed("请先配置专用备份身份")
        jobs = ProtectionJobs(
            uow_factory=self._uow_factory, backend=getattr(self, "_protection_backend", None),
        )
        try:
            if request.enabled:
                await jobs.confirm(
                    scope=_scope_args(scope), resource_id=resource_id, revision=request.expected_revision,
                    actor_id=scope.actor_id, operation="BACKUP", expected_db_version=db_version,
                )
            else:
                await jobs.pause_backup(
                    scope=_scope_args(scope), resource_id=resource_id, revision=request.expected_revision,
                )
        except DrillRejected as exc:
            if exc.code == "BINDING_CHANGED":
                raise state_conflict("保护配置已变化，请刷新后重新确认") from None
            if exc.code == "NO_EXECUTION_BACKEND":
                raise validation_failed("未配置获准的备份执行位置，计划未启用") from None
            if exc.code == "ORACLE_ARCHIVELOG_REQUIRED":
                raise validation_failed("Oracle 未启用 ARCHIVELOG，不能执行整库在线备份") from None
            if exc.code == "BACKUP_PRESET_UNSUPPORTED":
                raise validation_failed("当前数据库版本没有可用的备份预设或执行适配") from None
            if exc.code == "BACKUP_VERSION_MISMATCH":
                raise validation_failed("实际数据库版本与业务系统登记不符，请核对后重新启用") from None
            raise validation_failed("备份计划、参数或现场预检未通过，计划未启用") from None
        except TimeoutError:
            raise validation_failed("备份预检超时，计划未启用") from None
        return await self.get_data_protection(scope=scope, resource_id=resource_id)

    async def change_data_protection_drill_schedule(
        self, *, scope: ConfigurationScope, resource_id: UUID, request: ProtectionDrillScheduleChange,
    ) -> ProtectionPlanView:
        async with self._uow_factory() as uow:
            _, target = await self._get_database_resource_and_target(uow=uow, scope=scope, resource_id=resource_id)
            plan = await uow.protection_plans.get(resource_id=resource_id, **_scope_args(scope))
            if plan is None or plan.revision != request.expected_revision:
                raise state_conflict("保护配置已变化，请刷新后重新确认")
            if request.enabled and not await _identity_present(uow, scope, resource_id, "restore"):
                raise validation_failed("请先配置演练专用身份")
            db_version = target.version_code
        jobs = ProtectionJobs(uow_factory=self._uow_factory, backend=getattr(self, "_protection_backend", None))
        try:
            if request.enabled and resource_id in getattr(self, "_local_drill_resource_ids", ()):
                await jobs.request_drill(
                    scope=_scope_args(scope), resource_id=resource_id, revision=request.expected_revision,
                    actor_id=scope.actor_id, db_version=db_version,
                )
            elif request.enabled:
                await jobs.confirm(
                    scope=_scope_args(scope), resource_id=resource_id, revision=request.expected_revision,
                    actor_id=scope.actor_id, operation="DRILL", expected_db_version=db_version,
                )
            else:
                await jobs.pause_drill(scope=_scope_args(scope), resource_id=resource_id, revision=request.expected_revision)
        except DrillRejected as exc:
            if exc.code == "BINDING_CHANGED":
                raise state_conflict("保护配置已变化，请刷新后重新确认") from None
            raise validation_failed("演练计划或隔离空环境预检未通过，计划未启用") from None
        except TimeoutError:
            raise validation_failed("演练预检超时，计划未启用") from None
        return await self.get_data_protection(scope=scope, resource_id=resource_id)
