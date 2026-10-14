"""数据库资源级保护目标与人工恢复演练配置用例。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from aiops_agent.application.configuration.common import ConfigurationScope, add_configuration_event
from aiops_agent.application.configuration.business_system_service import _scope_args
from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed
from aiops_agent.application.work_items import cancel_recovery_drill_work_item, ensure_recovery_drill_work_item, review_recovery_drill_work_items, submit_recovery_drill_work_item
from aiops_agent.entities import RecoveryDrillEntity, RecoveryProfileEntity
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.contracts.aiops import (
    ResourceProtectionProfileUpsert,
    ResourceProtectionProfileView,
    RecoveryDrillCreate,
    RecoveryDrillPage,
    RecoveryDrillReview,
    RecoveryDrillView,
)
from platform_core.identity import uuid7


_SOURCE_DB_TYPES = {
    "ORACLE_RMAN": "ORACLE",
    "POSTGRESQL_BASEBACKUP": "POSTGRESQL",
    "POSTGRESQL_PGBACKREST": "POSTGRESQL",
    "POSTGRESQL_BARMAN": "POSTGRESQL",
    "POSTGRESQL_WALG": "POSTGRESQL",
    "MYSQL_XTRABACKUP": "MYSQL",
    "MYSQL_ENTERPRISE_BACKUP": "MYSQL",
    "MYSQL_LOGICAL_DUMP": "MYSQL",
}
_ASSURANCE_ORDER = {"BACKUP_METADATA": 1, "RESTORE_VALIDATE": 2, "DATABASE_OPEN": 3, "APPLICATION_VALIDATED": 4}


def _drill_satisfies_profile(*, drill: RecoveryDrillEntity, profile: RecoveryProfileEntity | None) -> bool:
    return bool(
        profile is not None
        and drill.result == "PASS"
        and _ASSURANCE_ORDER[drill.assurance_level] >= _ASSURANCE_ORDER[profile.required_assurance_level]
        and drill.backup_source_type in set(profile.required_backup_source_types_json)
        and drill.achieved_rpo_seconds is not None
        and drill.achieved_rto_seconds is not None
        and int(drill.achieved_rpo_seconds) <= int(profile.rpo_seconds)
        and int(drill.achieved_rto_seconds) <= int(profile.rto_seconds)
    )


def recovery_profile_view(entity: RecoveryProfileEntity) -> ResourceProtectionProfileView:
    return ResourceProtectionProfileView(
        recovery_profile_id=entity.recovery_profile_id,
        resource_id=entity.resource_id,
        version_no=int(entity.version_no),
        backup_policy_enabled=bool(entity.backup_policy_enabled),
        backup_frequency_hours=int(entity.backup_frequency_hours),
        backup_success_max_age_hours=int(entity.backup_success_max_age_hours),
        backup_retention_days=int(entity.backup_retention_days),
        rpo_seconds=int(entity.rpo_seconds),
        rto_seconds=int(entity.rto_seconds),
        drill_required=bool(entity.drill_required),
        required_drill_interval_days=int(entity.required_drill_interval_days),
        required_assurance_level=entity.required_assurance_level,
        required_backup_source_types=tuple(entity.required_backup_source_types_json),
        rto_clock_basis=entity.rto_clock_basis,
        source_note=entity.source_note,
        status=entity.status,
        effective_at=entity.effective_at,
        retired_at=entity.retired_at,
        confirmed_by=entity.confirmed_by,
        row_version=int(entity.row_version),
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def recovery_drill_view(entity: RecoveryDrillEntity) -> RecoveryDrillView:
    return RecoveryDrillView(
        drill_id=entity.drill_id,
        resource_id=entity.resource_id,
        recovery_profile_id=entity.recovery_profile_id,
        recovery_profile_version=entity.recovery_profile_version,
        db_type=entity.db_type,
        scenario=entity.scenario,
        assurance_level=entity.assurance_level,
        backup_source_type=entity.backup_source_type,
        environment=entity.environment,
        result=entity.result,
        status=entity.status,
        simulated_failure_at=entity.simulated_failure_at,
        recovered_through_at=entity.recovered_through_at,
        service_validated_at=entity.service_validated_at,
        achieved_rpo_seconds=entity.achieved_rpo_seconds,
        achieved_rto_seconds=entity.achieved_rto_seconds,
        recovery_marker=entity.recovery_marker_json,
        evidence=tuple(entity.evidence_json),
        source_trust_level=entity.source_trust_level,
        reviewed_by=entity.reviewed_by,
        reviewed_at=entity.reviewed_at,
        review_note=entity.review_note,
        notes=entity.notes,
        row_version=int(entity.row_version),
        created_at=entity.created_at,
        created_by=entity.created_by,
        updated_at=entity.updated_at,
        updated_by=entity.updated_by,
    )


class RecoveryConfigurationMixin:
    async def _get_database_resource_and_target(self, *, uow: AIOpsUnitOfWork, scope: ConfigurationScope, resource_id: UUID, lock: bool = False):
        assert uow.business_systems is not None and uow.targets is not None
        resource = await uow.business_systems.get_resource_scoped(resource_id=resource_id, lock=lock, **_scope_args(scope))
        if resource is None or resource.resource_type_key != "DATABASE":
            raise resource_not_found("Database Resource")
        database = await uow.business_systems.get_database_profile(resource_id)
        if database is None:
            raise resource_not_found("Database Resource Profile")
        target = await uow.targets.get_scoped(target_id=database.target_id, domain_id=scope.domain_id, lock=lock)
        if target is None:
            raise resource_not_found("Target")
        return resource, target

    async def get_recovery_profile(self, *, scope: ConfigurationScope, resource_id: UUID) -> ResourceProtectionProfileView | None:
        async with self._uow_factory() as uow:
            assert uow.recovery is not None
            await self._get_database_resource_and_target(uow=uow, scope=scope, resource_id=resource_id)
            entity = await uow.recovery.get_active_profile(resource_id=resource_id, **_scope_args(scope))
            return recovery_profile_view(entity) if entity else None

    async def upsert_recovery_profile(self, *, scope: ConfigurationScope, resource_id: UUID, request: ResourceProtectionProfileUpsert, idempotency_key: str) -> ResourceProtectionProfileView:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> ResourceProtectionProfileView:
            assert uow.recovery is not None and uow.targets is not None
            _resource, target = await self._get_database_resource_and_target(uow=uow, scope=scope, resource_id=resource_id, lock=True)
            source_db_types = {_SOURCE_DB_TYPES[item] for item in request.required_backup_source_types if item in _SOURCE_DB_TYPES}
            if source_db_types - {str(target.db_type)}:
                raise validation_failed("必需备份来源类型必须与数据库资源引擎一致")
            versions = await uow.recovery.list_profile_versions(resource_id=resource_id, lock=True, **_scope_args(scope))
            for prior in versions:
                if prior.status == "ACTIVE":
                    prior.status = "RETIRED"
                    prior.retired_at = now
                    prior.updated_at = now
                    prior.updated_by = scope.actor_id
            values = dict(
                recovery_profile_id=uuid7(), resource_id=resource_id, domain_id=scope.domain_id,
                version_no=max((int(row.version_no) for row in versions), default=0) + 1,
                backup_policy_enabled=request.backup_policy_enabled, backup_frequency_hours=request.backup_frequency_hours,
                backup_success_max_age_hours=request.backup_success_max_age_hours, backup_retention_days=request.backup_retention_days,
                rpo_seconds=request.rpo_seconds, rto_seconds=request.rto_seconds, drill_required=request.drill_required,
                required_drill_interval_days=request.required_drill_interval_days, required_assurance_level=request.required_assurance_level,
                required_backup_source_types_json=list(request.required_backup_source_types), rto_clock_basis=request.rto_clock_basis,
                source_note=request.source_note, status="ACTIVE", effective_at=now, confirmed_by=scope.actor_id,
                row_version=1, created_by=scope.actor_id, updated_by=scope.actor_id, created_at=now, updated_at=now,
            )
            if hasattr(scope, "tenant_id"):
                values["tenant_id"] = scope.tenant_id
            entity = RecoveryProfileEntity(**values)
            await uow.recovery.add_profile(entity)
            assert uow.work_items is not None
            if request.drill_required:
                await ensure_recovery_drill_work_item(uow=uow, scope=scope, target=target, profile=entity, now=now)
            else:
                await cancel_recovery_drill_work_item(uow=uow, scope=scope, target_id=target.target_id, now=now)
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="RECOVERY_PROFILE", aggregate_id=entity.recovery_profile_id, event_type="SYSTEM_PROTECTION_PROFILE_ACTIVATED", row_version=1, details={"resource_id": str(resource_id), "version_no": entity.version_no})
            return recovery_profile_view(entity)
        return await self._idempotent(scope=scope, operation="SYSTEM_PROTECTION_PROFILE_UPSERT", parent_resource=str(resource_id), idempotency_key=idempotency_key, payload=request.model_dump(mode="json"), response_type=ResourceProtectionProfileView, handler=handler)

    async def list_recovery_drills(self, *, scope: ConfigurationScope, resource_id: UUID) -> RecoveryDrillPage:
        async with self._uow_factory() as uow:
            assert uow.recovery is not None
            await self._get_database_resource_and_target(uow=uow, scope=scope, resource_id=resource_id)
            rows = await uow.recovery.list_recent_drills(resource_id=resource_id, **_scope_args(scope))
            return RecoveryDrillPage(items=tuple(recovery_drill_view(row) for row in rows))

    async def create_recovery_drill(self, *, scope: ConfigurationScope, resource_id: UUID, request: RecoveryDrillCreate, idempotency_key: str) -> RecoveryDrillView:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> RecoveryDrillView:
            assert uow.recovery is not None and uow.targets is not None
            _resource, target = await self._get_database_resource_and_target(uow=uow, scope=scope, resource_id=resource_id, lock=True)
            db_type = str(target.db_type)
            if request.recovery_marker.kind != db_type or _SOURCE_DB_TYPES.get(request.backup_source_type, db_type) != db_type:
                raise validation_failed("恢复坐标或备份来源类型与数据库资源引擎不一致")
            profile = await uow.recovery.get_active_profile(resource_id=resource_id, **_scope_args(scope))
            achieved_rpo = int((request.simulated_failure_at - request.recovered_through_at).total_seconds()) if request.recovered_through_at else None
            achieved_rto = int((request.service_validated_at - request.simulated_failure_at).total_seconds()) if request.service_validated_at else None
            values = dict(
                drill_id=uuid7(), resource_id=resource_id, execution_target_id=target.target_id, domain_id=scope.domain_id,
                recovery_profile_id=profile.recovery_profile_id if profile else None, recovery_profile_version=int(profile.version_no) if profile else None,
                db_type=db_type, scenario=request.scenario, assurance_level=request.assurance_level, backup_source_type=request.backup_source_type,
                environment=request.environment, result=request.result, status="SUBMITTED", simulated_failure_at=request.simulated_failure_at,
                recovered_through_at=request.recovered_through_at, service_validated_at=request.service_validated_at,
                achieved_rpo_seconds=achieved_rpo, achieved_rto_seconds=achieved_rto,
                recovery_marker_json=request.recovery_marker.model_dump(mode="json"), evidence_json=[item.model_dump(mode="json") for item in request.evidence],
                source_trust_level="USER_PROVIDED", notes=request.notes, row_version=1,
                created_by=scope.actor_id, updated_by=scope.actor_id, created_at=now, updated_at=now,
            )
            if hasattr(scope, "tenant_id"):
                values["tenant_id"] = scope.tenant_id
            entity = RecoveryDrillEntity(**values)
            await uow.recovery.add_drill(entity)
            if request.source_work_item_id is not None:
                if profile is None or not bool(profile.drill_required):
                    raise state_conflict("当前数据库资源未要求定期恢复演练，不能从计划工单提交")
                assert uow.work_items is not None
                await submit_recovery_drill_work_item(uow=uow, scope=scope, work_item_id=request.source_work_item_id, target_id=target.target_id, drill=entity, now=now)
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="RECOVERY_DRILL", aggregate_id=entity.drill_id, event_type="RECOVERY_DRILL_SUBMITTED", row_version=1, details={"resource_id": str(resource_id), "execution_target_id": str(target.target_id)})
            return recovery_drill_view(entity)
        return await self._idempotent(scope=scope, operation="RECOVERY_DRILL_CREATE", parent_resource=str(resource_id), idempotency_key=idempotency_key, payload=request.model_dump(mode="json"), response_type=RecoveryDrillView, handler=handler)

    async def review_recovery_drill(self, *, scope: ConfigurationScope, resource_id: UUID, drill_id: UUID, request: RecoveryDrillReview, expected_version: int, idempotency_key: str) -> RecoveryDrillView:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> RecoveryDrillView:
            assert uow.recovery is not None and uow.targets is not None
            await self._get_database_resource_and_target(uow=uow, scope=scope, resource_id=resource_id, lock=True)
            entity = await uow.recovery.get_drill_scoped(drill_id=drill_id, resource_id=resource_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Recovery Drill")
            self._check_version(entity.row_version, expected_version)
            if entity.status != "SUBMITTED":
                raise state_conflict("只有待审核演练记录可以审核")
            target = await uow.targets.get_scoped(target_id=entity.execution_target_id, domain_id=scope.domain_id)
            if target is None:
                raise resource_not_found("Target")
            entity.status = "VERIFIED" if request.decision == "VERIFY" else "REJECTED"
            entity.reviewed_by = scope.actor_id
            entity.reviewed_at = now
            entity.review_note = request.review_note
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.flush()
            profile = await uow.recovery.get_active_profile(resource_id=resource_id, **_scope_args(scope))
            assert uow.work_items is not None
            await review_recovery_drill_work_items(uow=uow, scope=scope, target=target, profile=profile, drill=entity, verified=request.decision == "VERIFY", satisfies_policy=request.decision == "VERIFY" and bool(profile and profile.drill_required) and _drill_satisfies_profile(drill=entity, profile=profile), now=now)
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="RECOVERY_DRILL", aggregate_id=entity.drill_id, event_type=f"RECOVERY_DRILL_{entity.status}", row_version=int(entity.row_version), details={"resource_id": str(resource_id)})
            return recovery_drill_view(entity)
        return await self._idempotent(scope=scope, operation="RECOVERY_DRILL_REVIEW", parent_resource=str(drill_id), idempotency_key=idempotency_key, payload={**request.model_dump(mode="json"), "row_version": expected_version}, response_type=RecoveryDrillView, handler=handler)
