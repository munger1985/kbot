"""Target 恢复目标与人工恢复演练配置用例。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    add_configuration_event,
)
from aiops_agent.application.errors import (
    resource_not_found,
    state_conflict,
    validation_failed,
)
from aiops_agent.entities import RecoveryDrillEntity, RecoveryProfileEntity
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.contracts.aiops import (
    RecoveryDrillCreate,
    RecoveryDrillPage,
    RecoveryDrillReview,
    RecoveryDrillView,
    TargetRecoveryProfileUpsert,
    TargetRecoveryProfileView,
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


def _profile_view(entity: RecoveryProfileEntity) -> TargetRecoveryProfileView:
    return TargetRecoveryProfileView(
        recovery_profile_id=entity.recovery_profile_id,
        target_id=entity.target_id,
        version_no=int(entity.version_no),
        rpo_seconds=int(entity.rpo_seconds),
        rto_seconds=int(entity.rto_seconds),
        required_drill_interval_days=int(entity.required_drill_interval_days),
        required_assurance_level=entity.required_assurance_level,
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
        target_id=entity.target_id,
        recovery_profile_id=entity.recovery_profile_id,
        recovery_profile_version=(
            int(entity.recovery_profile_version)
            if entity.recovery_profile_version is not None
            else None
        ),
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
        achieved_rpo_seconds=(
            int(entity.achieved_rpo_seconds)
            if entity.achieved_rpo_seconds is not None
            else None
        ),
        achieved_rto_seconds=(
            int(entity.achieved_rto_seconds)
            if entity.achieved_rto_seconds is not None
            else None
        ),
        recovery_marker=entity.recovery_marker_json,
        evidence=tuple(entity.evidence_json or ()),
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
    async def get_recovery_profile(
        self, *, scope: ConfigurationScope, target_id: UUID
    ) -> TargetRecoveryProfileView | None:
        async with self._uow_factory() as uow:
            assert uow.targets is not None and uow.recovery is not None
            target = await uow.targets.get_scoped(
                target_id=target_id, domain_id=scope.domain_id
            )
            if target is None:
                raise resource_not_found("Target")
            entity = await uow.recovery.get_active_profile(
                target_id=target_id, domain_id=scope.domain_id
            )
            return _profile_view(entity) if entity is not None else None

    async def upsert_recovery_profile(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        request: TargetRecoveryProfileUpsert,
        idempotency_key: str,
    ) -> TargetRecoveryProfileView:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> TargetRecoveryProfileView:
            assert uow.targets is not None and uow.recovery is not None
            target = await uow.targets.get_scoped(
                target_id=target_id, domain_id=scope.domain_id, lock=True
            )
            if target is None:
                raise resource_not_found("Target")
            versions = await uow.recovery.list_profile_versions(
                target_id=target_id, domain_id=scope.domain_id, lock=True
            )
            for prior in versions:
                if prior.status == "ACTIVE":
                    prior.status = "RETIRED"
                    prior.retired_at = now
                    prior.updated_at = now
                    prior.updated_by = scope.actor_id
            entity = RecoveryProfileEntity(
                recovery_profile_id=uuid7(),
                target_id=target_id,
                domain_id=scope.domain_id,
                version_no=max((int(row.version_no) for row in versions), default=0) + 1,
                rpo_seconds=request.rpo_seconds,
                rto_seconds=request.rto_seconds,
                required_drill_interval_days=request.required_drill_interval_days,
                required_assurance_level=request.required_assurance_level,
                rto_clock_basis=request.rto_clock_basis,
                source_note=request.source_note,
                status="ACTIVE",
                effective_at=now,
                retired_at=None,
                confirmed_by=scope.actor_id,
                row_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await uow.recovery.add_profile(entity)
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="RECOVERY_PROFILE",
                aggregate_id=entity.recovery_profile_id,
                event_type="RECOVERY_PROFILE_ACTIVATED",
                row_version=1,
                details={"target_id": str(target_id), "version_no": entity.version_no},
            )
            return _profile_view(entity)

        return await self._idempotent(
            scope=scope,
            operation="RECOVERY_PROFILE_UPSERT",
            parent_resource=str(target_id),
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=TargetRecoveryProfileView,
            handler=handler,
        )

    async def list_recovery_drills(
        self, *, scope: ConfigurationScope, target_id: UUID
    ) -> RecoveryDrillPage:
        async with self._uow_factory() as uow:
            assert uow.targets is not None and uow.recovery is not None
            target = await uow.targets.get_scoped(
                target_id=target_id, domain_id=scope.domain_id
            )
            if target is None:
                raise resource_not_found("Target")
            rows = await uow.recovery.list_recent_drills(
                target_id=target_id, domain_id=scope.domain_id
            )
            return RecoveryDrillPage(items=tuple(recovery_drill_view(row) for row in rows))

    async def create_recovery_drill(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        request: RecoveryDrillCreate,
        idempotency_key: str,
    ) -> RecoveryDrillView:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> RecoveryDrillView:
            assert uow.targets is not None and uow.recovery is not None
            target = await uow.targets.get_scoped(
                target_id=target_id, domain_id=scope.domain_id, lock=True
            )
            if target is None:
                raise resource_not_found("Target")
            db_type = str(target.db_type)
            if request.recovery_marker.kind != db_type:
                raise validation_failed("恢复坐标类型必须与 Target 数据库类型一致")
            source_db_type = _SOURCE_DB_TYPES.get(request.backup_source_type)
            if source_db_type is not None and source_db_type != db_type:
                raise validation_failed("备份来源类型必须与 Target 数据库类型一致")
            profile = await uow.recovery.get_active_profile(
                target_id=target_id, domain_id=scope.domain_id
            )
            rpo = (
                int((request.simulated_failure_at - request.recovered_through_at).total_seconds())
                if request.recovered_through_at is not None
                else None
            )
            rto = (
                int((request.service_validated_at - request.simulated_failure_at).total_seconds())
                if request.service_validated_at is not None
                else None
            )
            entity = RecoveryDrillEntity(
                drill_id=uuid7(),
                target_id=target_id,
                domain_id=scope.domain_id,
                recovery_profile_id=(profile.recovery_profile_id if profile else None),
                recovery_profile_version=(int(profile.version_no) if profile else None),
                db_type=db_type,
                scenario=request.scenario,
                assurance_level=request.assurance_level,
                backup_source_type=request.backup_source_type,
                environment=request.environment,
                result=request.result,
                status="SUBMITTED",
                simulated_failure_at=request.simulated_failure_at,
                recovered_through_at=request.recovered_through_at,
                service_validated_at=request.service_validated_at,
                achieved_rpo_seconds=rpo,
                achieved_rto_seconds=rto,
                recovery_marker_json=request.recovery_marker.model_dump(mode="json"),
                evidence_json=[item.model_dump(mode="json") for item in request.evidence],
                source_trust_level="USER_PROVIDED",
                source_ops_run_id=None,
                reviewed_by=None,
                reviewed_at=None,
                review_note=None,
                notes=request.notes,
                row_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await uow.recovery.add_drill(entity)
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="RECOVERY_DRILL",
                aggregate_id=entity.drill_id,
                event_type="RECOVERY_DRILL_SUBMITTED",
                row_version=1,
                details={"target_id": str(target_id), "result": entity.result},
            )
            return recovery_drill_view(entity)

        return await self._idempotent(
            scope=scope,
            operation="RECOVERY_DRILL_CREATE",
            parent_resource=str(target_id),
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=RecoveryDrillView,
            handler=handler,
        )

    async def review_recovery_drill(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        drill_id: UUID,
        request: RecoveryDrillReview,
        expected_version: int,
        idempotency_key: str,
    ) -> RecoveryDrillView:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> RecoveryDrillView:
            assert uow.recovery is not None
            entity = await uow.recovery.get_drill_scoped(
                drill_id=drill_id,
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Recovery Drill")
            self._check_version(entity.row_version, expected_version)
            if entity.status != "SUBMITTED":
                raise state_conflict("只有待审核演练记录可以审核")
            entity.status = "VERIFIED" if request.decision == "VERIFY" else "REJECTED"
            entity.reviewed_by = scope.actor_id
            entity.reviewed_at = now
            entity.review_note = request.review_note
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            # 人工审核只确认记录流程，不提升来源信任等级。
            entity.source_trust_level = "USER_PROVIDED"
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="RECOVERY_DRILL",
                aggregate_id=entity.drill_id,
                event_type=f"RECOVERY_DRILL_{entity.status}",
                row_version=int(entity.row_version),
                details={"target_id": str(target_id)},
            )
            return recovery_drill_view(entity)

        return await self._idempotent(
            scope=scope,
            operation="RECOVERY_DRILL_REVIEW",
            parent_resource=str(drill_id),
            idempotency_key=idempotency_key,
            payload={**request.model_dump(mode="json"), "row_version": expected_version},
            response_type=RecoveryDrillView,
            handler=handler,
        )
