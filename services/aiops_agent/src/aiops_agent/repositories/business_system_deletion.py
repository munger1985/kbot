"""业务系统删除的引用核验与定向清理，不删除资源或运行历史。"""

from uuid import UUID

from sqlalchemy import delete, or_, select

from aiops_agent.domain.operations.run import TERMINAL_RUN_STATUSES
from aiops_agent.entities import (
    AIOpsAgentVersionBusinessSystemEntity, BusinessSystemDraftChangeEntity,
    BusinessSystemDraftEntity, BusinessSystemEntity, BusinessSystemResourceEntity,
    DatabaseResourceEntity, DiagnosticSourceEntity, InspectionFireEntity,
    InspectionPlanEntity, ManagedResourceEntity, OpsRunEntity, OutboxEntity,
    RelationParticipantEntity, ResourceDiscoveryJobEntity, ResourceRelationEntity,
    TargetEntity,
)


class BusinessSystemDeletionRepositoryMixin:
    """复用当前事务与作用域，不提供全局级联删除入口。"""

    async def system_has_publication_history(self, system_id: UUID) -> bool:
        self._check_active()
        statements = (
            select(BusinessSystemDraftEntity.draft_id).where(
                BusinessSystemDraftEntity.business_system_id == system_id,
                BusinessSystemDraftEntity.status == "COMMITTED"),
            select(AIOpsAgentVersionBusinessSystemEntity.business_system_id).where(
                AIOpsAgentVersionBusinessSystemEntity.business_system_id == system_id),
            select(InspectionPlanEntity.business_system_id).where(
                InspectionPlanEntity.business_system_id == system_id),
            select(OutboxEntity.outbox_id).where(
                OutboxEntity.aggregate_type == "BUSINESS_SYSTEM",
                OutboxEntity.aggregate_id == system_id,
                OutboxEntity.event_type.in_((
                    "BUSINESS_SYSTEM_ENABLED", "BUSINESS_SYSTEM_DISABLED",
                    "BUSINESS_SYSTEM_RESOURCE_GRAPH_CREATED",
                ))),
        )
        for statement in statements:
            if (await self._session.execute(statement.limit(1))).scalar_one_or_none() is not None:
                return True
        return False

    async def resource_has_other_system(self, *, resource_id: UUID, system: BusinessSystemEntity) -> bool:
        """下线和已登记草稿也拥有资源；只排除已删除系统与无效关联。"""
        self._check_active()
        statement = (
            select(BusinessSystemEntity.business_system_id)
            .join(BusinessSystemResourceEntity,
                  BusinessSystemResourceEntity.business_system_id == BusinessSystemEntity.business_system_id)
            .where(
                BusinessSystemResourceEntity.resource_id == resource_id,
                BusinessSystemResourceEntity.status == "ACTIVE",
                BusinessSystemEntity.business_system_id != system.business_system_id,
                BusinessSystemEntity.domain_id == system.domain_id,
                BusinessSystemEntity.status != "DELETED",
            )
        )
        if hasattr(system, "tenant_id"):
            statement = statement.where(BusinessSystemEntity.tenant_id == system.tenant_id)
        return (await self._session.execute(statement.limit(1))).scalar_one_or_none() is not None

    async def target_has_unretired_resource(self, target_id: UUID) -> bool:
        self._check_active()
        statement = select(DatabaseResourceEntity.resource_id).join(
            ManagedResourceEntity, ManagedResourceEntity.resource_id == DatabaseResourceEntity.resource_id,
        ).where(DatabaseResourceEntity.target_id == target_id, ManagedResourceEntity.lifecycle_status != "RETIRED")
        return (await self._session.execute(statement.limit(1))).scalar_one_or_none() is not None

    async def credential_has_other_owner(self, *, credential_id: UUID, target_id: UUID) -> bool:
        """凭据也可能被监控源或发现任务引用，不能仅按 Target 判断独占。"""
        self._check_active()
        statements = (
            select(TargetEntity.target_id).where(
                TargetEntity.target_id != target_id,
                or_(TargetEntity.diagnostic_credential_id == credential_id,
                    TargetEntity.execution_credential_id == credential_id)),
            select(DiagnosticSourceEntity.diagnostic_source_id).where(or_(
                DiagnosticSourceEntity.auth_credential_id == credential_id,
                DiagnosticSourceEntity.webhook_credential_id == credential_id)),
            select(ResourceDiscoveryJobEntity.discovery_job_id).where(
                ResourceDiscoveryJobEntity.credential_id == credential_id),
        )
        for statement in statements:
            if (await self._session.execute(statement.limit(1))).scalar_one_or_none() is not None:
                return True
        return False

    async def list_deletion_plans(self, system_id: UUID) -> list[InspectionPlanEntity]:
        self._check_active()
        return list((await self._session.execute(
            select(InspectionPlanEntity).where(InspectionPlanEntity.business_system_id == system_id)
            .order_by(InspectionPlanEntity.inspection_plan_id).with_for_update()
            .execution_options(populate_existing=True),
        )).scalars())

    async def system_has_unfinished_work(self, *, system_id: UUID, target_ids: set[UUID]) -> bool:
        self._check_active()
        versions = select(AIOpsAgentVersionBusinessSystemEntity.agent_version_id).where(
            AIOpsAgentVersionBusinessSystemEntity.business_system_id == system_id)
        plans = select(InspectionPlanEntity.inspection_plan_id).where(
            InspectionPlanEntity.business_system_id == system_id)
        fires = select(InspectionFireEntity.inspection_fire_id).where(
            InspectionFireEntity.inspection_plan_id.in_(plans))
        relevant = or_(
            OpsRunEntity.inspection_fire_id.in_(fires),
            (OpsRunEntity.agent_version_id.in_(versions)
             & or_(OpsRunEntity.target_id.in_(target_ids), OpsRunEntity.target_id.is_(None))),
        )
        statement = select(OpsRunEntity.ops_run_id).where(
            relevant, OpsRunEntity.status.not_in(tuple(item.value for item in TERMINAL_RUN_STATUSES)))
        if (await self._session.execute(statement.limit(1))).scalar_one_or_none() is not None:
            return True
        statement = select(InspectionFireEntity.inspection_fire_id).where(
            InspectionFireEntity.inspection_plan_id.in_(plans),
            InspectionFireEntity.status.in_(("PLANNED", "QUEUED", "RUNNING")),
        )
        return (await self._session.execute(statement.limit(1))).scalar_one_or_none() is not None

    async def list_uncommitted_system_drafts(self, system_id: UUID) -> list[BusinessSystemDraftEntity]:
        self._check_active()
        return list((await self._session.execute(
            select(BusinessSystemDraftEntity).where(
                BusinessSystemDraftEntity.business_system_id == system_id,
                BusinessSystemDraftEntity.status != "COMMITTED",
            ).order_by(BusinessSystemDraftEntity.draft_id).with_for_update()
        )).scalars())

    async def draft_has_pending_discovery(self, draft_id: UUID) -> bool:
        self._check_active()
        statement = select(ResourceDiscoveryJobEntity.discovery_job_id).where(
            ResourceDiscoveryJobEntity.draft_id == draft_id,
            ResourceDiscoveryJobEntity.status.in_(("PENDING", "RUNNING")),
        )
        return (await self._session.execute(statement.limit(1))).scalar_one_or_none() is not None

    async def delete_draft_workspace(self, draft_id: UUID) -> None:
        """先清理子表，Oracle 与 PostgreSQL 使用同样的产品语义。"""
        self._check_active()
        for entity in (ResourceDiscoveryJobEntity, BusinessSystemDraftChangeEntity, BusinessSystemDraftEntity):
            await self._session.execute(delete(entity).where(entity.draft_id == draft_id))

    async def delete_unpublished_system(self, system_id: UUID) -> None:
        self._check_active()
        relation_ids = select(ResourceRelationEntity.relation_id).where(
            ResourceRelationEntity.business_system_id == system_id)
        await self._session.execute(delete(RelationParticipantEntity).where(
            RelationParticipantEntity.relation_id.in_(relation_ids)))
        for entity in (ResourceRelationEntity, BusinessSystemResourceEntity, BusinessSystemEntity):
            await self._session.execute(delete(entity).where(entity.business_system_id == system_id))
