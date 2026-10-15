"""业务系统资源图谱 Repository。"""

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    BusinessSystemDraftChangeEntity,
    BusinessSystemDraftEntity,
    BusinessSystemEntity,
    BusinessSystemResourceEntity,
    ComputeResourceEntity,
    DatabaseResourceEntity,
    ManagedResourceEntity,
    RelationParticipantEntity,
    ResourceDiscoveryJobEntity,
    ResourceRelationEntity,
    TargetEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class BusinessSystemRepository(AIOpsRepository):
    def __init__(self, session: AsyncSession, assert_active: Callable[[], None] | None = None) -> None:
        super().__init__(session, assert_active)

    async def add_system(self, entity: BusinessSystemEntity) -> BusinessSystemEntity:
        return await self._add(entity)

    async def add_resource(self, entity: ManagedResourceEntity) -> ManagedResourceEntity:
        return await self._add(entity)

    async def add_database_resource(self, entity: DatabaseResourceEntity) -> DatabaseResourceEntity:
        return await self._add(entity)

    async def add_compute_resource(self, entity: ComputeResourceEntity) -> ComputeResourceEntity:
        return await self._add(entity)

    async def add_system_resources(self, entities: list[BusinessSystemResourceEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def add_relations(self, entities: list[ResourceRelationEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def add_participants(self, entities: list[RelationParticipantEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def add_draft(self, entity: BusinessSystemDraftEntity) -> BusinessSystemDraftEntity:
        return await self._add(entity)

    async def add_draft_change(self, entity: BusinessSystemDraftChangeEntity) -> BusinessSystemDraftChangeEntity:
        return await self._add(entity)

    async def add_discovery_job(self, entity: ResourceDiscoveryJobEntity) -> ResourceDiscoveryJobEntity:
        return await self._add(entity)

    async def get_scoped(self, *, business_system_id: UUID, domain_id: int, lock: bool = False, include_deleted: bool = False) -> BusinessSystemEntity | None:
        self._check_active()
        statement = select(BusinessSystemEntity).where(
            BusinessSystemEntity.business_system_id == business_system_id,
            BusinessSystemEntity.domain_id == domain_id,
        )
        if not include_deleted:
            statement = statement.where(BusinessSystemEntity.status != "DELETED")
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_scoped(self, *, domain_id: int, status: str | None, limit: int) -> list[BusinessSystemEntity]:
        self._check_active()
        statement = select(BusinessSystemEntity).where(
            BusinessSystemEntity.domain_id == domain_id,
            BusinessSystemEntity.status != "DELETED",
        )
        if status is not None:
            statement = statement.where(BusinessSystemEntity.status == status)
        statement = statement.order_by(BusinessSystemEntity.updated_at.desc(), BusinessSystemEntity.business_system_id.desc()).limit(limit)
        return list((await self._session.execute(statement)).scalars())
    async def get_by_name(self, *, domain_id: int, display_name: str) -> BusinessSystemEntity | None:
        self._check_active()
        statement = select(BusinessSystemEntity).where(
            BusinessSystemEntity.domain_id == domain_id,
            BusinessSystemEntity.status != "DELETED",
            func.lower(BusinessSystemEntity.display_name) == display_name.casefold(),
        ).limit(1)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_code(self, *, domain_id: int, system_code: str) -> BusinessSystemEntity | None:
        self._check_active()
        statement = select(BusinessSystemEntity).where(
            BusinessSystemEntity.domain_id == domain_id,
            BusinessSystemEntity.status != "DELETED",
            func.lower(BusinessSystemEntity.system_code) == system_code.casefold(),
        ).limit(1)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_resource_scoped(self, *, resource_id: UUID, domain_id: int, lock: bool = False) -> ManagedResourceEntity | None:
        self._check_active()
        statement = select(ManagedResourceEntity).where(
            ManagedResourceEntity.resource_id == resource_id,
            ManagedResourceEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_database_profile(self, resource_id: UUID) -> DatabaseResourceEntity | None:
        self._check_active()
        return (await self._session.execute(select(DatabaseResourceEntity).where(DatabaseResourceEntity.resource_id == resource_id))).scalar_one_or_none()

    async def list_resources_scoped(self, *, domain_id: int, resource_type_key: str | None, limit: int) -> list[ManagedResourceEntity]:
        self._check_active()
        statement = select(ManagedResourceEntity).where(
            ManagedResourceEntity.domain_id == domain_id,
            ManagedResourceEntity.lifecycle_status != "RETIRED",
        )
        if resource_type_key is not None:
            statement = statement.where(ManagedResourceEntity.resource_type_key == resource_type_key)
        statement = statement.order_by(ManagedResourceEntity.updated_at.desc(), ManagedResourceEntity.resource_id.desc()).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def get_system_resource_scoped(self, *, business_system_id: UUID, system_resource_id: UUID, lock: bool = False) -> BusinessSystemResourceEntity | None:
        self._check_active()
        statement = select(BusinessSystemResourceEntity).where(
            BusinessSystemResourceEntity.business_system_id == business_system_id,
            BusinessSystemResourceEntity.system_resource_id == system_resource_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_relation_scoped(self, *, business_system_id: UUID, relation_id: UUID, lock: bool = False) -> ResourceRelationEntity | None:
        self._check_active()
        statement = select(ResourceRelationEntity).where(
            ResourceRelationEntity.business_system_id == business_system_id,
            ResourceRelationEntity.relation_id == relation_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def delete_relation_participants(self, relation_id: UUID) -> None:
        self._check_active()
        await self._session.execute(delete(RelationParticipantEntity).where(RelationParticipantEntity.relation_id == relation_id))

    async def delete_relation(self, relation_id: UUID) -> None:
        self._check_active()
        await self.delete_relation_participants(relation_id)
        await self._session.execute(delete(ResourceRelationEntity).where(ResourceRelationEntity.relation_id == relation_id))

    async def remove_system_resource(self, system_resource_id: UUID) -> None:
        self._check_active()
        relation_ids = list((await self._session.execute(select(RelationParticipantEntity.relation_id).where(
            RelationParticipantEntity.system_resource_id == system_resource_id
        ))).scalars())
        if relation_ids:
            await self._session.execute(delete(RelationParticipantEntity).where(RelationParticipantEntity.relation_id.in_(relation_ids)))
            await self._session.execute(delete(ResourceRelationEntity).where(ResourceRelationEntity.relation_id.in_(relation_ids)))
        await self._session.execute(delete(BusinessSystemResourceEntity).where(BusinessSystemResourceEntity.system_resource_id == system_resource_id))

    async def get_discovery_job_scoped(self, *, discovery_job_id: UUID, draft_id: UUID) -> ResourceDiscoveryJobEntity | None:
        self._check_active()
        statement = select(ResourceDiscoveryJobEntity).where(
            ResourceDiscoveryJobEntity.discovery_job_id == discovery_job_id,
            ResourceDiscoveryJobEntity.draft_id == draft_id,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_resource_for_target(self, *, target_id: UUID, domain_id: int) -> ManagedResourceEntity | None:
        self._check_active()
        statement = (
            select(ManagedResourceEntity)
            .join(DatabaseResourceEntity, DatabaseResourceEntity.resource_id == ManagedResourceEntity.resource_id)
            .where(
                DatabaseResourceEntity.target_id == target_id,
                ManagedResourceEntity.domain_id == domain_id,
            )
            .limit(1)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_database_resource_for_target(self, *, target_id: UUID, domain_id: int) -> tuple[ManagedResourceEntity, DatabaseResourceEntity] | None:
        self._check_active()
        statement = (
            select(ManagedResourceEntity, DatabaseResourceEntity)
            .join(DatabaseResourceEntity, DatabaseResourceEntity.resource_id == ManagedResourceEntity.resource_id)
            .where(
                DatabaseResourceEntity.target_id == target_id,
                ManagedResourceEntity.domain_id == domain_id,
            )
            .limit(1)
        )
        return (await self._session.execute(statement)).one_or_none()

    async def list_system_resource_rows(self, business_system_id: UUID) -> list[tuple[BusinessSystemResourceEntity, ManagedResourceEntity, DatabaseResourceEntity | None, TargetEntity | None]]:
        self._check_active()
        statement = (
            select(BusinessSystemResourceEntity, ManagedResourceEntity, DatabaseResourceEntity, TargetEntity)
            .join(ManagedResourceEntity, ManagedResourceEntity.resource_id == BusinessSystemResourceEntity.resource_id)
            .outerjoin(DatabaseResourceEntity, DatabaseResourceEntity.resource_id == ManagedResourceEntity.resource_id)
            .outerjoin(TargetEntity, TargetEntity.target_id == DatabaseResourceEntity.target_id)
            .where(BusinessSystemResourceEntity.business_system_id == business_system_id)
            .order_by(BusinessSystemResourceEntity.display_order, BusinessSystemResourceEntity.system_resource_id)
        )
        return list((await self._session.execute(statement)).all())

    async def list_relations(self, business_system_id: UUID) -> list[ResourceRelationEntity]:
        self._check_active()
        statement = select(ResourceRelationEntity).where(ResourceRelationEntity.business_system_id == business_system_id).order_by(ResourceRelationEntity.created_at, ResourceRelationEntity.relation_id)
        return list((await self._session.execute(statement)).scalars())

    async def list_participants(self, relation_ids: list[UUID]) -> list[RelationParticipantEntity]:
        self._check_active()
        if not relation_ids:
            return []
        statement = select(RelationParticipantEntity).where(RelationParticipantEntity.relation_id.in_(relation_ids)).order_by(RelationParticipantEntity.display_order, RelationParticipantEntity.participant_id)
        return list((await self._session.execute(statement)).scalars())

    async def resource_count(self, business_system_id: UUID) -> int:
        self._check_active()
        statement = select(func.count()).select_from(BusinessSystemResourceEntity).where(
            BusinessSystemResourceEntity.business_system_id == business_system_id,
            BusinessSystemResourceEntity.status == "ACTIVE",
        )
        return int((await self._session.execute(statement)).scalar_one())

    async def find_system_for_target(self, *, target_id: UUID, domain_id: int) -> BusinessSystemEntity | None:
        self._check_active()
        statement = (
            select(BusinessSystemEntity)
            .join(BusinessSystemResourceEntity, BusinessSystemResourceEntity.business_system_id == BusinessSystemEntity.business_system_id)
            .join(DatabaseResourceEntity, DatabaseResourceEntity.resource_id == BusinessSystemResourceEntity.resource_id)
            .where(
                DatabaseResourceEntity.target_id == target_id,
                BusinessSystemEntity.domain_id == domain_id,
                BusinessSystemResourceEntity.status == "ACTIVE",
            )
            .order_by(BusinessSystemEntity.updated_at.desc())
            .limit(1)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find_contexts_for_targets(self, *, target_ids: set[UUID], domain_id: int) -> dict[UUID, tuple[UUID, str, UUID, str]]:
        """批量返回 Target 所属业务系统和资源，供监控选择器分组。"""

        self._check_active()
        if not target_ids:
            return {}
        statement = (
            select(
                DatabaseResourceEntity.target_id,
                BusinessSystemEntity.business_system_id,
                BusinessSystemEntity.display_name,
                ManagedResourceEntity.resource_id,
                ManagedResourceEntity.display_name,
            )
            .join(ManagedResourceEntity, ManagedResourceEntity.resource_id == DatabaseResourceEntity.resource_id)
            .join(BusinessSystemResourceEntity, BusinessSystemResourceEntity.resource_id == ManagedResourceEntity.resource_id)
            .join(BusinessSystemEntity, BusinessSystemEntity.business_system_id == BusinessSystemResourceEntity.business_system_id)
            .where(
                DatabaseResourceEntity.target_id.in_(target_ids),
                BusinessSystemEntity.domain_id == domain_id,
                BusinessSystemResourceEntity.status == "ACTIVE",
            )
        )
        return {
            target_id: (system_id, system_name, resource_id, resource_name)
            for target_id, system_id, system_name, resource_id, resource_name in (await self._session.execute(statement)).all()
        }

    async def target_has_resource(self, *, target_id: UUID) -> bool:
        self._check_active()
        count = (await self._session.execute(select(func.count()).select_from(DatabaseResourceEntity).where(DatabaseResourceEntity.target_id == target_id))).scalar_one()
        return bool(count)

    async def get_draft_scoped(self, *, draft_id: UUID, domain_id: int, lock: bool = False) -> BusinessSystemDraftEntity | None:
        self._check_active()
        statement = select(BusinessSystemDraftEntity).where(
            BusinessSystemDraftEntity.draft_id == draft_id,
            BusinessSystemDraftEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_active_drafts(self, *, domain_id: int, now: datetime, limit: int) -> list[BusinessSystemDraftEntity]:
        self._check_active()
        statement = (
            select(BusinessSystemDraftEntity)
            .where(
                BusinessSystemDraftEntity.domain_id == domain_id,
                BusinessSystemDraftEntity.status == "ACTIVE",
                BusinessSystemDraftEntity.expires_at > now,
            )
            .order_by(BusinessSystemDraftEntity.updated_at.desc(), BusinessSystemDraftEntity.draft_id.desc())
            .limit(limit)
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_draft_changes(self, draft_id: UUID) -> list[BusinessSystemDraftChangeEntity]:
        self._check_active()
        statement = select(BusinessSystemDraftChangeEntity).where(BusinessSystemDraftChangeEntity.draft_id == draft_id).order_by(BusinessSystemDraftChangeEntity.sequence_no)
        return list((await self._session.execute(statement)).scalars())
