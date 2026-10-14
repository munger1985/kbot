"""业务系统、可管理组件与资源拓扑 Repository。"""

from collections.abc import Callable
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    BusinessSystemComponentEntity,
    BusinessSystemEntity,
    BusinessSystemSiteEntity,
    ComponentMembershipEntity,
    ManagedComponentEntity,
    ResourceRelationEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class BusinessSystemRepository(AIOpsRepository):
    def __init__(self, session: AsyncSession, assert_active: Callable[[], None] | None = None) -> None:
        super().__init__(session, assert_active)

    async def add_system(self, entity: BusinessSystemEntity) -> BusinessSystemEntity:
        return await self._add(entity)

    async def add_components(self, entities: list[ManagedComponentEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def add_component_links(self, entities: list[BusinessSystemComponentEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def add_sites(self, entities: list[BusinessSystemSiteEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def add_members(self, entities: list[ComponentMembershipEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def add_relations(self, entities: list[ResourceRelationEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def get_scoped(self, *, business_system_id: UUID, domain_id: int, lock: bool = False) -> BusinessSystemEntity | None:
        self._check_active()
        statement = select(BusinessSystemEntity).where(
            BusinessSystemEntity.business_system_id == business_system_id,
            BusinessSystemEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_scoped(self, *, domain_id: int, status: str | None, limit: int) -> list[BusinessSystemEntity]:
        self._check_active()
        statement = select(BusinessSystemEntity).where(
            BusinessSystemEntity.domain_id == domain_id,
        )
        if status is not None:
            statement = statement.where(BusinessSystemEntity.status == status)
        return list((await self._session.execute(statement.order_by(BusinessSystemEntity.updated_at.desc(), BusinessSystemEntity.business_system_id.desc()).limit(limit))).scalars())

    async def get_by_name(self, *, domain_id: int, display_name: str) -> BusinessSystemEntity | None:
        self._check_active()
        return (await self._session.execute(select(BusinessSystemEntity).where(
            BusinessSystemEntity.domain_id == domain_id,
            func.lower(BusinessSystemEntity.display_name) == display_name.casefold(),
        ).limit(1))).scalar_one_or_none()

    async def get_by_code(self, *, domain_id: int, system_code: str) -> BusinessSystemEntity | None:
        self._check_active()
        return (await self._session.execute(select(BusinessSystemEntity).where(
            BusinessSystemEntity.domain_id == domain_id,
            func.lower(BusinessSystemEntity.system_code) == system_code.casefold(),
        ).limit(1))).scalar_one_or_none()

    async def list_sites(self, business_system_id: UUID) -> list[BusinessSystemSiteEntity]:
        self._check_active()
        return list((await self._session.execute(select(BusinessSystemSiteEntity).where(BusinessSystemSiteEntity.business_system_id == business_system_id).order_by(BusinessSystemSiteEntity.sort_order, BusinessSystemSiteEntity.site_id))).scalars())

    async def list_components(self, business_system_id: UUID) -> list[ManagedComponentEntity]:
        self._check_active()
        statement = (
            select(ManagedComponentEntity)
            .join(BusinessSystemComponentEntity, BusinessSystemComponentEntity.component_id == ManagedComponentEntity.component_id)
            .where(BusinessSystemComponentEntity.business_system_id == business_system_id)
            .order_by(BusinessSystemComponentEntity.sort_order, ManagedComponentEntity.component_id)
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_component_links(self, business_system_id: UUID) -> list[BusinessSystemComponentEntity]:
        self._check_active()
        return list((await self._session.execute(select(BusinessSystemComponentEntity).where(
            BusinessSystemComponentEntity.business_system_id == business_system_id
        ))).scalars())

    async def get_component_scoped(self, *, business_system_id: UUID, component_id: UUID, domain_id: int, lock: bool = False) -> ManagedComponentEntity | None:
        self._check_active()
        statement = (
            select(ManagedComponentEntity)
            .join(BusinessSystemComponentEntity, BusinessSystemComponentEntity.component_id == ManagedComponentEntity.component_id)
            .join(BusinessSystemEntity, BusinessSystemEntity.business_system_id == BusinessSystemComponentEntity.business_system_id)
            .where(
                BusinessSystemComponentEntity.business_system_id == business_system_id,
                ManagedComponentEntity.component_id == component_id,
                BusinessSystemEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_component_by_id_scoped(self, *, component_id: UUID, domain_id: int, lock: bool = False) -> ManagedComponentEntity | None:
        self._check_active()
        statement = select(ManagedComponentEntity).where(
            ManagedComponentEntity.component_id == component_id,
            ManagedComponentEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_members(self, business_system_id: UUID) -> list[ComponentMembershipEntity]:
        self._check_active()
        statement = (
            select(ComponentMembershipEntity)
            .join(BusinessSystemComponentEntity, BusinessSystemComponentEntity.component_id == ComponentMembershipEntity.component_id)
            .where(BusinessSystemComponentEntity.business_system_id == business_system_id)
            .order_by(ComponentMembershipEntity.sort_order, ComponentMembershipEntity.membership_id)
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_component_members(self, component_id: UUID) -> list[ComponentMembershipEntity]:
        self._check_active()
        return list((await self._session.execute(select(ComponentMembershipEntity).where(ComponentMembershipEntity.component_id == component_id).order_by(ComponentMembershipEntity.sort_order, ComponentMembershipEntity.membership_id))).scalars())

    async def list_relations(self, business_system_id: UUID) -> list[ResourceRelationEntity]:
        self._check_active()
        return list((await self._session.execute(select(ResourceRelationEntity).where(ResourceRelationEntity.business_system_id == business_system_id).order_by(ResourceRelationEntity.sort_order, ResourceRelationEntity.relation_id))).scalars())

    async def component_count(self, business_system_id: UUID) -> int:
        self._check_active()
        return int((await self._session.execute(select(func.count()).select_from(BusinessSystemComponentEntity).where(BusinessSystemComponentEntity.business_system_id == business_system_id))).scalar_one())

    async def member_count(self, business_system_id: UUID) -> int:
        self._check_active()
        statement = select(func.count()).select_from(ComponentMembershipEntity).join(
            BusinessSystemComponentEntity,
            BusinessSystemComponentEntity.component_id == ComponentMembershipEntity.component_id,
        ).where(BusinessSystemComponentEntity.business_system_id == business_system_id)
        return int((await self._session.execute(statement)).scalar_one())

    async def find_system_for_target(self, *, target_id: UUID, domain_id: int) -> BusinessSystemEntity | None:
        self._check_active()
        statement = (
            select(BusinessSystemEntity)
            .join(BusinessSystemComponentEntity, BusinessSystemComponentEntity.business_system_id == BusinessSystemEntity.business_system_id)
            .join(ComponentMembershipEntity, ComponentMembershipEntity.component_id == BusinessSystemComponentEntity.component_id)
            .where(
                ComponentMembershipEntity.target_id == target_id,
                BusinessSystemEntity.domain_id == domain_id,
            )
            .limit(1)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find_component_for_target(self, *, target_id: UUID, domain_id: int) -> ManagedComponentEntity | None:
        self._check_active()
        statement = select(ManagedComponentEntity).join(
            ComponentMembershipEntity,
            ComponentMembershipEntity.component_id == ManagedComponentEntity.component_id,
        ).where(
            ComponentMembershipEntity.target_id == target_id,
            ManagedComponentEntity.domain_id == domain_id,
        ).limit(1)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find_contexts_for_targets(
        self,
        *,
        target_ids: set[UUID],
        domain_id: int,
    ) -> dict[UUID, tuple[UUID, str, UUID, str]]:
        """批量返回 Target 所属业务系统和组件，供监控选择器分组。"""

        self._check_active()
        if not target_ids:
            return {}
        statement = (
            select(
                ComponentMembershipEntity.target_id,
                BusinessSystemEntity.business_system_id,
                BusinessSystemEntity.display_name,
                ManagedComponentEntity.component_id,
                ManagedComponentEntity.display_name,
            )
            .join(ManagedComponentEntity, ManagedComponentEntity.component_id == ComponentMembershipEntity.component_id)
            .join(BusinessSystemComponentEntity, BusinessSystemComponentEntity.component_id == ManagedComponentEntity.component_id)
            .join(BusinessSystemEntity, BusinessSystemEntity.business_system_id == BusinessSystemComponentEntity.business_system_id)
            .where(
                ComponentMembershipEntity.target_id.in_(target_ids),
                BusinessSystemEntity.domain_id == domain_id,
            )
        )
        return {
            target_id: (system_id, system_name, component_id, component_name)
            for target_id, system_id, system_name, component_id, component_name in (await self._session.execute(statement)).all()
        }

    async def target_is_assigned_elsewhere(self, *, target_id: UUID, component_id: UUID) -> bool:
        self._check_active()
        count = (await self._session.execute(select(func.count()).select_from(ComponentMembershipEntity).where(ComponentMembershipEntity.target_id == target_id, ComponentMembershipEntity.component_id != component_id))).scalar_one()
        return bool(count)

    async def target_is_assigned(self, *, target_id: UUID) -> bool:
        self._check_active()
        count = (await self._session.execute(select(func.count()).select_from(ComponentMembershipEntity).where(ComponentMembershipEntity.target_id == target_id))).scalar_one()
        return bool(count)

    async def clear_topology_edges(self, business_system_id: UUID) -> None:
        self._check_active()
        component_ids = list((await self._session.execute(
            select(BusinessSystemComponentEntity.component_id).where(
                BusinessSystemComponentEntity.business_system_id == business_system_id
            )
        )).scalars())
        await self._session.execute(delete(ResourceRelationEntity).where(ResourceRelationEntity.business_system_id == business_system_id))
        if component_ids:
            await self._session.execute(delete(ComponentMembershipEntity).where(ComponentMembershipEntity.component_id.in_(component_ids)))
        await self._session.execute(delete(BusinessSystemSiteEntity).where(BusinessSystemSiteEntity.business_system_id == business_system_id))

    async def remove_components(self, *, business_system_id: UUID, component_ids: list[UUID]) -> None:
        self._check_active()
        if not component_ids:
            return
        await self._session.execute(delete(BusinessSystemComponentEntity).where(
            BusinessSystemComponentEntity.business_system_id == business_system_id,
            BusinessSystemComponentEntity.component_id.in_(component_ids),
        ))
        await self._session.execute(delete(ManagedComponentEntity).where(
            ManagedComponentEntity.component_id.in_(component_ids),
            ~ManagedComponentEntity.component_id.in_(select(BusinessSystemComponentEntity.component_id)),
        ))

    async def replace_topology(self, business_system_id: UUID) -> None:
        self._check_active()
        component_ids = list((await self._session.execute(
            select(BusinessSystemComponentEntity.component_id).where(
                BusinessSystemComponentEntity.business_system_id == business_system_id
            )
        )).scalars())
        await self._session.execute(delete(ResourceRelationEntity).where(ResourceRelationEntity.business_system_id == business_system_id))
        if component_ids:
            await self._session.execute(delete(ComponentMembershipEntity).where(ComponentMembershipEntity.component_id.in_(component_ids)))
        await self._session.execute(delete(BusinessSystemComponentEntity).where(BusinessSystemComponentEntity.business_system_id == business_system_id))
        await self._session.execute(delete(BusinessSystemSiteEntity).where(BusinessSystemSiteEntity.business_system_id == business_system_id))
        if component_ids:
            await self._session.execute(delete(ManagedComponentEntity).where(
                ManagedComponentEntity.component_id.in_(component_ids),
                ~ManagedComponentEntity.component_id.in_(select(BusinessSystemComponentEntity.component_id)),
            ))
