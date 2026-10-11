"""DBA 工作项聚合 Repository。"""

from collections.abc import Callable, Collection
from datetime import datetime
from uuid import UUID

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    ResponsibilityGroupEntity,
    ResponsibilityGroupMemberEntity,
    AIOpsAgentEntity,
    TargetEntity,
    WorkItemActivityEntity,
    WorkItemEntity,
    WorkItemLinkEntity,
    WorkItemOccurrenceEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


_TERMINAL_STATUSES = ("CLOSED", "CANCELLED")


class WorkItemRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add(self, entity: WorkItemEntity) -> WorkItemEntity:
        return await self._add(entity)

    async def add_occurrence(
        self, entity: WorkItemOccurrenceEntity
    ) -> WorkItemOccurrenceEntity:
        return await self._add(entity)

    async def add_link(self, entity: WorkItemLinkEntity) -> WorkItemLinkEntity:
        return await self._add(entity)

    async def add_activity(
        self, entity: WorkItemActivityEntity
    ) -> WorkItemActivityEntity:
        return await self._add(entity)

    async def add_group(self, entity: ResponsibilityGroupEntity) -> ResponsibilityGroupEntity:
        return await self._add(entity)

    async def add_group_member(self, entity: ResponsibilityGroupMemberEntity) -> ResponsibilityGroupMemberEntity:
        return await self._add(entity)

    async def get_group(self, *, group_id: UUID, domain_id: int, lock: bool = False) -> ResponsibilityGroupEntity | None:
        statement = select(ResponsibilityGroupEntity).where(
            ResponsibilityGroupEntity.responsibility_group_id == group_id,
            ResponsibilityGroupEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_groups(self, *, domain_id: int) -> list[ResponsibilityGroupEntity]:
        rows = await self._session.scalars(select(ResponsibilityGroupEntity).where(
            ResponsibilityGroupEntity.domain_id == domain_id,
        ).order_by(ResponsibilityGroupEntity.name, ResponsibilityGroupEntity.responsibility_group_id))
        return list(rows)

    async def list_group_members(self, *, group_id: UUID) -> list[ResponsibilityGroupMemberEntity]:
        rows = await self._session.scalars(select(ResponsibilityGroupMemberEntity).where(
            ResponsibilityGroupMemberEntity.responsibility_group_id == group_id,
        ).order_by(ResponsibilityGroupMemberEntity.member_role, ResponsibilityGroupMemberEntity.user_id))
        return list(rows)

    async def get_group_member(self, *, group_id: UUID, user_id: str) -> ResponsibilityGroupMemberEntity | None:
        return (await self._session.execute(select(ResponsibilityGroupMemberEntity).where(
            ResponsibilityGroupMemberEntity.responsibility_group_id == group_id,
            ResponsibilityGroupMemberEntity.user_id == user_id,
        ))).scalar_one_or_none()

    async def active_member_count(self, *, group_id: UUID) -> int:
        value = await self._session.scalar(select(func.count()).select_from(ResponsibilityGroupMemberEntity).where(
            ResponsibilityGroupMemberEntity.responsibility_group_id == group_id,
            ResponsibilityGroupMemberEntity.status == "ACTIVE",
        ))
        return int(value or 0)

    async def group_routing_counts(self, *, group_id: UUID) -> tuple[int, int, int]:
        agent_count = await self._session.scalar(
            select(func.count()).select_from(AIOpsAgentEntity).where(
                AIOpsAgentEntity.default_responsibility_group_id == group_id
            )
        )
        target_count = await self._session.scalar(
            select(func.count()).select_from(TargetEntity).where(
                TargetEntity.default_responsibility_group_id == group_id
            )
        )
        unassigned_count = await self._session.scalar(
            select(func.count()).select_from(WorkItemEntity).where(
                WorkItemEntity.responsibility_group_id == group_id,
                WorkItemEntity.assignee_user_id.is_(None),
                WorkItemEntity.status.not_in(_TERMINAL_STATUSES),
            )
        )
        return (
            int(agent_count or 0),
            int(target_count or 0),
            int(unassigned_count or 0),
        )

    async def default_responsibility_group(self, *, target_id: UUID, agent_id: UUID) -> tuple[UUID | None, str | None]:
        target_group = await self._session.scalar(
            select(TargetEntity.default_responsibility_group_id)
            .join(
                ResponsibilityGroupEntity,
                ResponsibilityGroupEntity.responsibility_group_id
                == TargetEntity.default_responsibility_group_id,
            )
            .where(
                TargetEntity.target_id == target_id,
                ResponsibilityGroupEntity.status == "ACTIVE",
            )
        )
        if target_group is not None:
            return target_group, "TARGET_DEFAULT"
        agent_group = await self._session.scalar(
            select(AIOpsAgentEntity.default_responsibility_group_id)
            .join(
                ResponsibilityGroupEntity,
                ResponsibilityGroupEntity.responsibility_group_id
                == AIOpsAgentEntity.default_responsibility_group_id,
            )
            .where(
                AIOpsAgentEntity.agent_id == agent_id,
                ResponsibilityGroupEntity.status == "ACTIVE",
            )
        )
        return (agent_group, "AGENT_DEFAULT") if agent_group is not None else (None, None)

    async def group_name(self, *, group_id: UUID | None) -> str | None:
        if group_id is None:
            return None
        return await self._session.scalar(
            select(ResponsibilityGroupEntity.name).where(
                ResponsibilityGroupEntity.responsibility_group_id == group_id
            )
        )

    async def active_assignment_count(self, *, group_id: UUID, user_id: str) -> int:
        value = await self._session.scalar(
            select(func.count()).select_from(WorkItemEntity).where(
                WorkItemEntity.responsibility_group_id == group_id,
                WorkItemEntity.assignee_user_id == user_id,
                WorkItemEntity.status.not_in(_TERMINAL_STATUSES),
            )
        )
        return int(value or 0)

    async def work_item_ids_for_target_finding_types(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        finding_types: set[str],
    ) -> list[UUID]:
        if not finding_types:
            return []
        rows = await self._session.scalars(
            select(WorkItemEntity.work_item_id)
            .join(
                WorkItemOccurrenceEntity,
                WorkItemOccurrenceEntity.work_item_id == WorkItemEntity.work_item_id,
            )
            .where(
                WorkItemEntity.domain_id == domain_id,
                WorkItemEntity.target_id == target_id,
                WorkItemEntity.status.not_in(_TERMINAL_STATUSES),
                WorkItemOccurrenceEntity.finding_type.in_(tuple(finding_types)),
            )
            .distinct()
            .order_by(WorkItemEntity.work_item_id)
        )
        return list(rows)

    async def occurrence_exists(
        self, *, work_item_id: UUID, ops_run_id: UUID, finding_id: str
    ) -> bool:
        self._check_active()
        value = await self._session.scalar(
            select(func.count(WorkItemOccurrenceEntity.occurrence_id)).where(
                WorkItemOccurrenceEntity.work_item_id == work_item_id,
                WorkItemOccurrenceEntity.ops_run_id == ops_run_id,
                WorkItemOccurrenceEntity.finding_id == finding_id,
            )
        )
        return bool(value)

    async def get_scoped(
        self, *, work_item_id: UUID, domain_id: int, lock: bool = False
    ) -> WorkItemEntity | None:
        self._check_active()
        statement = select(WorkItemEntity).where(
            WorkItemEntity.work_item_id == work_item_id,
            WorkItemEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_open_by_fingerprint(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        fingerprint: str,
        lock: bool = False,
    ) -> WorkItemEntity | None:
        self._check_active()
        statement = (
            select(WorkItemEntity)
            .where(
                WorkItemEntity.domain_id == domain_id,
                WorkItemEntity.target_id == target_id,
                WorkItemEntity.fingerprint == fingerprint,
                WorkItemEntity.status.not_in(_TERMINAL_STATUSES),
            )
            .order_by(
                WorkItemEntity.updated_at.desc(),
                WorkItemEntity.work_item_id.desc(),
            )
        )
        if lock:
            statement = statement.with_for_update()
        else:
            statement = statement.limit(1)
        return (await self._session.execute(statement)).scalars().first()

    async def page(
        self,
        *,
        domain_id: int,
        status: str | None = None,
        priority: str | None = None,
        target_id: UUID | None = None,
        assignee_user_id: str | None = None,
        unassigned: bool = False,
        overdue_before: datetime | None = None,
        after_due_at: datetime | None = None,
        after_id: UUID | None = None,
        limit: int = 51,
    ) -> list[tuple[WorkItemEntity, str, str | None]]:
        self._check_active()
        statement = (
            select(
                WorkItemEntity,
                TargetEntity.display_name,
                ResponsibilityGroupEntity.name,
            )
            .join(TargetEntity, TargetEntity.target_id == WorkItemEntity.target_id)
            .outerjoin(
                ResponsibilityGroupEntity,
                ResponsibilityGroupEntity.responsibility_group_id
                == WorkItemEntity.responsibility_group_id,
            )
            .where(WorkItemEntity.domain_id == domain_id)
        )
        if status is None:
            statement = statement.where(
                WorkItemEntity.status.not_in(_TERMINAL_STATUSES)
            )
        else:
            statement = statement.where(WorkItemEntity.status == status)
        if priority is not None:
            priorities = tuple(value for value in priority.split(",") if value)
            statement = statement.where(WorkItemEntity.priority.in_(priorities))
        if target_id is not None:
            statement = statement.where(WorkItemEntity.target_id == target_id)
        if assignee_user_id is not None:
            statement = statement.where(
                WorkItemEntity.assignee_user_id == assignee_user_id
            )
        if unassigned:
            statement = statement.where(WorkItemEntity.assignee_user_id.is_(None))
        if overdue_before is not None:
            statement = statement.where(
                WorkItemEntity.status.not_in(_TERMINAL_STATUSES),
                WorkItemEntity.resolution_due_at < overdue_before,
            )
        if after_due_at is not None and after_id is not None:
            statement = statement.where(
                or_(
                    WorkItemEntity.resolution_due_at > after_due_at,
                    (WorkItemEntity.resolution_due_at == after_due_at)
                    & (WorkItemEntity.work_item_id > after_id),
                )
            )
        statement = statement.order_by(
            WorkItemEntity.resolution_due_at,
            WorkItemEntity.work_item_id,
        ).limit(limit)
        return list((await self._session.execute(statement)).all())

    async def queue_summary(
        self, *, domain_id: int, actor_id: str, now: datetime
    ) -> dict[str, int]:
        self._check_active()
        open_condition = WorkItemEntity.status.not_in(_TERMINAL_STATUSES)
        row = (
            await self._session.execute(
                select(
                    func.sum(case((open_condition & (WorkItemEntity.assignee_user_id == actor_id), 1), else_=0)),
                    func.sum(case((open_condition & WorkItemEntity.assignee_user_id.is_(None), 1), else_=0)),
                    func.sum(case((open_condition & WorkItemEntity.priority.in_(("P1", "P2")), 1), else_=0)),
                    func.sum(case((open_condition & (WorkItemEntity.resolution_due_at < now), 1), else_=0)),
                    func.sum(case((WorkItemEntity.status == "PENDING_VERIFICATION", 1), else_=0)),
                ).where(WorkItemEntity.domain_id == domain_id)
            )
        ).one()
        return {
            "my_open": int(row[0] or 0),
            "unassigned": int(row[1] or 0),
            "urgent": int(row[2] or 0),
            "overdue": int(row[3] or 0),
            "pending_verification": int(row[4] or 0),
        }

    async def target_name(self, *, target_id: UUID, domain_id: int) -> str | None:
        self._check_active()
        return await self._session.scalar(
            select(TargetEntity.display_name).where(
                TargetEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
        )

    async def list_occurrences(
        self, *, work_item_id: UUID
    ) -> list[WorkItemOccurrenceEntity]:
        self._check_active()
        rows = await self._session.scalars(
            select(WorkItemOccurrenceEntity)
            .where(WorkItemOccurrenceEntity.work_item_id == work_item_id)
            .order_by(WorkItemOccurrenceEntity.observed_at.desc())
        )
        return list(rows)

    async def list_links(self, *, work_item_id: UUID) -> list[WorkItemLinkEntity]:
        self._check_active()
        rows = await self._session.scalars(
            select(WorkItemLinkEntity)
            .where(WorkItemLinkEntity.work_item_id == work_item_id)
            .order_by(WorkItemLinkEntity.created_at.desc())
        )
        return list(rows)

    async def list_activities(
        self, *, work_item_id: UUID, limit: int = 200
    ) -> list[WorkItemActivityEntity]:
        self._check_active()
        rows = await self._session.scalars(
            select(WorkItemActivityEntity)
            .where(WorkItemActivityEntity.work_item_id == work_item_id)
            .order_by(WorkItemActivityEntity.created_at.desc())
            .limit(limit)
        )
        return list(rows)

    async def existing_link(
        self,
        *,
        work_item_id: UUID,
        resource_kind: str,
        resource_id: UUID,
        link_role: str,
    ) -> WorkItemLinkEntity | None:
        self._check_active()
        return (
            await self._session.execute(
                select(WorkItemLinkEntity).where(
                    WorkItemLinkEntity.work_item_id == work_item_id,
                    WorkItemLinkEntity.resource_kind == resource_kind,
                    WorkItemLinkEntity.resource_id == resource_id,
                    WorkItemLinkEntity.link_role == link_role,
                )
            )
        ).scalar_one_or_none()

    async def list_for_run_ids(
        self, *, ops_run_ids: Collection[UUID]
    ) -> list[WorkItemEntity]:
        self._check_active()
        normalized = tuple(dict.fromkeys(ops_run_ids))
        if not normalized:
            return []
        rows = await self._session.scalars(
            select(WorkItemEntity)
            .join(
                WorkItemOccurrenceEntity,
                WorkItemOccurrenceEntity.work_item_id
                == WorkItemEntity.work_item_id,
            )
            .where(WorkItemOccurrenceEntity.ops_run_id.in_(normalized))
            .distinct()
        )
        return list(rows)

    async def list_by_resource(
        self, *, resource_kind: str, resource_id: UUID
    ) -> list[WorkItemEntity]:
        self._check_active()
        rows = await self._session.scalars(
            select(WorkItemEntity)
            .join(
                WorkItemLinkEntity,
                WorkItemLinkEntity.work_item_id == WorkItemEntity.work_item_id,
            )
            .where(
                WorkItemLinkEntity.resource_kind == resource_kind,
                WorkItemLinkEntity.resource_id == resource_id,
            )
            .order_by(
                WorkItemEntity.updated_at.desc(),
                WorkItemEntity.work_item_id.desc(),
            )
        )
        return list(rows)
