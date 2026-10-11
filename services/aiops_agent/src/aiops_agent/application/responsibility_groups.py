"""DBA 责任组治理用例。"""

from datetime import UTC, datetime
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, row_version_changed, state_conflict
from aiops_agent.entities import ResponsibilityGroupEntity, ResponsibilityGroupMemberEntity
from platform_core.contracts.aiops import (
    ResponsibilityGroupCreate,
    ResponsibilityGroupMemberUpsert,
    ResponsibilityGroupMemberView,
    ResponsibilityGroupPage,
    ResponsibilityGroupPatch,
    ResponsibilityGroupSummary,
    ResponsibilityGroupView,
)
from platform_core.identity import uuid7


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class ResponsibilityGroupService:
    """维护域内责任组；成员资格由 Main API 在调用前依据 IAM 复核。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def list_groups(self, *, scope) -> ResponsibilityGroupPage:
        async with self._uow_factory() as uow:
            rows = await uow.work_items.list_groups(domain_id=scope.domain_id)
            items = [await self._summary(uow, row) for row in rows]
        return ResponsibilityGroupPage(items=tuple(items), has_more=False)

    async def get_group(self, *, group_id: UUID, scope) -> ResponsibilityGroupView:
        async with self._uow_factory() as uow:
            group = await uow.work_items.get_group(group_id=group_id, domain_id=scope.domain_id)
            if group is None:
                raise resource_not_found("责任组")
            members = await uow.work_items.list_group_members(group_id=group_id)
            summary = await self._summary(uow, group)
            member_views = []
            for row in members:
                member_views.append(ResponsibilityGroupMemberView(
                    user_id=row.user_id,
                    member_role=row.member_role,
                    status=row.status,
                    active_work_item_count=await uow.work_items.active_assignment_count(
                        group_id=group_id,
                        user_id=row.user_id,
                    ),
                    created_at=_utc(row.created_at),
                ))
        return ResponsibilityGroupView(
            **summary.model_dump(),
            members=tuple(member_views),
        )

    async def create_group(self, *, body: ResponsibilityGroupCreate, scope) -> ResponsibilityGroupView:
        async with self._uow_factory() as uow:
            now = _utc(await uow.runs.database_now())
            group_id = uuid7()
            await uow.work_items.add_group(ResponsibilityGroupEntity(
                responsibility_group_id=group_id, domain_id=scope.domain_id, name=body.name,
                description=body.description, status="ACTIVE", lead_user_id=body.lead_user_id,
                created_by=scope.actor_id, updated_by=scope.actor_id, created_at=now, updated_at=now, row_version=1,
            ))
            if body.lead_user_id is not None:
                await uow.work_items.add_group_member(
                    ResponsibilityGroupMemberEntity(
                        responsibility_group_id=group_id,
                        user_id=body.lead_user_id,
                        member_role="LEAD",
                        status="ACTIVE",
                        created_by=scope.actor_id,
                        updated_by=scope.actor_id,
                        created_at=now,
                        updated_at=now,
                    )
                )
            await uow.commit()
        return await self.get_group(group_id=group_id, scope=scope)

    async def patch_group(self, *, group_id: UUID, body: ResponsibilityGroupPatch, scope) -> ResponsibilityGroupView:
        async with self._uow_factory() as uow:
            group = await uow.work_items.get_group(group_id=group_id, domain_id=scope.domain_id, lock=True)
            if group is None:
                raise resource_not_found("责任组")
            if int(group.row_version) != body.expected_row_version:
                raise row_version_changed()
            now = _utc(await uow.runs.database_now())
            if "lead_user_id" in body.model_fields_set:
                members = await uow.work_items.list_group_members(group_id=group_id)
                for row in members:
                    if row.status == "ACTIVE" and row.member_role == "LEAD":
                        row.member_role = "MEMBER"
                        row.updated_by, row.updated_at = scope.actor_id, now
                if body.lead_user_id is not None:
                    member = next(
                        (row for row in members if row.user_id == body.lead_user_id),
                        None,
                    )
                    if member is None:
                        await uow.work_items.add_group_member(
                            ResponsibilityGroupMemberEntity(
                                responsibility_group_id=group_id,
                                user_id=body.lead_user_id,
                                member_role="LEAD",
                                status="ACTIVE",
                                created_by=scope.actor_id,
                                updated_by=scope.actor_id,
                                created_at=now,
                                updated_at=now,
                            )
                        )
                    else:
                        member.member_role, member.status = "LEAD", "ACTIVE"
                        member.updated_by, member.updated_at = scope.actor_id, now
            for field in ("name", "description", "status", "lead_user_id"):
                if field in body.model_fields_set:
                    setattr(group, field, getattr(body, field))
            group.updated_by, group.updated_at = scope.actor_id, now
            await uow.commit()
        return await self.get_group(group_id=group_id, scope=scope)

    async def put_member(self, *, group_id: UUID, user_id: str, body: ResponsibilityGroupMemberUpsert, scope) -> ResponsibilityGroupView:
        async with self._uow_factory() as uow:
            group = await uow.work_items.get_group(group_id=group_id, domain_id=scope.domain_id)
            if group is None or group.status != "ACTIVE":
                raise state_conflict("只能向有效责任组添加成员")
            now = _utc(await uow.runs.database_now())
            member = await uow.work_items.get_group_member(group_id=group_id, user_id=user_id)
            if member is None:
                await uow.work_items.add_group_member(ResponsibilityGroupMemberEntity(
                    responsibility_group_id=group_id, user_id=user_id, member_role=body.member_role,
                    status="ACTIVE", created_by=scope.actor_id, updated_by=scope.actor_id, created_at=now, updated_at=now,
                ))
            else:
                member.member_role, member.status = body.member_role, "ACTIVE"
                member.updated_by, member.updated_at = scope.actor_id, now
            if body.member_role == "LEAD":
                for row in await uow.work_items.list_group_members(group_id=group_id):
                    if row.user_id != user_id and row.status == "ACTIVE" and row.member_role == "LEAD":
                        row.member_role = "MEMBER"
                        row.updated_by, row.updated_at = scope.actor_id, now
                group.lead_user_id = user_id
                group.updated_by, group.updated_at = scope.actor_id, now
            elif group.lead_user_id == user_id:
                group.lead_user_id = None
            group.updated_by, group.updated_at = scope.actor_id, now
            await uow.commit()
        return await self.get_group(group_id=group_id, scope=scope)

    async def remove_member(self, *, group_id: UUID, user_id: str, scope) -> ResponsibilityGroupView:
        async with self._uow_factory() as uow:
            group = await uow.work_items.get_group(
                group_id=group_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if group is None:
                raise resource_not_found("责任组")
            member = await uow.work_items.get_group_member(group_id=group_id, user_id=user_id)
            if member is None:
                raise resource_not_found("责任组成员")
            active_count = await uow.work_items.active_assignment_count(
                group_id=group_id,
                user_id=user_id,
            )
            if active_count:
                raise state_conflict(
                    f"该成员仍有 {active_count} 个活动工作项，请先改派或退回责任组队列"
                )
            member.status, member.updated_by = "INACTIVE", scope.actor_id
            member.updated_at = _utc(await uow.runs.database_now())
            if group.lead_user_id == user_id:
                group.lead_user_id = None
            group.updated_by, group.updated_at = scope.actor_id, member.updated_at
            await uow.commit()
        return await self.get_group(group_id=group_id, scope=scope)

    async def _summary(self, uow, group) -> ResponsibilityGroupSummary:
        agent_count, target_count, unassigned_count = (
            await uow.work_items.group_routing_counts(
                group_id=group.responsibility_group_id
            )
        )
        return ResponsibilityGroupSummary(
            responsibility_group_id=group.responsibility_group_id, name=group.name,
            description=group.description, status=group.status, lead_user_id=group.lead_user_id,
            active_member_count=await uow.work_items.active_member_count(group_id=group.responsibility_group_id),
            agent_binding_count=agent_count,
            target_binding_count=target_count,
            unassigned_work_item_count=unassigned_count,
            row_version=int(group.row_version), created_at=_utc(group.created_at), updated_at=_utc(group.updated_at),
        )
