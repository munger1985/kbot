"""DBA 责任组聚合治理测试。"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest

from aiops_agent.application.responsibility_groups import ResponsibilityGroupService
from platform_core.contracts.aiops import ResponsibilityGroupPatch
from platform_core.identity import uuid7


class _Repository:
    def __init__(self, group, members):
        self.group = group
        self.members = members
        self.active = False
        self.assignments = {}

    def _check(self):
        if not self.active:
            raise RuntimeError("仓储访问发生在事务外")

    async def get_group(self, **kwargs):
        self._check()
        return self.group if kwargs["group_id"] == self.group.responsibility_group_id else None

    async def list_groups(self, **kwargs):
        self._check()
        return [self.group]

    async def list_group_members(self, **kwargs):
        self._check()
        return list(self.members)

    async def get_group_member(self, *, user_id, **kwargs):
        self._check()
        return next((row for row in self.members if row.user_id == user_id), None)

    async def add_group_member(self, entity):
        self._check()
        self.members.append(entity)
        return entity

    async def active_member_count(self, **kwargs):
        self._check()
        return sum(row.status == "ACTIVE" for row in self.members)

    async def active_assignment_count(self, *, user_id, **kwargs):
        self._check()
        return self.assignments.get(user_id, 0)

    async def group_routing_counts(self, **kwargs):
        self._check()
        return 2, 3, 4


class _Uow:
    def __init__(self, repository, now):
        self.work_items = repository
        self.runs = SimpleNamespace(database_now=lambda: asyncio.sleep(0, result=now))

    async def __aenter__(self):
        self.work_items.active = True
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.work_items.active = False

    async def commit(self):
        return None


class ResponsibilityGroupServiceTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 11, tzinfo=UTC)
        self.scope = SimpleNamespace(domain_id=7, actor_id="user-1")
        self.group = SimpleNamespace(
            responsibility_group_id=uuid7(), domain_id=7, name="生产 DBA",
            description=None, status="ACTIVE", lead_user_id="lead-1",
            row_version=1, created_at=self.now, updated_at=self.now,
            updated_by=self.scope.actor_id,
        )
        self.members = [SimpleNamespace(
            responsibility_group_id=self.group.responsibility_group_id,
            user_id="lead-1", member_role="LEAD", status="ACTIVE",
            created_at=self.now, updated_at=self.now, updated_by=self.scope.actor_id,
        )]
        self.repository = _Repository(self.group, self.members)
        self.service = ResponsibilityGroupService(
            uow_factory=lambda: _Uow(self.repository, self.now)
        )

    def test_group_detail_loads_assignment_counts_inside_transaction(self):
        result = asyncio.run(self.service.get_group(
            group_id=self.group.responsibility_group_id, scope=self.scope
        ))
        self.assertEqual(1, result.active_member_count)
        self.assertEqual(2, result.agent_binding_count)
        self.assertEqual(3, result.target_binding_count)
        self.assertEqual(4, result.unassigned_work_item_count)

    def test_new_lead_is_automatically_added_and_previous_lead_demoted(self):
        asyncio.run(self.service.patch_group(
            group_id=self.group.responsibility_group_id,
            body=ResponsibilityGroupPatch(expected_row_version=1, lead_user_id="lead-2"),
            scope=self.scope,
        ))
        self.assertEqual("MEMBER", self.members[0].member_role)
        self.assertEqual("lead-2", self.group.lead_user_id)
        self.assertEqual("LEAD", self.members[1].member_role)

    def test_member_with_active_work_items_cannot_be_removed(self):
        self.repository.assignments["lead-1"] = 2
        with self.assertRaisesRegex(Exception, "活动工作项"):
            asyncio.run(self.service.remove_member(
                group_id=self.group.responsibility_group_id,
                user_id="lead-1",
                scope=self.scope,
            ))


if __name__ == "__main__":
    unittest.main()
