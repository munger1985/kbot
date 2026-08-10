"""Main API Domain 生命周期应用服务测试。"""

from __future__ import annotations

import unittest

from main_api.application import DomainConflictError, DomainManagementService


class _DomainRepository:
    def __init__(self):
        self.rows = []

    async def get_by_name(self, *, name):
        return next(
            (
                row
                for row in self.rows
                if row.name == name
            ),
            None,
        )

    async def add(self, entity):
        entity.domain_id = len(self.rows) + 1
        self.rows.append(entity)
        return entity


class _Uow:
    def __init__(self, repository):
        self.domains = repository
        self.access = _AccessRepository()
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self):
        self.committed = True


class _AccessRepository:
    def __init__(self):
        self.users = {}
        self.roles = []

    async def get_user(self, user_id):
        return self.users.get(user_id)

    async def add_user(self, entity):
        self.users[entity.user_id] = entity
        return entity

    async def upsert_member_role(self, **values):
        self.roles.append(values)


class DomainManagementServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_assigns_platform_scope_and_actor(self):
        repository = _DomainRepository()
        uow = _Uow(repository)
        service = DomainManagementService(
            uow_factory=lambda: uow,
        )

        result = await service.create(
            name="研发知识域",
            description="测试 Domain",
            actor_id="ui-tester",
        )

        self.assertEqual(1, result["domain_id"])
        self.assertEqual("ACTIVE", result["status"])
        self.assertEqual("ui-tester", repository.rows[0].created_by)
        self.assertTrue(uow.committed)

    async def test_duplicate_name_is_rejected(self):
        repository = _DomainRepository()
        service = DomainManagementService(
            uow_factory=lambda: _Uow(repository),
        )
        await service.create(
            name="研发知识域",
            description=None,
            actor_id="ui-tester",
        )

        with self.assertRaises(DomainConflictError):
            await service.create(
                name="研发知识域",
                description=None,
                actor_id="ui-tester",
            )


if __name__ == "__main__":
    unittest.main()
