"""Main API Domain 生命周期应用服务测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from main_api.application import (
    DomainConflictError,
    DomainLifecycleError,
    DomainManagementService,
)


class _DomainRepository:
    def __init__(self):
        self.rows = []

    async def get_by_name(self, *, name):
        return next((row for row in self.rows if row.name == name), None)

    async def get(self, *, domain_id):
        return next(
            (row for row in self.rows if int(row.domain_id) == int(domain_id)),
            None,
        )

    async def list_by_ids(self, *, domain_ids):
        wanted = {int(value) for value in domain_ids}
        return [row for row in self.rows if int(row.domain_id) in wanted]

    async def add(self, entity):
        if getattr(entity, "domain_id", None) is None:
            next_id = max((int(row.domain_id) for row in self.rows), default=0) + 1
            entity.domain_id = next_id
        self.rows.append(entity)
        return entity


class _AccessRepository:
    def __init__(self):
        self.users = {
            "ui-tester": SimpleNamespace(
                user_id="ui-tester", status="ACTIVE",
                account_origin="PLATFORM",
            )
        }
        self.roles = []
        self.applications = {}
        self.members = {}
        self.app_domains = {}
        self.scopes = {}

    async def get_user(self, user_id):
        return self.users.get(user_id)

    async def add_user(self, entity):
        self.users[entity.user_id] = entity
        return entity

    async def upsert_member_role(self, **values):
        self.roles.append(values)

    async def get_application(self, app_id):
        return self.applications.get(app_id)

    async def get_app_member(self, *, app_id, user_id):
        return self.members.get((app_id, user_id))

    async def list_member_roles(self, *, app_id, domain_id=None):
        del domain_id
        return [row for row in self.roles if row.app_id == app_id]

    async def list_app_domains(self, *, app_id):
        return [
            row for (stored_app, _domain_id), row in self.app_domains.items()
            if stored_app == app_id
        ]

    async def get_app_domain(self, *, app_id, domain_id):
        return self.app_domains.get((app_id, int(domain_id)))

    async def add_app_domain(self, row):
        self.app_domains[(row.app_id, int(row.domain_id))] = row

    async def list_member_role_scopes(self, *, app_id, user_id, role_code):
        return tuple(self.scopes.get((app_id, user_id, role_code), ()))

    async def replace_member_role_scopes(self, *, app_id, user_id, role_code, domain_ids):
        self.scopes[(app_id, user_id, role_code)] = tuple(int(value) for value in domain_ids)


class _Uow:
    def __init__(self, repository, access=None):
        self.domains = repository
        self.access = access if access is not None else _AccessRepository()
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self):
        self.committed = True


def _app_access(*, user_id="assistantadmin", scope_mode="ALL_APP_DOMAINS", scopes=()):
    access = _AccessRepository()
    access.users[user_id] = SimpleNamespace(
        user_id=user_id, status="ACTIVE", account_origin="APP",
    )
    access.applications["assistant"] = SimpleNamespace(app_id="assistant", status="ACTIVE")
    access.members[("assistant", user_id)] = SimpleNamespace(
        app_id="assistant", user_id=user_id, status="ACTIVE",
    )
    access.roles.append(SimpleNamespace(
        app_id="assistant", user_id=user_id, role_code="ADMIN",
        scope_mode=scope_mode, status="ACTIVE",
    ))
    if scopes:
        access.scopes[("assistant", user_id, "ADMIN")] = tuple(scopes)
    return access


def _seed_domain(repository, *, domain_id, name, status="ACTIVE", row_version=1):
    row = SimpleNamespace(
        domain_id=domain_id,
        name=name,
        status=status,
        description="测试 Domain",
        row_version=row_version,
        created_by="init",
        updated_by="init",
        created_at=None,
        updated_at=None,
    )
    repository.rows.append(row)
    return row


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

    async def test_domain_api_cannot_bootstrap_reserved_admin(self):
        repository = _DomainRepository()
        service = DomainManagementService(
            uow_factory=lambda: _Uow(repository),
        )

        with self.assertRaisesRegex(
            DomainConflictError, "只能通过项目初始化脚本创建"
        ):
            await service.create(
                name="非法初始化域",
                description=None,
                actor_id="admin",
            )

    async def test_app_user_with_all_domains_binds_without_writing_scope(self):
        repository = _DomainRepository()
        access = _app_access(scope_mode="ALL_APP_DOMAINS")
        uow = _Uow(repository, access)
        service = DomainManagementService(uow_factory=lambda: uow)

        result = await service.create_for_app(
            app_id="assistant",
            name="华东销售",
            description="隔离范围",
            actor_id="assistantadmin",
        )

        self.assertEqual(1, result["domain_id"])
        self.assertEqual("ACTIVE", result["status"])
        self.assertIn(("assistant", 1), access.app_domains)
        self.assertEqual({}, access.scopes)
        self.assertTrue(uow.committed)

    async def test_selected_domains_appends_created_domain_to_scope(self):
        repository = _DomainRepository()
        _seed_domain(repository, domain_id=1, name="已有域")
        access = _app_access(scope_mode="SELECTED_DOMAINS", scopes=(1,))
        access.app_domains[("assistant", 1)] = SimpleNamespace(
            app_id="assistant", domain_id=1, status="ACTIVE",
        )
        uow = _Uow(repository, access)
        service = DomainManagementService(uow_factory=lambda: uow)

        result = await service.create_for_app(
            app_id="assistant",
            name="新建销售域",
            description=None,
            actor_id="assistantadmin",
        )

        self.assertEqual(2, result["domain_id"])
        self.assertEqual((1, 2), access.scopes[("assistant", "assistantadmin", "ADMIN")])

    async def test_list_for_app_includes_disabled_authorized_domains(self):
        repository = _DomainRepository()
        _seed_domain(repository, domain_id=3, name="停用域", status="DISABLED")
        access = _app_access(scope_mode="ALL_APP_DOMAINS")
        access.app_domains[("assistant", 3)] = SimpleNamespace(
            app_id="assistant", domain_id=3, status="DISABLED",
        )
        service = DomainManagementService(uow_factory=lambda: _Uow(repository, access))

        result = await service.list_for_app(app_id="assistant", user_id="assistantadmin")

        self.assertEqual(1, len(result["items"]))
        self.assertEqual("DISABLED", result["items"][0]["status"])
        self.assertEqual(3, result["items"][0]["domain_id"])

    async def test_update_for_app_renames_and_increments_version(self):
        repository = _DomainRepository()
        _seed_domain(repository, domain_id=8, name="旧名称")
        access = _app_access(scope_mode="ALL_APP_DOMAINS")
        access.app_domains[("assistant", 8)] = SimpleNamespace(
            app_id="assistant", domain_id=8, status="ACTIVE",
        )
        uow = _Uow(repository, access)
        service = DomainManagementService(uow_factory=lambda: uow)

        result = await service.update_for_app(
            app_id="assistant",
            domain_id=8,
            user_id="assistantadmin",
            actor_id="assistantadmin",
            expected_row_version=1,
            name="新名称",
            description="更新说明",
            description_set=True,
        )

        self.assertEqual("新名称", result["name"])
        self.assertEqual("更新说明", result["description"])
        self.assertEqual(2, result["row_version"])
        self.assertTrue(uow.committed)

    async def test_disable_for_app_syncs_app_domain_status(self):
        repository = _DomainRepository()
        _seed_domain(repository, domain_id=8, name="可停用域")
        access = _app_access(scope_mode="ALL_APP_DOMAINS")
        access.app_domains[("assistant", 8)] = SimpleNamespace(
            app_id="assistant", domain_id=8, status="ACTIVE",
        )
        service = DomainManagementService(uow_factory=lambda: _Uow(repository, access))

        result = await service.disable_for_app(
            app_id="assistant",
            domain_id=8,
            user_id="assistantadmin",
            actor_id="assistantadmin",
            expected_row_version=1,
        )

        self.assertEqual("DISABLED", result["status"])
        self.assertEqual("DISABLED", access.app_domains[("assistant", 8)].status)
        self.assertEqual(2, result["row_version"])

    async def test_bootstrap_portal_cannot_be_renamed_or_disabled(self):
        repository = _DomainRepository()
        _seed_domain(repository, domain_id=41, name="assistant_portal")
        access = _app_access(scope_mode="ALL_APP_DOMAINS")
        access.app_domains[("assistant", 41)] = SimpleNamespace(
            app_id="assistant", domain_id=41, status="ACTIVE",
        )
        service = DomainManagementService(uow_factory=lambda: _Uow(repository, access))

        with self.assertRaises(DomainLifecycleError) as renamed:
            await service.update_for_app(
                app_id="assistant",
                domain_id=41,
                user_id="assistantadmin",
                actor_id="assistantadmin",
                expected_row_version=1,
                name="改名引导域",
            )
        self.assertEqual("DOMAIN_BOOTSTRAP_PROTECTED", renamed.exception.code)

        with self.assertRaises(DomainLifecycleError) as disabled:
            await service.disable_for_app(
                app_id="assistant",
                domain_id=41,
                user_id="assistantadmin",
                actor_id="assistantadmin",
                expected_row_version=1,
            )
        self.assertEqual("DOMAIN_BOOTSTRAP_PROTECTED", disabled.exception.code)
        self.assertEqual("ACTIVE", repository.rows[0].status)


if __name__ == "__main__":
    unittest.main()
