"""Domain 创建与案例资源事件共用事务；故障不能只提交一半。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from platform_core.app_resources.entity import CaseResourceEntity
from platform_core.identity import uuid7


ORACLE = CaseResourceEntity.__tablename__.startswith("KBOT_")


class Uow:
    def __init__(self):
        self.commit = AsyncMock()
        self.case_resources = SimpleNamespace(register=AsyncMock())
        self.errors = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, kind, *args):
        self.errors.append(kind)


@pytest.mark.parametrize("app_id", ["aiops", "knowledge_retrieval"])
@pytest.mark.parametrize("failure", [False, True])
def test_app_domain_creation_registers_case_event_before_commit_only_for_aiops(app_id, failure):
    async def run():
        uow = Uow()
        called = []
        async def register(**kwargs):
            assert uow.commit.await_count == 0
            called.append(kwargs)
            if failure:
                raise RuntimeError("事件登记失败")
        uow.case_resources.register.side_effect = register
        if ORACLE:
            from main_api.application.domain_management import DomainManagementService
            service = DomainManagementService(uow_factory=lambda: uow)
            service._require_app_actor = AsyncMock()
            service._bind_app_domain = AsyncMock()
            service._ensure_creator_access = AsyncMock()
            service._insert_domain = AsyncMock(return_value=SimpleNamespace(
                domain_id=73, name="ops-space", status="ACTIVE", description=None, row_version=1))
            async def create():
                return await service.create_for_app(app_id=app_id, name="ops-space",
                    description=None, actor_id="operator")
            boundary = None
        else:
            from main_api.application.resource_domain_management import ResourceDomainManagementService
            from main_api.repositories.resource_domain import ResourceDomainRepository
            domain, tenant, actor = uuid7(), uuid7(), uuid7()
            body = {"domain_id": domain, "tenant_id": tenant}
            result = SimpleNamespace(mappings=lambda: SimpleNamespace(one=lambda: body))
            session = SimpleNamespace(execute=AsyncMock(return_value=result))
            repository = ResourceDomainRepository(session)
            repository.claim_creation = AsyncMock(return_value=None)
            repository.complete_creation = AsyncMock()
            repository._audit = AsyncMock()
            repository._payload = AsyncMock(return_value=body)
            uow.resource_domains = repository
            service = ResourceDomainManagementService(uow_factory=lambda: uow,
                module_access=SimpleNamespace(is_app_available=AsyncMock(return_value=True)),
                knowledge_client=None, data_query_client=None)
            async def create():
                return await service.create(tenant_id=tenant, actor_id=actor, key="create-ops",
                    payload={"owner_app_id": app_id, "code": "ops-space", "display_name": "运维空间"})
            boundary = patch("main_api.repositories.resource_domain.CaseResourceRepository",
                return_value=uow.case_resources)
        async def verify():
            if failure and app_id == "aiops":
                with pytest.raises(RuntimeError, match="事件登记失败"):
                    await create()
                uow.commit.assert_not_awaited()
                assert uow.errors[-1] == RuntimeError
            else:
                await create()
                uow.commit.assert_awaited_once()
            assert len(called) == int(app_id == "aiops")
        if boundary is None:
            await verify()
        else:
            with boundary as factory:
                await verify()
                if app_id == "aiops":
                    factory.assert_called_once_with(session)
    asyncio.run(run())
