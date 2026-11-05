"""通知待办权威对账的事务边界、失败轮转和 CAS 回归。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from main_api.application.notification_projection import reconcile_workflow_items
from main_api.repositories.notifications import NotificationRepository
from platform_core.contracts.aiops.notification_projection import WorkflowProjectionSnapshot
from platform_core.identity import uuid7


def test_authoritative_read_occurs_outside_uow_and_failed_read_never_closes_item():
    async def run():
        row = dict(work_item_id=uuid7(), domain_id=73, resource_type="aiops_hitl",
            resource_id=str(uuid7()), row_version=8)
        active = 0
        repository = SimpleNamespace(workflow_candidates=AsyncMock(return_value=[row]),
            workflow_operation_candidates=AsyncMock(return_value=[]),
            reconcile_workflow_item=AsyncMock(return_value=True))
        class Uow:
            notifications = repository
            async def __aenter__(self):
                nonlocal active
                active += 1
                return self
            async def __aexit__(self, *args):
                nonlocal active
                active -= 1
            async def commit(self):
                pass
        class Resolver:
            failed = False
            async def resolve(self, request):
                assert active == 0
                if self.failed:
                    raise TimeoutError()
                return WorkflowProjectionSnapshot(request=request, source_status="ANSWERED",
                    source_version=3, work_item_status="COMPLETED")
        resolver = Resolver()
        assert await reconcile_workflow_items(uow_factory=Uow, resolver=resolver) == 1
        repository.reconcile_workflow_item.assert_awaited_with(row=row, status="COMPLETED")
        resolver.failed = True
        assert await reconcile_workflow_items(uow_factory=Uow, resolver=resolver) == 0
        repository.reconcile_workflow_item.assert_awaited_with(row=row, status=None)
    asyncio.run(run())


def test_writeback_is_guarded_by_id_domain_resource_observed_version_and_open_state():
    async def run():
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(rowcount=0)))
        row = dict(work_item_id=uuid7(), domain_id=73, resource_type="aiops_hitl",
            resource_id=str(uuid7()), row_version=8)
        assert not await NotificationRepository(session).reconcile_workflow_item(row=row, status="COMPLETED")
        statement = session.execute.await_args.args[0]
        sql = str(statement)
        for column in ("work_item_id", "domain_id", "resource_id", "resource_type", "row_version", "status"):
            assert column.upper() in sql.upper()
        parameters = statement.compile().params.values()
        for value in (*row.values(), "OPEN", "COMPLETED"):
            assert value in parameters
    asyncio.run(run())


def test_approver_resolution_excludes_applicant_disabled_and_out_of_scope_members():
    from main_api.repositories.access_control import AccessControlRepository
    async def run():
        repository = object.__new__(AccessControlRepository)
        repository.list_app_members = AsyncMock(return_value=[
            SimpleNamespace(user_id=value, status="DISABLED" if value == "disabled" else "ACTIVE")
            for value in ("applicant", "reviewer", "disabled", "other-domain")])
        repository.get_user = AsyncMock(return_value=SimpleNamespace(status="ACTIVE"))
        repository.list_roles = AsyncMock(return_value=())
        repository.permissions_for = AsyncMock(side_effect=lambda **values:
            {"aiops:use", "aiops:proposal:approve"} if values["user_id"] == "reviewer" else set())
        result = await repository.notification_approvers(domain_id=73, exclude_user_id="applicant")
        assert result == {"reviewer"}
        assert all(call.kwargs["domain_id"] == 73 for call in repository.permissions_for.await_args_list)
    asyncio.run(run())
