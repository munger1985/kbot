"""权威来源丢失只关闭平台投影，临时错误不得伪造终态。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from main_api.application.aiops_workflow_reconciliation import WorkflowSourceUnavailable
from platform_core.identity import uuid7


@pytest.mark.parametrize("missing", [True, False])
def test_reconciliation_tick_distinguishes_missing_source_from_timeout(missing):
    async def run():
        from main_api.application import notification_projection as module
        oracle = hasattr(module, "reconcile_workflow_items")
        stop = asyncio.Event()
        identity, tenant = uuid7(), uuid7()
        operation = dict(operation_id=uuid7(), domain_id=73, resource_id=str(identity), row_version=8)
        item = dict(work_item_id=uuid7(), domain_id=73, resource_id=str(identity),
            resource_type="aiops_hitl", row_version=8)
        async def resolve(request):
            if missing:
                raise WorkflowSourceUnavailable(request)
            raise TimeoutError()
        async def write(**kwargs):
            return True
        repo = SimpleNamespace()
        if oracle:
            repo.workflow_candidates = AsyncMock(return_value=[item])
            repo.workflow_operation_candidates = AsyncMock(return_value=[operation])
            repo.reconcile_workflow_item = AsyncMock(side_effect=write)
            repo.reconcile_workflow_operation = AsyncMock(side_effect=write)
        else:
            operation = dict(id=operation["operation_id"], tenant_id=tenant, source_service="aiops",
                source_operation_id=identity, row_version=8, source_version=None,
                route_params={"domain_id": str(uuid7())}, initiator_id=uuid7())
            item = dict(id=item["work_item_id"], tenant_id=tenant, source_service="aiops",
                action_type="aiops.hitl.resolve",
                resource_type="aiops_hitl", resource_id=identity, row_version=8,
                route_params={"domain_id": operation["route_params"]["domain_id"]})
            repo.list_operation_candidates = AsyncMock(return_value=[operation])
            repo.list_work_item_candidates = AsyncMock(return_value=[item])
            repo.apply_retention = AsyncMock()
            repo.update_operation_status = AsyncMock(side_effect=write)
            repo.update_work_item_resolution = AsyncMock(side_effect=write)
            repo.defer_operation_reconciliation = AsyncMock()
        active = 0
        class Unit:
            notifications = notification_projection = repo
            async def __aenter__(self):
                nonlocal active
                active += 1
                return self
            async def __aexit__(self, *_):
                nonlocal active
                active -= 1
            async def commit(self):
                if not oracle:
                    if repo.update_work_item_resolution.await_count:
                        stop.set()
        async def resolve_without_lock(request):
            assert active == 0
            return await resolve(request)
        resolver = SimpleNamespace(resolve=resolve_without_lock)
        if oracle:
            await module.reconcile_workflow_items(uow_factory=Unit, resolver=resolver)
            assert repo.reconcile_workflow_item.await_args.kwargs["status"] == ("CANCELLED" if missing else None)
            kwargs = repo.reconcile_workflow_operation.await_args.kwargs
            assert kwargs["status"] == ("EXPIRED" if missing else None)
            assert kwargs.get("source_unavailable", False) == missing
        else:
            app = SimpleNamespace(state=SimpleNamespace(workflow_reconciliation=resolver))
            await asyncio.wait_for(module.run_notification_reconciler(
                app, Unit, stop=stop, interval_seconds=0.001), timeout=2)
            if missing:
                kwargs = repo.update_operation_status.await_args.kwargs
                assert kwargs["target_status"] == "EXPIRED" and kwargs["source_unavailable"] is True
                assert repo.update_work_item_resolution.await_args.kwargs["resolution_status"] == "CANCELLED"
            else:
                repo.update_operation_status.assert_not_awaited()
                assert repo.update_work_item_resolution.await_args.kwargs["resolved"] is False
                repo.defer_operation_reconciliation.assert_awaited_once()
    asyncio.run(run())


def test_missing_source_writeback_preserves_scope_version_and_active_state():
    async def run():
        from main_api.application import notification_projection as module
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(rowcount=0)))
        if hasattr(module, "reconcile_workflow_items"):
            from main_api.repositories.notifications import NotificationRepository
            row = dict(operation_id=uuid7(), domain_id=73, resource_id=str(uuid7()), row_version=8)
            await NotificationRepository(session).reconcile_workflow_operation(
                row=row, status="EXPIRED", source_unavailable=True)
        else:
            from main_api.repositories.notification_projection import update_notification_operation_status
            assert not await update_notification_operation_status(session, operation_id=uuid7(),
                tenant_id=uuid7(), source_service="aiops", source_operation_id=uuid7(),
                target_status="EXPIRED", expected_row_version=8, source_unavailable=True)
        statement = session.execute.await_args.args[0]
        sql = str(statement).lower()
        assert "source_version is null" in sql
        assert "row_version" in sql and "status in" in sql
        assert "source_operation_id" in sql
        assert "NOTIFICATION_SOURCE_UNAVAILABLE" in statement.compile().params.values()
        assert session.execute.await_count == 1
    asyncio.run(run())


def test_statement_summary_can_be_stored_as_text_without_changing_evidence():
    from aiops_agent.application.workload import sanitize_statement_text
    raw = "select /* a\x00b */ * from t where a = 123"
    assert "\x00" not in sanitize_statement_text(raw)
    assert "\x00" in raw
