"""持久化案例资源事件的后台执行与业务对账；不检查数据库 Schema。"""

import asyncio
from time import monotonic

from loguru import logger
from platform_clients.case_resource import CaseResourceClientError

from platform_core.contracts.aiops.case_resource import CaseResourceRequest, CaseResourceStatus


class CaseResourceLifecycle:
    def __init__(self, *, uow_factory, client):
        self._uow_factory, self._client = uow_factory, client
        self._cursor = None

    async def status(self, *, domain_id, tenant_id=None):
        async with self._uow_factory() as uow:
            row = await uow.case_resources.get(domain_id=domain_id, tenant_id=tenant_id)
            if row is None:
                return CaseResourceStatus(resource_status="UNINITIALIZED")
            scope = await uow.case_resources.scope(domain_id=domain_id, tenant_id=tenant_id)
            state = row.status if scope and scope["eligible"] else "PAUSED"
            return CaseResourceStatus(resource_status=state, collection_id=row.collection_id,
                configuration_status=row.configuration_status,
                missing_model_roles=row.missing_model_roles or [], error_code=row.error_code)

    async def discover(self):
        # 每批保存键集游标，避免大空间集合一直只处理第一页。
        async with self._uow_factory() as uow:
            rows = await uow.case_resources.discover(after=self._cursor, limit=100)
        for scope in rows:
            try:
                async with self._uow_factory() as uow:
                    await uow.case_resources.register(**scope)
                    await uow.commit()
            except Exception:
                # 单个历史空间的数据问题不能阻断其他空间；下一轮仍会重新发现。
                logger.warning("案例资源对账事件登记失败，其他空间继续处理")
        self._cursor = rows[-1]["domain_id"] if len(rows) == 100 else None

    async def tick(self):
        async with self._uow_factory() as uow:
            item = await uow.case_resources.claim()
            await uow.commit()
        if item is None:
            return False
        result, error, retryable = None, None, True
        try:
            request = CaseResourceRequest(tenant_id=item["tenant_id"],
                domain_id=str(item["domain_id"]), actor_id=item["actor_id"],
                expected_collection_id=item["collection_id"], allow_create=item["allow_create"])
            result = await self._client.ensure(request)
        except ValueError:
            error, retryable = "CASE_RESOURCE_REQUEST_INVALID", False
        except Exception as exc:
            # 不持久化网络响应、凭证或驱动异常文本。
            if isinstance(exc, CaseResourceClientError):
                error, retryable = exc.code, exc.retryable
            else:
                error = "CASE_RESOURCE_UNAVAILABLE"
        async with self._uow_factory() as uow:
            await uow.case_resources.complete(item, result=result, error=error, retryable=retryable)
            await uow.commit()
        return True


async def run_case_resource_worker(lifecycle, *, stop):
    next_discovery = 0.0
    while not stop.is_set():
        try:
            if monotonic() >= next_discovery:
                await lifecycle.discover()
                next_discovery = monotonic() + 30
            processed = await lifecycle.tick()
        except Exception as exc:
            # 正常仓储访问失败由运维日志暴露；不改用运行时 DDL 或 Schema 探测。
            logger.warning("案例资源后台批次失败（{}），将在后续轮询重试", type(exc).__name__)
            processed = False
        if processed:
            continue
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
        except TimeoutError:
            pass
