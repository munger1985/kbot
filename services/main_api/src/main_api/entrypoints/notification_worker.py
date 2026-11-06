"""Main API 通知 Outbox 投影 Worker。"""

from __future__ import annotations

import asyncio
import aiohttp
from platform_clients.case_resource import CaseResourceClient
from main_api.application.case_resource_lifecycle import CaseResourceLifecycle, run_case_resource_worker
import signal
from time import monotonic

from loguru import logger

from main_api.application import NotificationDispatcher
from main_api.application.notification_projection import reconcile_output_operations, reconcile_workflow_items
from main_api.application.aiops_workflow_reconciliation import AIOpsWorkflowReconciliation
from main_api.application.aiops_output_reconciliation import AIOpsOutputReconciliation
from main_api.application.output_authority import OutputAuthorityService
from main_api.config import get_main_api_settings
from main_api.persistence import create_main_api_uow
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_clients.aiops import AIOpsClientAuth, AIOpsManagementClient
from platform_core.security import create_auth_context_codec, create_service_identity_codec


PROCESS_NAME = "notification_worker"


async def main() -> None:
    settings = get_main_api_settings()
    config = settings.notifications
    LogManager(LogConfig(
        service="main_api",
        process=PROCESS_NAME,
        log_dir=settings.log.dir,
        level=settings.log.level,
        rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    runtime = create_database_runtime()
    uow_factory = create_main_api_uow(runtime.session_factory)
    dispatcher = NotificationDispatcher(
        uow_factory=uow_factory,
        batch_size=config.dispatcher_batch_size,
        lease_seconds=config.dispatcher_lease_seconds,
        max_attempts=config.dispatcher_max_attempts,
        external_config=settings.external_delivery,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for event_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(event_signal, stop.set)
    logger.info("通知投影 Worker 已启动")
    projection_client = AIOpsManagementClient(
        base_url=settings.aiops.base_url, timeout_seconds=min(settings.aiops.timeout_seconds, 10),
        auth=AIOpsClientAuth(caller_service=settings.api.service_name, audience=settings.aiops.audience,
            scopes=("aiops.output.reconcile", "aiops.notification.reconcile"), auth_context_codec=create_auth_context_codec(),
            service_identity_codec=create_service_identity_codec()))
    resolver = AIOpsOutputReconciliation(client=projection_client,
        authority=OutputAuthorityService(uow_factory=uow_factory),
        caller_service=settings.api.service_name)
    workflow_resolver = AIOpsWorkflowReconciliation(client=projection_client,
        caller_service=settings.api.service_name)
    case_session = aiohttp.ClientSession()
    case_lifecycle = CaseResourceLifecycle(uow_factory=uow_factory,
        client=CaseResourceClient(base_url=settings.knowledge_core.base_url,
            audience=settings.knowledge_core.audience, caller_service=settings.api.service_name,
            session=case_session))
    case_worker = asyncio.create_task(run_case_resource_worker(case_lifecycle, stop=stop))
    next_reconciliation = 0.0
    try:
        while not stop.is_set():
            processed = await dispatcher.dispatch_once()
            if monotonic() >= next_reconciliation:
                try:
                    await reconcile_output_operations(uow_factory=uow_factory, resolver=resolver)
                    await reconcile_workflow_items(uow_factory=uow_factory, resolver=workflow_resolver)
                except Exception:
                    logger.exception("AIOps 通知任务对账批次失败")
                next_reconciliation = monotonic() + config.output_reconcile_seconds
            if processed:
                continue
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=config.dispatcher_poll_seconds,
                )
            except TimeoutError:
                pass
    finally:
        stop.set()
        await case_worker
        await case_session.close()
        await projection_client.close()
        await runtime.close()
        logger.info("通知投影 Worker 已停止")


if __name__ == "__main__":
    asyncio.run(main())
