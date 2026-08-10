"""Data Query Worker Bootstrap。"""

import asyncio

from loguru import logger

from data_query.bootstrap.common import (
    DataQueryProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from data_query.config import DataQuerySettings, get_data_query_settings
from data_query.adapters import DatabaseCredentialService, DataSourceExecutorResolver
from platform_core.managed_credentials import ManagedCredentialCipher
from data_query.connectors.schema_introspector import DatabaseSchemaIntrospector
from data_query.persistence import create_data_query_uow_factory
from data_query.workers import (
    DataQueryResultExpiryWorker,
    DataQueryWorkerService,
    SchemaSnapshotWorker,
    SemanticModelGenerationWorker,
)
from platform_core.database import create_database_runtime
from platform_clients import AIModelClient, AIModelConfigClient


def create_data_query_worker_probe(settings: DataQuerySettings | None = None):
    """创建 Data Query Worker：快照采集在后台循环，HTTP 仅暴露系统探针。"""
    resolved = settings or get_data_query_settings()
    configure_process_logging(resolved, process="worker")
    database_runtime = create_database_runtime(resolved)
    uow_factory = create_data_query_uow_factory(database_runtime.session_factory)
    credential_service = DatabaseCredentialService(
        uow_factory=uow_factory,
        cipher=ManagedCredentialCipher.from_environment(),
    )
    snapshot_worker = SchemaSnapshotWorker(
        uow_factory=uow_factory,
        credential_service=credential_service,
        introspector=DatabaseSchemaIntrospector(),
    )
    executor_resolver = DataSourceExecutorResolver(
            uow_factory=uow_factory,
            credential_service=credential_service,
    )
    query_workers = tuple(
        DataQueryWorkerService(
            uow_factory=uow_factory,
            executor_resolver=executor_resolver,
            worker_id=f"{resolved.worker.worker_id}:query:{index}",
            lease_seconds=resolved.worker.lease_seconds,
            result_availability_hours=resolved.worker.result_availability_hours,
        )
        for index in range(resolved.worker.concurrency)
    )
    semantic_model_worker = SemanticModelGenerationWorker(
        uow_factory=uow_factory,
        model_config_client=AIModelConfigClient(
            base_url=resolved.llm.base_url,
            timeout=resolved.llm.timeout_seconds,
            caller_service=resolved.worker.service_name,
            audience=resolved.llm.audience,
        ),
        model_client=AIModelClient(
            caller_service=resolved.worker.service_name,
            llm_config=resolved.llm,
        ),
        worker_id=f"{resolved.worker.worker_id}:semantic-model",
        lease_seconds=resolved.worker.lease_seconds,
    )
    expiry_worker = DataQueryResultExpiryWorker(
        uow_factory=uow_factory,
        batch_size=resolved.worker.result_expiry_batch_size,
    )
    stop_event = asyncio.Event()
    tasks: list[asyncio.Task[None]] = []

    async def start_worker() -> None:
        tasks.append(asyncio.create_task(
            snapshot_worker.run_forever(
                interval_seconds=resolved.worker.claim_interval_seconds, stop=stop_event
            ), name="data-query-schema-snapshots",
        ))
        for index, query_worker in enumerate(query_workers):
            tasks.append(asyncio.create_task(
                _run_query_worker(query_worker, resolved.worker.claim_interval_seconds, stop_event),
                name=f"data-query-runs-{index}",
            ))
        tasks.append(asyncio.create_task(
            _run_semantic_model_worker(semantic_model_worker, resolved.worker.claim_interval_seconds, stop_event),
            name="data-query-semantic-model-generation",
        ))
        tasks.append(asyncio.create_task(
            _run_result_expiry_worker(
                expiry_worker,
                resolved.worker.result_expiry_sweep_interval_seconds,
                resolved.worker.result_expiry_batch_size,
                stop_event,
            ),
            name="data-query-result-expiry",
        ))

    async def stop_worker() -> None:
        stop_event.set()
        if tasks:
            await asyncio.gather(*tasks)

    runtime = DataQueryProcessRuntime(
        settings=resolved,
        service_name=resolved.worker.service_name,
        database_runtime=database_runtime,
        uow_factory=uow_factory,
        on_start=start_worker,
        on_close=stop_worker,
    )
    return create_process_app(
        title="KBot Data Query Worker",
        description="Data Query 后台 Worker；DQ-01 仅提供系统探针。",
        runtime=runtime,
        debug=resolved.platform.debug,
    )


async def _run_query_worker(
    worker: DataQueryWorkerService, interval_seconds: float, stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        if not await worker.process_one():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except TimeoutError:
                pass


async def _run_semantic_model_worker(
    worker: SemanticModelGenerationWorker, interval_seconds: float, stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            processed = await worker.process_one()
        except Exception:
            logger.exception("语义模型生成 Worker 单次轮询失败，将继续重试")
            processed = False
        if not processed:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except TimeoutError:
                pass


async def _run_result_expiry_worker(
    worker: DataQueryResultExpiryWorker,
    interval_seconds: float,
    batch_size: int,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            purged = await worker.process_batch()
        except Exception:
            logger.exception("查询结果到期清理单次执行失败，将继续重试")
            purged = 0
        if purged < batch_size:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except TimeoutError:
                pass
