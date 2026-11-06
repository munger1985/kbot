"""纯计算子系统组合根；不获取 KC 会话或公共文档仓储。"""

import asyncio
import time
from loguru import logger
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from knowledge_core.adapters.bounded_parse_executor import BoundedParseExecutor
from knowledge_core.adapters.parse_job_store import ParseJobStore
from knowledge_core.api.parse_jobs_auth import compute_access
from knowledge_core.api.parse_jobs_router import create_parse_jobs_router
from knowledge_core.application.parse_jobs import ParseComputationService


def install_parse_computation(app: FastAPI, *, spool_root: Path, source_resolver,
                              memory_bytes=536870912, cpu_seconds=120, max_file_bytes=67108864,
                              max_jobs=2, max_unacked_events=32, converter_artifacts_path=None,
                              max_artifact_bytes=134217728, ocr=None):
    """认证中间件先验证 JWT；source_resolver 必须来自可信 App 注册表。"""
    service = ParseComputationService(ParseJobStore(spool_root),
        BoundedParseExecutor(memory_bytes=memory_bytes, cpu_seconds=cpu_seconds, max_file_bytes=max_file_bytes,
                            converter_artifacts_path=converter_artifacts_path, max_artifact_bytes=max_artifact_bytes, ocr=ocr),
        max_jobs=max_jobs, max_unacked_events=max_unacked_events)
    app.include_router(create_parse_jobs_router(service=service, access_dependency=compute_access,
                                               source_resolver=source_resolver))
    app.state.parse_computation_service = service
    return service


def create_parse_computation_api(*, audience, spool_root, source_registry,
                                auth_context_codec=None, service_identity_codec=None, **budgets):
    """可独立运行的受认证计算 API；不依赖旧 KC Schema 或旧解析任务线格式。"""
    from platform_core.security import (create_scoped_internal_auth_middleware,
        create_auth_context_codec, create_service_identity_codec)
    from knowledge_core.api.parse_jobs_auth import OWNERS

    @asynccontextmanager
    async def lifespan(app):
        stopped = asyncio.Event()
        async def cleanup():
            while not stopped.is_set():
                try:
                    await asyncio.to_thread(app.state.parse_computation_service.store.sweep_expired, time.time())
                except Exception as exc:
                    logger.warning("解析计算暂存回收失败：error_type={}", type(exc).__name__)
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=60)
                except TimeoutError:
                    pass
        task = asyncio.create_task(cleanup())
        try:
            yield
        finally:
            stopped.set()
            await task
            await app.state.parse_computation_service.close()

    app = FastAPI(title="App 知识解析计算", lifespan=lifespan)
    app.state.auth_context_codec = auth_context_codec or create_auth_context_codec()
    app.state.service_identity_codec = service_identity_codec or create_service_identity_codec()
    app.middleware("http")(create_scoped_internal_auth_middleware(
        audience=audience, allowed_callers={name:frozenset({"knowledge.parse.compute"}) for name in OWNERS}))

    async def resolve_source(job, request):
        return source_registry.reader(job, request.state.auth_context)

    @app.get("/healthz", tags=["System"])
    async def healthz():
        return {"status": "ok", "service": audience}

    @app.get("/readyz", tags=["System"])
    async def readyz():
        return {"status": "ok", "component": "parse-computation", "database": "NOT_APPLICABLE"}

    install_parse_computation(app, spool_root=spool_root, source_resolver=resolve_source, **budgets)
    return app
