"""Knowledge Core 微服务入口。

服务独占 KC 持久化能力，只暴露内部契约。Parser、Portal 和检索客户端通过
HTTP 访问，不能获取 KC 数据库会话或直接访问 KC 表。
"""

import os
import sys
from contextlib import asynccontextmanager
from uuid import UUID
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger
from sqlalchemy import text

from platform_core.config.settings import get_app_config, get_embed_config, get_knowledge_core_config
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from knowledge_core.adapters.local_object_store import LocalKnowledgeObjectStore
from knowledge_core.adapters.local_parser_artifact_store import LocalParserArtifactStore
from knowledge_core.adapters.embedding import AIModelEmbeddingGateway, resolve_embedding_model
from knowledge_core.api.intake_router import router as intake_router
from knowledge_core.api.collection_router import router as collection_router
from knowledge_core.api.index_task_router import router as index_task_router
from knowledge_core.api.profile_task_router import router as profile_task_router
from knowledge_core.api.discovery_router import router as discovery_router
from knowledge_core.api.evidence_router import router as evidence_router
from knowledge_core.api.parse_task_router import router as parse_task_router
from knowledge_core.api.status_router import router as status_router
from knowledge_core.api.purge_task_router import router as purge_task_router
from knowledge_core.application.intake import KnowledgeCoreIntakeService
from knowledge_core.application.multipart import KnowledgeCoreMultipartOrchestrator
from knowledge_core.application.parse_tasks import KnowledgeCoreParseTaskService
from knowledge_core.application.indexing import KnowledgeCoreEvidenceIndexService
from knowledge_core.application.discovery import KnowledgeCoreProfileService
from knowledge_core.application.retrieval import KnowledgeCoreDiscoveryService
from knowledge_core.application.evidence_retrieval import KnowledgeCoreEvidenceRetrievalService
from knowledge_core.application.query_embeddings import CollectionQueryEmbeddingProvider
from knowledge_core.application.status import KnowledgeCoreStatusService
from knowledge_core.application.scope import KnowledgeCoreScopeService
from knowledge_core.application.collection_purge import KnowledgeCoreCollectionPurgeService
from platform_clients import AIModelClient, AIModelConfigClient
from knowledge_core.persistence import create_kc_uow
from platform_core.platform.port_check import check_port_available
from platform_core.security import create_internal_auth_middleware


config = get_knowledge_core_config()
app_config = get_app_config()

SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化独立服务运行时，不在 API 进程内启动 KC Worker。"""
    app.state.service_name = SERVICE_NAME
    LogManager(LogConfig(
        service_name=SERVICE_NAME,
        log_dir=app_config.log.dir,
        level=app_config.log.level,
        rotation=app_config.log.rotation,
        retention=app_config.log.retention,
    )).setup()
    db_runtime = create_database_runtime()
    app.state.db_runtime = db_runtime
    kc_uow_factory = lambda: create_kc_uow(db_runtime.session_factory)
    intake_service = KnowledgeCoreIntakeService(
        app_id=app_config.app_id,
        receipt_ttl_seconds=config.receipt_ttl_seconds,
        uow_factory=kc_uow_factory,
        parse_policy_overrides={
            "vlm_model": config.parser_vlm_model or None,
            "visual_description_prompt": config.parser_visual_description_prompt,
        },
    )
    app.state.kc_multipart_orchestrator = KnowledgeCoreMultipartOrchestrator(
        intake_service=intake_service,
        object_store=LocalKnowledgeObjectStore(Path(config.local_object_storage_path)),
    )
    app.state.kc_parse_task_service = KnowledgeCoreParseTaskService(
        uow_factory=kc_uow_factory,
        artifact_store=LocalParserArtifactStore(Path(config.local_object_storage_path)),
    )
    model_config_client = AIModelConfigClient(
        base_url=config.embedding_service_url or get_embed_config().service_url,
        caller_service=SERVICE_NAME,
    )
    model_client = AIModelClient(caller_service=SERVICE_NAME)

    async def model_resolver(collection_model_id: int):
        return await resolve_embedding_model(model_config_client, collection_model_id)

    app.state.kc_index_service = KnowledgeCoreEvidenceIndexService(
        uow_factory=kc_uow_factory,
        embedding_gateway=AIModelEmbeddingGateway(client=model_client),
        model_resolver=model_resolver,
    )
    app.state.kc_profile_service = KnowledgeCoreProfileService(uow_factory=kc_uow_factory)
    app.state.kc_query_embedding_provider = CollectionQueryEmbeddingProvider(
        uow_factory=kc_uow_factory,
        embedding_gateway=AIModelEmbeddingGateway(client=model_client),
        model_resolver=model_resolver,
    )

    class UowDiscoverySearchPort:
        async def search_text(self, *, collection_id: UUID, query: str, limit: int, max_security_level: int):
            async with kc_uow_factory() as uow:
                return await uow.discovery.search_text(collection_id=collection_id, query=query, limit=limit, max_security_level=max_security_level)

        async def search_vector(self, *, collection_id: UUID, vector: list[float], limit: int, max_security_level: int):
            async with kc_uow_factory() as uow:
                return await uow.discovery.search_vector(collection_id=collection_id, vector=vector, limit=limit, max_security_level=max_security_level)

    app.state.kc_discovery_service = KnowledgeCoreDiscoveryService(
        search_port=UowDiscoverySearchPort(),
        query_embedding_provider=app.state.kc_query_embedding_provider,
    )

    class UowEvidenceSearchPort:
        async def search_text(self, *, scope, query: str, limit: int, max_security_level: int):
            async with kc_uow_factory() as uow:
                return await uow.evidence.search_text(scope=scope, query=query, limit=limit, max_security_level=max_security_level)

        async def search_vector(self, *, scope, vector: list[float], limit: int, max_security_level: int):
            async with kc_uow_factory() as uow:
                return await uow.evidence.search_vector(scope=scope, vector=vector, limit=limit, max_security_level=max_security_level)

        async def expand_context(self, *, anchors, limit: int):
            async with kc_uow_factory() as uow:
                return await uow.evidence.expand_context(anchors=list(anchors), limit=limit)

    app.state.kc_evidence_service = KnowledgeCoreEvidenceRetrievalService(
        search_port=UowEvidenceSearchPort(),
        query_embedding_provider=app.state.kc_query_embedding_provider,
    )
    app.state.kc_status_service = KnowledgeCoreStatusService(app_id=app_config.app_id, uow_factory=kc_uow_factory)
    app.state.kc_scope_service = KnowledgeCoreScopeService(app_id=app_config.app_id, uow_factory=kc_uow_factory)
    from knowledge_core.application.collections import KnowledgeCoreBindingService, KnowledgeCoreCollectionService
    app.state.kc_collection_service = KnowledgeCoreCollectionService(app_id=app_config.app_id, uow_factory=kc_uow_factory)
    app.state.kc_binding_service = KnowledgeCoreBindingService(app_id=app_config.app_id, uow_factory=kc_uow_factory)
    app.state.kc_purge_service = KnowledgeCoreCollectionPurgeService(uow_factory=kc_uow_factory)
    logger.info("正在启动服务 [{}]，进程号={}", SERVICE_NAME, os.getpid())
    try:
        yield
    finally:
        await db_runtime.close()
        logger.info("正在停止服务 [{}]", SERVICE_NAME)


app = FastAPIOffline(
    title=f"{SERVICE_NAME} API",
    description="Knowledge Core 的内部入库、解析与检索服务边界。",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if app_config.debug else None,
    redoc_url="/redoc" if app_config.debug else None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(log_requests)
app.middleware("http")(
    create_internal_auth_middleware(audience=SERVICE_NAME)
)
app.include_router(intake_router)
app.include_router(collection_router)
app.include_router(index_task_router)
app.include_router(profile_task_router)
app.include_router(discovery_router)
app.include_router(evidence_router)
app.include_router(parse_task_router)
app.include_router(status_router)
app.include_router(purge_task_router)


@app.get("/health", tags=["System"], summary="存活检查")
async def health() -> dict[str, Any]:
    """存活检查不访问数据库。"""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/healthz", tags=["System"], summary="存活检查")
async def healthz() -> dict[str, Any]:
    return await health()


@app.get("/readyz", tags=["System"], summary="就绪检查")
async def readyz(request: Request) -> dict[str, Any]:
    checks: dict[str, str] = {}
    try:
        async with request.app.state.db_runtime.session_factory() as session:
            await session.execute(text("SELECT 1 FROM DUAL"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = type(exc).__name__
    try:
        object_root = Path(config.local_object_storage_path)
        object_root.mkdir(parents=True, exist_ok=True)
        checks["object_store"] = "ok" if object_root.is_dir() else "unavailable"
    except Exception as exc:
        checks["object_store"] = type(exc).__name__
    ready = all(value == "ok" for value in checks.values())
    payload = {"status": "ready" if ready else "not_ready", "service": SERVICE_NAME, "checks": checks}
    if not ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


if __name__ == "__main__":
    if not check_port_available(SERVICE_HOST, SERVICE_PORT, SERVICE_NAME):
        sys.exit(1)
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_config=None, loop="asyncio")
