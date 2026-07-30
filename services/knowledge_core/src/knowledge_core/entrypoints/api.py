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

from knowledge_core.config import get_knowledge_core_settings
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from knowledge_core.adapters.local_object_store import LocalKnowledgeObjectStore
from knowledge_core.adapters.local_parser_artifact_store import LocalParserArtifactStore
from knowledge_core.adapters.oracle_job_wakeup import (
    OracleDbmsAlertPublisher,
)
from knowledge_core.adapters.embedding import AIModelEmbeddingGateway, resolve_embedding_model
from knowledge_core.api.intake_router import router as intake_router
from knowledge_core.api.collection_router import router as collection_router
from knowledge_core.api.index_task_router import router as index_task_router
from knowledge_core.api.profile_task_router import router as profile_task_router
from knowledge_core.api.discovery_router import router as discovery_router
from knowledge_core.api.evidence_router import router as evidence_router
from knowledge_core.api.visual_router import router as visual_router
from knowledge_core.api.parse_task_router import router as parse_task_router
from knowledge_core.api.status_router import router as status_router
from knowledge_core.api.purge_task_router import router as purge_task_router
from knowledge_core.api.projection_task_router import (
    router as projection_task_router,
)
from knowledge_core.application.intake import KnowledgeCoreIntakeService
from knowledge_core.application.multipart import KnowledgeCoreMultipartOrchestrator
from knowledge_core.application.parse_tasks import KnowledgeCoreParseTaskService
from knowledge_core.application.indexing import KnowledgeCoreEvidenceIndexService
from knowledge_core.application.discovery import KnowledgeCoreProfileService
from knowledge_core.application.retrieval import KnowledgeCoreDiscoveryService
from knowledge_core.application.evidence_retrieval import KnowledgeCoreEvidenceRetrievalService
from knowledge_core.application.llm_reranking import (
    CollectionRetrievalModelResolver,
    KnowledgeCoreLlmReranker,
)
from knowledge_core.application.query_embeddings import CollectionQueryEmbeddingProvider
from knowledge_core.application.status import KnowledgeCoreStatusService
from knowledge_core.application.scope import KnowledgeCoreScopeService
from knowledge_core.application.collection_purge import KnowledgeCoreCollectionPurgeService
from knowledge_core.application.projection_tasks import (
    KnowledgeCoreProjectionTaskService,
)
from knowledge_core.application.visual_search import KnowledgeCoreVisualService
from platform_clients import AIModelClient, AIModelConfigClient
from knowledge_core.persistence import create_kc_uow
from platform_core.platform.port_check import check_port_available
from platform_core.prompts import PromptResolver, load_prompt_catalog
from platform_core.security import create_internal_auth_middleware


settings = get_knowledge_core_settings()
config = settings.api

SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化独立服务运行时，不在 API 进程内启动 KC Worker。"""
    app.state.service_name = SERVICE_NAME
    LogManager(LogConfig(
        service="knowledge_core",
        process="api",
        log_dir=settings.log.dir,
        level=settings.log.level,
        rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    db_runtime = create_database_runtime()
    app.state.db_runtime = db_runtime
    job_wakeup_publisher = (
        OracleDbmsAlertPublisher()
        if settings.job_wakeup.mode == "DBMS_ALERT"
        else None
    )
    kc_uow_factory = lambda: create_kc_uow(
        db_runtime.session_factory,
        job_wakeup_publisher=job_wakeup_publisher,
    )
    intake_service = KnowledgeCoreIntakeService(
        receipt_ttl_seconds=config.receipt_ttl_seconds,
        uow_factory=kc_uow_factory,
        parse_policy_overrides={
            "ocr_model": settings.parse_policy.ocr_model or None,
            "parse_strategy": settings.parse_policy.parse_strategy,
            "visual_description_prompt": (
                settings.parse_policy.visual_description_prompt
            ),
            "full_page_visual_prompt": (
                settings.parse_policy.full_page_visual_prompt
            ),
            "visual_min_text_characters": (
                settings.parse_policy.visual_min_text_characters
            ),
            "visual_min_mean_confidence": (
                settings.parse_policy.visual_min_mean_confidence
            ),
            "visual_max_gibberish_ratio": (
                settings.parse_policy.visual_max_gibberish_ratio
            ),
            "visual_max_concurrency": (
                settings.parse_policy.visual_max_concurrency
            ),
        },
    )
    app.state.kc_intake_service = intake_service
    app.state.kc_multipart_orchestrator = KnowledgeCoreMultipartOrchestrator(
        intake_service=intake_service,
        object_store=LocalKnowledgeObjectStore(
            Path(settings.storage.local_object_storage_path)
        ),
    )
    app.state.kc_parse_task_service = KnowledgeCoreParseTaskService(
        uow_factory=kc_uow_factory,
        artifact_store=LocalParserArtifactStore(
            Path(settings.storage.local_object_storage_path)
        ),
    )
    app.state.kc_projection_task_service = (
        KnowledgeCoreProjectionTaskService(
            uow_factory=kc_uow_factory,
        )
    )
    model_config_client = AIModelConfigClient(
        base_url=settings.embedding.base_url,
        timeout=settings.embedding.health_check_timeout_seconds,
        caller_service=SERVICE_NAME,
        audience=settings.embedding.audience,
    )
    visual_model_config_client = AIModelConfigClient(
        base_url=settings.visual.base_url,
        timeout=settings.visual.timeout_seconds,
        caller_service=SERVICE_NAME,
        audience=settings.visual.audience,
    )
    retrieval_model_config_client = AIModelConfigClient(
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout_seconds,
        caller_service=SERVICE_NAME,
        audience=settings.llm.audience,
    )
    model_client = AIModelClient(
        caller_service=SERVICE_NAME,
        embedding_config=settings.embedding,
        llm_config=settings.llm,
        visual_config=settings.visual,
    )
    app.state.kc_llm_reranker = KnowledgeCoreLlmReranker(
        model_resolver=CollectionRetrievalModelResolver(
            uow_factory=kc_uow_factory,
            model_config_client=retrieval_model_config_client,
        ),
        model_client=model_client,
        prompt_resolver=PromptResolver(
            session_factory=db_runtime.session_factory,
            catalog=load_prompt_catalog(),
        ),
    )

    async def model_resolver(collection_model_id: UUID):
        return await resolve_embedding_model(
            model_config_client,
            collection_model_id,
            expected_dimension=settings.vector.dimensions,
        )

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
    app.state.kc_visual_service = KnowledgeCoreVisualService(
        uow_factory=kc_uow_factory,
        model_config_client=visual_model_config_client,
        model_client=model_client,
    )
    app.state.kc_index_service.visual_service = app.state.kc_visual_service

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
    app.state.kc_status_service = KnowledgeCoreStatusService(
        uow_factory=kc_uow_factory,
    )
    app.state.kc_scope_service = KnowledgeCoreScopeService(
        uow_factory=kc_uow_factory,
    )
    from knowledge_core.application.collections import KnowledgeCoreBindingService, KnowledgeCoreCollectionService
    app.state.kc_collection_service = KnowledgeCoreCollectionService(
        uow_factory=kc_uow_factory,
    )
    app.state.kc_binding_service = KnowledgeCoreBindingService(
        uow_factory=kc_uow_factory,
    )
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
    docs_url="/docs" if settings.platform.debug else None,
    redoc_url="/redoc" if settings.platform.debug else None,
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
app.include_router(visual_router)
app.include_router(parse_task_router)
app.include_router(status_router)
app.include_router(purge_task_router)
app.include_router(projection_task_router)


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
        object_root = Path(settings.storage.local_object_storage_path)
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
