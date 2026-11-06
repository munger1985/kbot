"""App 知识摄取组合根；场景用例复用基础模块并固定自有存储。"""

from pathlib import Path
from knowledge_retrieval_app.adapters.knowledge_store import LocalKnowledgeStore
from knowledge_retrieval_app.adapters.parse_computation import HttpParseComputation
from knowledge_retrieval_app.api.knowledge_source import create_knowledge_source_router
from knowledge_retrieval_app.api.knowledge_source_auth import source_scope
from knowledge_retrieval_app.application.knowledge_ingestion import KnowledgeIngestionService
from knowledge_retrieval_app.application.knowledge_projection import KnowledgeProjectionService


def install_app_knowledge(app, *, uow_factory, store_root: Path, compute_client, context_factory, register_routes=True):
    """context_factory 由已验证任务委托构造上下文，不接受请求内 Python 回调。"""
    service = KnowledgeIngestionService(uow_factory=uow_factory, object_store=LocalKnowledgeStore(store_root),
                                       compute=HttpParseComputation(compute_client, context_factory))
    app.state.knowledge_ingestion_service = service
    app.state.knowledge_projection_service = KnowledgeProjectionService(service)
    if register_routes:
        app.include_router(create_knowledge_source_router(ingestion=service, scope_dependency=source_scope))
    return service


def create_app_knowledge_api(*, audience, uow_factory, store_root, compute_client, context_factory,
                             auth_context_codec=None, service_identity_codec=None):
    """独立内部摄取组件；业务入口的最终替换和进程启动属于 K7。"""
    from fastapi import FastAPI
    from platform_core.security import (create_scoped_internal_auth_middleware,
        create_auth_context_codec, create_service_identity_codec)
    app = FastAPI(title="App 自有知识摄取")
    app.state.auth_context_codec = auth_context_codec or create_auth_context_codec()
    app.state.service_identity_codec = service_identity_codec or create_service_identity_codec()
    app.middleware("http")(create_scoped_internal_auth_middleware(audience=audience,
        allowed_callers={"kbot-knowledge-core-api":frozenset({"knowledge.source.read"})}))
    install_app_knowledge(app, uow_factory=uow_factory, store_root=store_root,
                         compute_client=compute_client, context_factory=context_factory)
    return app
