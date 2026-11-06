"""知识模块组合根；固定 App 绑定，显式注入业务授权与模型配置。"""

from fastapi import FastAPI
from platform_core.security import (create_auth_context_codec, create_scoped_internal_auth_middleware,
                                     create_service_identity_codec)
from ..api.knowledge_retrieval import create_knowledge_retrieval_router, install_knowledge_errors
from ..api.knowledge_retrieval_auth import knowledge_access_dependency
from ..application.knowledge_retrieval import KnowledgeRetrievalService


def install_knowledge_retrieval(app, *, audience, delegation_codec, allowed_callers, register_routes=True, **service_dependencies):
    service = KnowledgeRetrievalService(**service_dependencies)
    dependency = knowledge_access_dependency(audience=audience, delegation_codec=delegation_codec,
                                              allowed_callers=frozenset(allowed_callers))
    app.state.knowledge_retrieval_service = service
    if register_routes:
        app.include_router(create_knowledge_retrieval_router(service=service, access_dependency=dependency))
    install_knowledge_errors(app)
    return service


def create_knowledge_retrieval_api(*, audience, delegation_codec, allowed_callers,
                                   auth_context_codec=None, service_identity_codec=None,
                                   management_callers=(), **dependencies):
    app = FastAPI(title="App 独占知识解析与召回接入")
    app.state.auth_context_codec = auth_context_codec or create_auth_context_codec()
    app.state.service_identity_codec = service_identity_codec or create_service_identity_codec()
    scopes = frozenset({"knowledge.scope.create", "knowledge.retrieve", "knowledge.content.read"})
    app.middleware("http")(create_scoped_internal_auth_middleware(audience=audience,
        allowed_callers={**{caller: scopes for caller in allowed_callers},
                         **{caller: scopes | {"knowledge.publish", "knowledge.withdraw"} for caller in management_callers}}))
    install_knowledge_retrieval(app, audience=audience, delegation_codec=delegation_codec,
        allowed_callers=tuple(set(allowed_callers) | set(management_callers)), **dependencies)
    return app
