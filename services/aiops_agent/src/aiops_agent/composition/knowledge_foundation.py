"""摄取、授权检索和内容读取的同进程组合；不启动旧 KC 数据路径。"""

from fastapi import FastAPI
from knowledge_algorithms.contracts.errors import require
from platform_core.security import (create_auth_context_codec, create_service_identity_codec,
                                   create_scoped_internal_auth_middleware)
from .knowledge_ingestion import install_app_knowledge
from .knowledge_retrieval import install_knowledge_retrieval


def create_knowledge_foundation_api(*, audience, allowed_callers, compute_caller,
        uow_factory, store_root, compute_client, context_factory, authorization,
        delegation_codec, profiles, tokenizer, image_cost, cursor_codec,
        model_gateway_factory=None, text_scan_factory=None, limits=None,
        auth_context_codec=None, service_identity_codec=None, management_callers=()):
    """业务 App 必须显式提供授权器、允许视图策略和冻结配置；没有默认全域授权。"""
    app = FastAPI(title="App 独占知识基础能力")
    app.state.auth_context_codec = auth_context_codec or create_auth_context_codec()
    app.state.service_identity_codec = service_identity_codec or create_service_identity_codec()
    read_scopes = frozenset({"knowledge.scope.create", "knowledge.retrieve", "knowledge.content.read"})
    callers = {caller: read_scopes for caller in allowed_callers}
    require(not (set(allowed_callers) & set(management_callers)), "读取与管理调用方必须显式分离")
    callers.update({caller: read_scopes | {"knowledge.publish", "knowledge.withdraw"} for caller in management_callers})
    require(compute_caller not in callers, "纯计算调用方不能兼任业务读取或管理调用方")
    callers[compute_caller] = frozenset({"knowledge.source.read"})
    app.middleware("http")(create_scoped_internal_auth_middleware(audience=audience, allowed_callers=callers))
    ingestion = install_app_knowledge(app, uow_factory=uow_factory, store_root=store_root,
        compute_client=compute_client, context_factory=context_factory)
    install_knowledge_retrieval(app, audience=audience, delegation_codec=delegation_codec,
        allowed_callers=tuple(set(allowed_callers) | set(management_callers)), uow_factory=uow_factory, object_store=ingestion.store,
        authorization=authorization, profiles=profiles, tokenizer=tokenizer, image_cost=image_cost,
        cursor_codec=cursor_codec, model_gateway_factory=model_gateway_factory,
        text_scan_factory=text_scan_factory, limits=limits)
    return app
