"""App 源端点仅接受可信解析服务和同范围的冻结委托。"""

from fastapi import HTTPException, Request
from platform_core.contracts import AuthContext, ServiceIdentity
from aiops_agent.domain.knowledge_ingestion import IngestionScope, OWNER


def source_scope(request: Request):
    identity = getattr(request.state, "service_identity", None)
    context = getattr(request.state, "auth_context", None)
    if (not isinstance(identity, ServiceIdentity) or not isinstance(context, AuthContext)
            or identity.subject != "kbot-knowledge-core-api"
            or "knowledge.source.read" not in identity.scopes or "knowledge.source.read" not in context.scopes
            or context.app_id != OWNER or context.domain_id is None):
        raise HTTPException(403, detail={"code":"SOURCE_DELEGATION_REQUIRED"})
    return IngestionScope(int(context.domain_id), None, context.asserted_user_id or context.client_id)
