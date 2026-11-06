"""纯计算 API 的固定 App 委托认证；不得从请求正文选择 owner。"""

from fastapi import HTTPException, Request
from platform_core.contracts import AuthContext, ServiceIdentity
from knowledge_algorithms.contracts.identity import fingerprint
from knowledge_core.domain.parse_jobs import ComputeAccess


OWNERS = {
    "kbot-knowledge-retrieval-app-api": "knowledge_retrieval",
    "kbot-knowledge-retrieval-app-worker": "knowledge_retrieval",
    "kbot-aiops-api": "aiops",
    "kbot-aiops-worker": "aiops",
}


def compute_access(request: Request):
    identity = getattr(request.state, "service_identity", None)
    context = getattr(request.state, "auth_context", None)
    if (not isinstance(identity, ServiceIdentity) or not isinstance(context, AuthContext)
            or identity.subject not in OWNERS or "knowledge.parse.compute" not in identity.scopes
            or "knowledge.parse.compute" not in context.scopes or context.domain_id is None):
        raise HTTPException(403, detail={"code":"COMPUTE_DELEGATION_REQUIRED"})
    owner = OWNERS[identity.subject]
    if context.app_id != owner:
        raise HTTPException(403, detail={"code":"APP_SCOPE_MISMATCH"})
    subject = context.asserted_user_id or context.client_id
    scope = fingerprint({"domain_id":int(context.domain_id)})
    return ComputeAccess(owner, subject, scope)
