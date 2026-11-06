"""知识读取端点的双 JWT 与专用委托校验；请求体不得创建身份。"""

from fastapi import HTTPException, Request
from uuid import UUID
from platform_core.contracts import AuthContext, ServiceIdentity
from knowledge_algorithms.contracts.runtime import DelegationClaims
from knowledge_algorithms.contracts.wire import decode
from ..domain.knowledge_ingestion import OWNER
from ..domain.knowledge_retrieval import RetrievalAccess


def knowledge_access_dependency(*, audience, delegation_codec, allowed_callers):
    async def access(request: Request):
        identity = getattr(request.state, "service_identity", None)
        context = getattr(request.state, "auth_context", None)
        path = request.url.path
        operation = ("knowledge.publish" if path.endswith((":publish", "/citation-targets"))
                     else "knowledge.withdraw" if path.endswith(":withdraw")
                     else "knowledge.scope.create" if path.endswith("/scopes")
                     else "knowledge.content.read" if "/content/" in path or "/evidence:" in path
                     else "knowledge.retrieve")
        if (not isinstance(identity, ServiceIdentity) or not isinstance(context, AuthContext)
                or identity.subject not in allowed_callers or identity.audience != audience
                or operation not in identity.scopes or operation not in context.scopes
                or context.app_id != OWNER or context.domain_id is None):
            raise HTTPException(403, detail={"code": "KNOWLEDGE_DELEGATION_REQUIRED"})
        subject = context.asserted_user_id or context.client_id
        try:
            tenant, domain = None, int(context.domain_id)
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(403, detail={"code": "KNOWLEDGE_CONTEXT_INVALID"}) from exc
        token = request.headers.get("X-Knowledge-Delegation")
        claims = None
        if token:
            try:
                claims = decode(DelegationClaims, delegation_codec.decode(token, audience=audience))
                if (claims.owner_binding_ref != OWNER or claims.subject_ref != subject
                        or claims.domain_ref != str(domain) or claims.tenant_ref != (str(tenant) if tenant else None)
                        or operation not in claims.operations):
                    raise ValueError("知识委托与当前身份不一致")
            except (ValueError, TypeError) as exc:
                raise HTTPException(403, detail={"code": "KNOWLEDGE_DELEGATION_INVALID"}) from exc
        if identity.subject.endswith(("agent-runtime-api", "agent-runtime-worker")) and (
                claims is None or claims.run_ref is None or claims.spec_hash is None):
            raise HTTPException(403, detail={"code": "KNOWLEDGE_RUN_DELEGATION_REQUIRED"})
        return RetrievalAccess(domain, tenant, subject,
            run_ref=claims.run_ref if claims else None, spec_hash=claims.spec_hash if claims else None,
            delegated_scope_hash=claims.scope_hash if claims else None)
    return access
