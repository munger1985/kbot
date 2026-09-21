"""知识检索应用内部 API 的共享鉴权与错误映射。"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from knowledge_retrieval_app.application import AgentApplicationError, KnowledgeRetrievalApplicationError
from platform_core.contracts import AuthContext, ServiceIdentity


TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "REJECTED"})


def require_actor(request: Request, domain_id: int) -> str:
    identity = getattr(request.state, "service_identity", None)
    if not isinstance(identity, ServiceIdentity) or "knowledge_retrieval.manage" not in identity.scopes:
        raise HTTPException(403, {"code": "SERVICE_SCOPE_DENIED"})
    context = getattr(request.state, "auth_context", None)
    if not isinstance(context, AuthContext) or context.domain_id is None:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"})
    if int(context.domain_id) != domain_id:
        raise HTTPException(403, {"code": "DOMAIN_SCOPE_MISMATCH"})
    return context.asserted_user_id or context.client_id


def auth_context(request: Request) -> AuthContext:
    context = getattr(request.state, "auth_context", None)
    if not isinstance(context, AuthContext):
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"})
    return context


def raise_application_error(exc: AgentApplicationError | KnowledgeRetrievalApplicationError) -> None:
    raise HTTPException(exc.status_code, {"code": exc.code, "message": exc.message}) from exc


def idempotency_key(request: Request, header_value: str | None) -> str:
    value = str(header_value or "").strip()
    if value:
        return value
    return auth_context(request).request_id


def run_response(view: dict, created: bool = False) -> JSONResponse:
    del created
    status_code = 200 if view.get("status") in TERMINAL_STATUSES else 202
    return JSONResponse(status_code=status_code, content=view)
