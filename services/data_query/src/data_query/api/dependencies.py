"""认证后的 Data Query API 依赖。"""

from fastapi import HTTPException, Request

from platform_core.contracts import AuthContext, ServiceIdentity


def get_auth_context(request: Request) -> AuthContext:
    context = getattr(request.state, "auth_context", None)
    if not isinstance(context, AuthContext):
        raise RuntimeError("请求尚未通过 Data Query 内部认证")
    return context


def require_scope(request: Request, scope: str) -> ServiceIdentity:
    identity = getattr(request.state, "service_identity", None)
    if not isinstance(identity, ServiceIdentity):
        raise RuntimeError("请求尚未通过 Data Query 服务身份验证")
    if scope not in identity.scopes:
        raise HTTPException(status_code=403, detail={"code": "SERVICE_SCOPE_DENIED"})
    return identity


def domain_id_from_context(context: AuthContext) -> int:
    try:
        domain_id = int(context.domain_id or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=403, detail={"code": "INVALID_DOMAIN_CONTEXT"}
        ) from exc
    if domain_id < 1:
        raise HTTPException(
            status_code=403, detail={"code": "DOMAIN_CONTEXT_REQUIRED"}
        )
    return domain_id


def actor_id_from_context(context: AuthContext) -> str:
    """使用可信 AuthContext 中的稳定 Actor，禁止外部 Header 覆盖。"""
    return context.asserted_user_id or context.client_id
