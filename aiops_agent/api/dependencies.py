"""AIOps AuthContext 与 Service Identity 依赖。"""

from __future__ import annotations

from fastapi import HTTPException, Request

from platform_core.contracts import AuthContext, ServiceIdentity


def get_aiops_auth_context(request: Request) -> AuthContext:
    context = getattr(request.state, "auth_context", None)
    if not isinstance(context, AuthContext):
        raise RuntimeError("请求尚未通过 AIOps AuthContext 验证")
    return context


def require_service_scope(request: Request, scope: str) -> ServiceIdentity:
    identity = getattr(request.state, "service_identity", None)
    if not isinstance(identity, ServiceIdentity):
        raise RuntimeError("请求尚未通过 Service Identity 验证")
    if scope not in identity.scopes:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "SERVICE_SCOPE_DENIED",
                "message": f"当前服务身份缺少 scope：{scope}",
            },
        )
    return identity
