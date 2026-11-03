"""仅限输出 Worker 的当前授权查询，不作为公开业务或登录接口。"""

from fastapi import APIRouter, HTTPException, Request

from main_api.application.output_authority import OutputAuthorityDenied
from platform_core.contracts import PrincipalKind
from platform_core.contracts.aiops.output_authority import OutputAuthorityRequest, OutputAuthoritySnapshot
from platform_core.security import (
    create_auth_context_codec, create_service_identity_codec,
    create_scoped_internal_auth_middleware, get_auth_context,
)


OUTPUT_AUTHORITY_PATH = "/internal/v1/platform/aiops/output-authority"
OUTPUT_AUTHORITY_SCOPE = "aiops.output.authorize"

router = APIRouter(prefix=OUTPUT_AUTHORITY_PATH, tags=["AIOps Output Authority"])


def install_output_authority_boundary(app, *, audience: str, caller_service: str):
    """只保护明确内部前缀；不改变公开路由、业务权限或其他内部接口。"""
    protected = create_scoped_internal_auth_middleware(
        audience=audience, allowed_callers={caller_service: frozenset({OUTPUT_AUTHORITY_SCOPE})},
        service_identity_codec=create_service_identity_codec(),
        auth_context_codec=create_auth_context_codec(), public_paths=set(),
    )

    @app.middleware("http")
    async def output_authority_auth(request, call_next):
        path = request.url.path
        if path == OUTPUT_AUTHORITY_PATH or path.startswith(OUTPUT_AUTHORITY_PATH + "/"):
            return await protected(request, call_next)
        return await call_next(request)

    app.include_router(router, include_in_schema=False)


@router.post("", response_model=OutputAuthoritySnapshot)
async def resolve_output_authority(payload: OutputAuthorityRequest, request: Request):
    context = get_auth_context(request)
    identity = getattr(request.state, "service_identity", None)
    if (context.principal_kind != PrincipalKind.SERVICE or identity is None
        or OUTPUT_AUTHORITY_SCOPE not in identity.scopes):
        raise HTTPException(403, {"code": "AIOPS_OUTPUT_AUTHORITY_CALLER_DENIED"})
    try:
        return await request.app.state.output_authority_service.resolve(payload)
    except OutputAuthorityDenied as exc:
        raise HTTPException(403, {"code": "AIOPS_OUTPUT_AUTHORITY_REVOKED",
            "message": "发起人的当前账号、业务空间或操作权限已失效"}) from exc
