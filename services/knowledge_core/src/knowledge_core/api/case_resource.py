"""独立保护的系统资源入口；不接受用户或普通服务令牌。"""

from fastapi import APIRouter, HTTPException, Request

from knowledge_core.application.case_resource import CaseResourceConflict
from platform_core.contracts import PrincipalKind
from platform_core.contracts.aiops.case_resource import (
    CASE_RESOURCE_PATH, CASE_RESOURCE_SCOPE, CaseResourceRequest, CaseResourceResult,
)
from platform_core.security import (
    create_auth_context_codec, create_service_identity_codec,
    create_scoped_internal_auth_middleware, get_auth_context,
)

router = APIRouter()


def install_case_resource_boundary(app, *, audience, caller_service):
    protected = create_scoped_internal_auth_middleware(audience=audience,
        allowed_callers={caller_service: frozenset({CASE_RESOURCE_SCOPE})},
        service_identity_codec=create_service_identity_codec(),
        auth_context_codec=create_auth_context_codec(), public_paths=set())

    @app.middleware("http")
    async def case_resource_auth(request, call_next):
        if request.url.path == CASE_RESOURCE_PATH:
            return await protected(request, call_next)
        return await call_next(request)

    app.include_router(router)


@router.post(CASE_RESOURCE_PATH, response_model=CaseResourceResult, tags=["System Resources"])
async def ensure_case_resource(payload: CaseResourceRequest, request: Request):
    context = get_auth_context(request)
    identity = getattr(request.state, "service_identity", None)
    if (context.principal_kind != PrincipalKind.SERVICE or identity is None
        or CASE_RESOURCE_SCOPE not in identity.scopes):
        raise HTTPException(403, {"code": "CASE_RESOURCE_CALLER_DENIED"})
    try:
        return await request.app.state.case_resource_service.ensure(payload)
    except CaseResourceConflict as exc:
        raise HTTPException(409, {"code": exc.code}) from exc
