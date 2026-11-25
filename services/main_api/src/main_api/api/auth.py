"""统一工作台与 App 登录公开接口。"""

from typing import cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from main_api.application import UserAuthService, UserTokenClaims
from platform_core.authorization import resolve_business_app_id
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(prefix=f"{PUBLIC_API_V1}/auth", tags=["Authentication"])


def _canonical_app_id(value: str) -> str:
    try:
        return resolve_business_app_id(value)
    except ValueError as exc:
        raise HTTPException(404, {"code": "APP_NOT_FOUND"}) from exc


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CredentialPayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class AppLoginPayload(CredentialPayload):
    domain_id: int = Field(gt=0)


class SwitchDomainPayload(_Payload):
    domain_id: int = Field(gt=0)


class SessionDomainItem(_Payload):
    domain_id: int = Field(gt=0)
    name: str
    status: str


class SessionDomainsResponse(_Payload):
    app_id: str
    domain_id: int = Field(gt=0)
    domains: list[SessionDomainItem]


class SessionAppEntry(_Payload):
    app_id: str
    display_name: str
    domains: list[SessionDomainItem]
    can_create_first_domain: bool = Field(
        default=False, description="获授权的知识检索管理员可从工作台开通首个工作域。",
    )


class SessionEntriesResponse(_Payload):
    user_id: str
    display_name: str | None
    account_origin: str
    apps: list[SessionAppEntry]
    system_management: bool = Field(description="当前平台会话具有全局 AI 模型管理权限。")
    user_management: bool = Field(description="当前平台会话具有全局用户管理权限。")


class ExchangeAppPayload(_Payload):
    app_id: str = Field(min_length=1, max_length=128)
    domain_id: int = Field(gt=0)


class FirstDomainPayload(_Payload):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)


class PasswordChangePayload(_Payload):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value: str) -> str:
        if not (
            any(char.islower() for char in value)
            and any(char.isupper() for char in value)
            and any(char.isdigit() for char in value)
            and any(not char.isalnum() for char in value)
        ):
            raise ValueError("新密码必须同时包含大小写字母、数字和特殊字符")
        return value


def _service(request: Request) -> UserAuthService:
    return cast(UserAuthService, request.app.state.user_auth_service)


def _claims(request: Request) -> UserTokenClaims:
    claims = getattr(request.state, "user_token_claims", None)
    if claims is None:
        raise HTTPException(401, {"code": "USER_SESSION_REQUIRED", "message": "该接口需要用户登录 Token"})
    return cast(UserTokenClaims, claims)


@router.post("/login", openapi_extra={"x-sensitive-input": True})
async def workspace_login(payload: CredentialPayload, request: Request):
    """使用现有 KBot 账号登录统一工作台，无需指定账号来源或 App。"""
    return await _service(request).workspace_login(user_id=payload.user_id.strip(), password=payload.password)


@router.post("/platform/login")
async def platform_login(payload: CredentialPayload, request: Request):
    return await _service(request).platform_login(user_id=payload.user_id.strip(), password=payload.password)


@router.post("/apps")
async def list_login_apps(payload: CredentialPayload, request: Request):
    return await _service(request).list_login_apps(user_id=payload.user_id.strip(), password=payload.password)


@router.post("/apps/{app_id}/domains")
async def list_login_domains(app_id: str, payload: CredentialPayload, request: Request):
    app_id = _canonical_app_id(app_id)
    return await _service(request).list_login_domains(
        user_id=payload.user_id.strip(), password=payload.password, app_id=app_id
    )


@router.post("/apps/{app_id}/login")
async def app_login(app_id: str, payload: AppLoginPayload, request: Request):
    app_id = _canonical_app_id(app_id)
    return await _service(request).app_login(
        user_id=payload.user_id.strip(), password=payload.password,
        app_id=app_id, domain_id=payload.domain_id,
    )


@router.get("/me")
async def get_current_user(request: Request):
    return await _service(request).profile(claims=_claims(request))


@router.get("/entries", response_model=SessionEntriesResponse)
async def list_session_entries(request: Request):
    """使用工作台根会话列出可进入的 App 与 Domain。"""
    return await _service(request).list_session_entries(
        claims=_claims(request)
    )


@router.post("/exchange")
async def exchange_app_session(payload: ExchangeAppPayload, request: Request):
    """使用工作台根会话免密换取显式授权范围内的业务会话。"""
    return await _service(request).exchange_app_session(
        claims=_claims(request),
        app_id=_canonical_app_id(payload.app_id.strip()),
        domain_id=payload.domain_id,
    )


@router.post("/knowledge-first-domain")
async def create_knowledge_first_domain(payload: FirstDomainPayload, request: Request):
    """为获授权的知识检索管理员开通首域并免密进入。"""
    return await _service(request).create_knowledge_first_domain(
        claims=_claims(request), name=payload.name, description=payload.description,
    )


@router.post("/refresh")
async def refresh_session(request: Request):
    """在当前 Token 仍有效时续签相同入口上下文。"""
    return await _service(request).refresh_session(claims=_claims(request))


@router.get("/domains", response_model=SessionDomainsResponse)
async def list_session_domains(request: Request):
    """读取当前业务 App 的可切换工作域，不复用 Domain 管理接口。"""
    return await _service(request).list_session_domains(claims=_claims(request))


@router.post("/switch-domain")
async def switch_domain(payload: SwitchDomainPayload, request: Request):
    return await _service(request).switch_domain(claims=_claims(request), domain_id=payload.domain_id)


@router.post("/password")
async def change_password(payload: PasswordChangePayload, request: Request):
    return await _service(request).change_password(
        claims=_claims(request), current_password=payload.current_password,
        new_password=payload.new_password,
    )


__all__ = ["router"]
