"""平台普通用户登录与会话管理公开接口。"""

from typing import cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from main_api.application import UserAuthService, UserTokenClaims
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(prefix=f"{PUBLIC_API_V1}/auth", tags=["Authentication"])


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginPayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)
    domain_id: int = Field(gt=0)


class LoginDomainPayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class SwitchDomainPayload(_Payload):
    domain_id: int = Field(gt=0)


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
        raise HTTPException(
            401,
            {
                "code": "USER_SESSION_REQUIRED",
                "message": "该接口需要平台用户登录 Token",
            },
        )
    return cast(UserTokenClaims, claims)


@router.post("/login")
async def login(payload: LoginPayload, request: Request):
    """使用用户名、密码及目标 Domain 换取用户 Token。"""
    return await _service(request).login(
        user_id=payload.user_id.strip(),
        password=payload.password,
        domain_id=payload.domain_id,
    )


@router.post("/domains")
async def list_login_domains(payload: LoginDomainPayload, request: Request):
    """验证用户名和密码后返回可选择的有效 Domain。"""
    return await _service(request).list_login_domains(
        user_id=payload.user_id.strip(), password=payload.password
    )


@router.get("/me")
async def get_current_user(request: Request):
    """返回当前登录用户、可访问 Domain 和全部成员关系。"""
    return await _service(request).profile(claims=_claims(request))


@router.post("/switch-domain")
async def switch_domain(payload: SwitchDomainPayload, request: Request):
    """校验用户成员关系后签发目标 Domain 的新 Token。"""
    return await _service(request).switch_domain(
        claims=_claims(request), domain_id=payload.domain_id
    )


@router.post("/password")
async def change_password(payload: PasswordChangePayload, request: Request):
    """修改当前登录用户密码并签发替换 Token。"""
    return await _service(request).change_password(
        claims=_claims(request),
        current_password=payload.current_password,
        new_password=payload.new_password,
    )


__all__ = ["router"]
