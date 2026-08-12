"""KM 独立页面用户登录、密码更新与短期令牌。"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import bcrypt
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError, JWTError

from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.security import PortalApiKeyError, extract_bearer_token


_ALGORITHM = "HS256"
_AUDIENCE = "kbot-km-ui"
_TOKEN_TYPE = "kbot-km-user+jwt"


class KmUserAuthenticationError(ValueError):
    """KM 用户登录或修改密码失败。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class KmUserTokenClaims:
    user_id: str
    domain_id: int
    must_change_password: bool
    expires_at: datetime


class KmUserTokenCodec:
    """签发并校验仅可用于 KM 公开页面的用户 JWT。"""

    def __init__(self, *, secret: str, issuer: str, ttl_seconds: int):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("KM 用户 JWT 密钥至少需要 32 字节")
        self._secret = secret
        self._issuer = issuer
        self._ttl_seconds = ttl_seconds

    def issue(
        self, *, user_id: str, domain_id: int, must_change_password: bool
    ) -> tuple[str, datetime]:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self._ttl_seconds)
        token = jwt.encode(
            {
                "iss": self._issuer,
                "aud": _AUDIENCE,
                "sub": user_id,
                "domain_id": domain_id,
                "must_change_password": must_change_password,
                "typ": _TOKEN_TYPE,
                "iat": int(now.timestamp()),
                "nbf": int(now.timestamp()),
                "exp": int(expires_at.timestamp()),
                "jti": str(uuid4()),
            },
            self._secret,
            algorithm=_ALGORITHM,
        )
        return token, expires_at

    def verify_authorization(self, authorization: str | None) -> KmUserTokenClaims:
        token = extract_bearer_token(authorization)
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[_ALGORITHM],
                audience=_AUDIENCE,
                issuer=self._issuer,
                options={
                    "require_aud": True,
                    "require_exp": True,
                    "require_iat": True,
                    "require_iss": True,
                    "require_jti": True,
                    "require_nbf": True,
                    "require_sub": True,
                },
            )
            if claims.get("typ") != _TOKEN_TYPE:
                raise KmUserAuthenticationError("INVALID_KM_TOKEN", "KM 用户令牌类型无效")
            return KmUserTokenClaims(
                user_id=str(claims["sub"]),
                domain_id=int(claims["domain_id"]),
                must_change_password=bool(claims.get("must_change_password")),
                expires_at=datetime.fromtimestamp(int(claims["exp"]), timezone.utc),
            )
        except KmUserAuthenticationError:
            raise
        except ExpiredSignatureError as exc:
            raise KmUserAuthenticationError("KM_TOKEN_EXPIRED", "KM 登录已过期，请重新登录") from exc
        except (JWTClaimsError, JWTError, KeyError, TypeError, ValueError) as exc:
            raise KmUserAuthenticationError("INVALID_KM_TOKEN", "KM 用户令牌无效") from exc


class KmUserAuthService:
    """校验 KM 用户凭据并管理登录 Token。"""

    def __init__(self, *, uow_factory, codec: KmUserTokenCodec):
        self._uow_factory = uow_factory
        self._codec = codec

    async def login(
        self, *, user_id: str, password: str, domain_id: int | None
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
            credential = await uow.access.get_user_credential(user_id)
            domain_ids = await uow.access.list_active_km_domain_ids(user_id)
        valid_password = bool(
            user is not None
            and user.status == "ACTIVE"
            and credential is not None
            and await asyncio.to_thread(
                bcrypt.checkpw,
                password.encode("utf-8"),
                credential.password_hash.encode("ascii"),
            )
        )
        if not valid_password:
            raise KmUserAuthenticationError("INVALID_CREDENTIALS", "用户名或密码错误")
        if not domain_ids:
            raise KmUserAuthenticationError("KM_ACCESS_DENIED", "用户没有启用的 KM 访问权限")
        selected_domain = domain_id if domain_id is not None else domain_ids[0]
        if selected_domain not in domain_ids:
            raise KmUserAuthenticationError("KM_ACCESS_DENIED", "用户无权访问指定 Domain")
        must_change = credential.must_change_password == "Y"
        token, expires_at = self._codec.issue(
            user_id=user_id,
            domain_id=selected_domain,
            must_change_password=must_change,
        )
        return {
            "access_token": token,
            "token_type": "Bearer",
            "expires_at": expires_at,
            "user_id": user_id,
            "display_name": user.display_name,
            "domain_id": selected_domain,
            "available_domain_ids": list(domain_ids),
            "must_change_password": must_change,
        }

    async def change_password(
        self, *, claims: KmUserTokenClaims, current_password: str, new_password: str
    ) -> dict[str, object]:
        if current_password == new_password:
            raise KmUserAuthenticationError("PASSWORD_REUSED", "新密码不能与当前密码相同")
        async with self._uow_factory() as uow:
            credential = await uow.access.get_user_credential(claims.user_id)
            if credential is None or not await asyncio.to_thread(
                bcrypt.checkpw,
                current_password.encode("utf-8"),
                credential.password_hash.encode("ascii"),
            ):
                raise KmUserAuthenticationError("INVALID_CREDENTIALS", "当前密码错误")
            password_hash = await asyncio.to_thread(
                bcrypt.hashpw,
                new_password.encode("utf-8"),
                bcrypt.gensalt(rounds=12),
            )
            await uow.access.set_user_password(
                credential=credential,
                password_hash=password_hash.decode("ascii"),
            )
            await uow.commit()
        token, expires_at = self._codec.issue(
            user_id=claims.user_id,
            domain_id=claims.domain_id,
            must_change_password=False,
        )
        return {
            "access_token": token,
            "token_type": "Bearer",
            "expires_at": expires_at,
            "user_id": claims.user_id,
            "domain_id": claims.domain_id,
            "must_change_password": False,
        }

    def authenticate_request(self, authorization: str | None) -> AuthContext | None:
        if not authorization:
            return None
        token = authorization.partition(" ")[2].strip()
        if token.startswith("kbot_sk_"):
            return None
        try:
            claims = self._codec.verify_authorization(authorization)
        except KmUserAuthenticationError as exc:
            raise PortalApiKeyError(exc.code, str(exc)) from exc
        return AuthContext(
            principal_kind=PrincipalKind.PORTAL,
            client_id="km-user-session",
            api_key_id="km-user-jwt",
            domain_id=str(claims.domain_id),
            asserted_user_id=claims.user_id,
            request_id=str(uuid4()),
            trace_id=str(uuid4()),
        )

    def verify(self, authorization: str | None) -> KmUserTokenClaims:
        return self._codec.verify_authorization(authorization)


def create_km_user_token_codec(*, settings) -> KmUserTokenCodec:
    """从部署 Secret 创建 KM 用户令牌编解码器。"""
    env_name = settings.security.km_user_jwt_secret_env
    secret = os.getenv(env_name)
    if not secret:
        raise RuntimeError(f"KM 用户 JWT 密钥环境变量 {env_name} 未设置")
    return KmUserTokenCodec(
        secret=secret,
        issuer=settings.security.km_user_jwt_issuer,
        ttl_seconds=settings.security.km_user_jwt_ttl_seconds,
    )


__all__ = [
    "KmUserAuthenticationError",
    "KmUserAuthService",
    "KmUserTokenClaims",
    "KmUserTokenCodec",
    "create_km_user_token_codec",
]
