"""平台用户登录、Domain 切换、密码更新与短期令牌。"""

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
_AUDIENCE = "kbot-public-api"
_TOKEN_TYPE = "kbot-user+jwt"
KM_PORTAL_DOMAIN_NAME = "km_portal"


class UserAuthenticationError(ValueError):
    """平台用户认证、Domain 选择或密码更新失败。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class UserTokenClaims:
    user_id: str
    domain_id: int
    must_change_password: bool
    password_version: int
    expires_at: datetime


class UserTokenCodec:
    """签发并校验供 Main API 公开路由使用的用户 JWT。"""

    def __init__(self, *, secret: str, issuer: str, ttl_seconds: int):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("平台用户 JWT 密钥至少需要 32 字节")
        self._secret = secret
        self._issuer = issuer
        self._ttl_seconds = ttl_seconds

    def issue(
        self,
        *,
        user_id: str,
        domain_id: int,
        must_change_password: bool,
        password_version: int,
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
                "pwd": password_version,
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

    def verify_authorization(self, authorization: str | None) -> UserTokenClaims:
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
                raise UserAuthenticationError(
                    "INVALID_USER_TOKEN", "平台用户令牌类型无效"
                )
            return UserTokenClaims(
                user_id=str(claims["sub"]),
                domain_id=int(claims["domain_id"]),
                must_change_password=bool(claims.get("must_change_password")),
                password_version=int(claims["pwd"]),
                expires_at=datetime.fromtimestamp(int(claims["exp"]), timezone.utc),
            )
        except UserAuthenticationError:
            raise
        except ExpiredSignatureError as exc:
            raise UserAuthenticationError(
                "USER_TOKEN_EXPIRED", "用户登录已过期，请重新登录"
            ) from exc
        except (JWTClaimsError, JWTError, KeyError, TypeError, ValueError) as exc:
            raise UserAuthenticationError(
                "INVALID_USER_TOKEN", "平台用户令牌无效"
            ) from exc


class UserAuthService:
    """校验平台用户凭据并管理公开 API 用户会话。"""

    def __init__(self, *, uow_factory, codec: UserTokenCodec):
        self._uow_factory = uow_factory
        self._codec = codec

    async def _credential_snapshot(
        self, *, user_id: str
    ) -> tuple[str | None, str | None, str | None, bool, int | None]:
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
            credential = await uow.access.get_user_credential(user_id)
            return (
                user.status if user is not None else None,
                user.display_name if user is not None else None,
                credential.password_hash if credential is not None else None,
                bool(
                    credential is not None
                    and credential.must_change_password == "Y"
                ),
                (
                    self._timestamp_version(credential.password_updated_at)
                    if credential is not None
                    else None
                ),
            )

    async def _verify_credentials(self, *, user_id: str, password: str):
        status, display_name, password_hash, must_change_password, password_version = (
            await self._credential_snapshot(user_id=user_id)
        )
        valid = bool(
            status == "ACTIVE"
            and password_hash
            and await asyncio.to_thread(
                bcrypt.checkpw,
                password.encode("utf-8"),
                password_hash.encode("ascii"),
            )
        )
        if not valid:
            raise UserAuthenticationError(
                "INVALID_CREDENTIALS", "用户名或密码错误"
            )
        return display_name, must_change_password, password_version

    async def login(
        self, *, user_id: str, password: str, domain_id: int
    ) -> dict[str, object]:
        display_name, must_change_password, password_version = await self._verify_credentials(
            user_id=user_id, password=password
        )
        async with self._uow_factory() as uow:
            domain_ids = await uow.access.list_active_domain_ids(user_id)
            domain = await uow.domains.get(domain_id=domain_id)
            domain_status = domain.status if domain is not None else None
            domain_name = domain.name if domain is not None else None
        if (
            domain is None
            or domain_status != "ACTIVE"
            or domain_id not in domain_ids
        ):
            raise UserAuthenticationError(
                "DOMAIN_ACCESS_DENIED", "用户没有所选 Domain 的有效访问权限"
            )
        return self._token_response(
            user_id=user_id,
            display_name=display_name,
            domain_id=domain_id,
            domain_name=domain_name,
            must_change_password=must_change_password,
            password_version=int(password_version),
        )

    async def list_login_domains(
        self, *, user_id: str, password: str
    ) -> dict[str, object]:
        display_name, must_change_password, _ = await self._verify_credentials(
            user_id=user_id, password=password
        )
        async with self._uow_factory() as uow:
            domain_ids = await uow.access.list_active_domain_ids(user_id)
            domains = await uow.domains.list_by_ids(domain_ids=domain_ids)
            items = [
                {
                    "domain_id": int(row.domain_id),
                    "name": row.name,
                    "status": row.status,
                }
                for row in domains
                if row.status == "ACTIVE"
            ]
        return {
            "user_id": user_id,
            "display_name": display_name,
            "must_change_password": must_change_password,
            "domains": items,
        }

    async def login_for_domain_name(
        self, *, user_id: str, password: str, domain_name: str
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            domain = await uow.domains.get_by_name(name=domain_name)
            domain_id = (
                int(domain.domain_id)
                if domain is not None and domain.status == "ACTIVE"
                else None
            )
        if domain_id is None:
            raise UserAuthenticationError(
                "DOMAIN_UNAVAILABLE", "固定 Domain 尚未初始化或未启用"
            )
        return await self.login(
            user_id=user_id, password=password, domain_id=domain_id
        )

    def _token_response(
        self,
        *,
        user_id: str,
        display_name: str | None,
        domain_id: int,
        domain_name: str | None,
        must_change_password: bool,
        password_version: int,
    ) -> dict[str, object]:
        token, expires_at = self._codec.issue(
            user_id=user_id,
            domain_id=domain_id,
            must_change_password=must_change_password,
            password_version=password_version,
        )
        return {
            "access_token": token,
            "token_type": "Bearer",
            "expires_at": expires_at,
            "user_id": user_id,
            "display_name": display_name,
            "domain_id": domain_id,
            "domain_name": domain_name,
            "must_change_password": must_change_password,
        }

    async def profile(self, *, claims: UserTokenClaims) -> dict[str, object]:
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(claims.user_id)
            memberships = await uow.access.list_user_memberships(
                user_id=claims.user_id
            )
            domain_ids = await uow.access.list_active_domain_ids(claims.user_id)
            domains = await uow.domains.list_by_ids(domain_ids=domain_ids)
            user_snapshot = (
                {
                    "user_id": user.user_id,
                    "display_name": user.display_name,
                    "status": user.status,
                }
                if user is not None
                else None
            )
            domain_items = [
                {
                    "domain_id": int(row.domain_id),
                    "name": row.name,
                    "status": row.status,
                }
                for row in domains
            ]
            membership_items = [
                {
                    "app_id": row.app_id,
                    "domain_id": int(row.domain_id),
                    "role_code": row.role_code,
                    "status": row.status,
                }
                for row in memberships
            ]
        if user_snapshot is None or user_snapshot["status"] != "ACTIVE":
            raise UserAuthenticationError(
                "USER_DISABLED", "用户不存在或已停用"
            )
        return {
            **user_snapshot,
            "domain_id": claims.domain_id,
            "must_change_password": claims.must_change_password,
            "domains": domain_items,
            "memberships": membership_items,
        }

    async def switch_domain(
        self, *, claims: UserTokenClaims, domain_id: int
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(claims.user_id)
            domain_ids = await uow.access.list_active_domain_ids(claims.user_id)
            domain = await uow.domains.get(domain_id=domain_id)
            user_snapshot = (
                (user.user_id, user.display_name, user.status)
                if user is not None
                else None
            )
            domain_snapshot = (
                (domain.name, domain.status) if domain is not None else None
            )
        if (
            user_snapshot is None
            or user_snapshot[2] != "ACTIVE"
            or domain_snapshot is None
            or domain_snapshot[1] != "ACTIVE"
            or domain_id not in domain_ids
        ):
            raise UserAuthenticationError(
                "DOMAIN_ACCESS_DENIED", "用户没有所选 Domain 的有效访问权限"
            )
        return self._token_response(
            user_id=user_snapshot[0],
            display_name=user_snapshot[1],
            domain_id=domain_id,
            domain_name=domain_snapshot[0],
            must_change_password=claims.must_change_password,
            password_version=claims.password_version,
        )

    async def change_password(
        self,
        *,
        claims: UserTokenClaims,
        current_password: str,
        new_password: str,
    ) -> dict[str, object]:
        if current_password == new_password:
            raise UserAuthenticationError(
                "PASSWORD_REUSED", "新密码不能与当前密码相同"
            )
        async with self._uow_factory() as uow:
            credential = await uow.access.get_user_credential(claims.user_id)
            valid = bool(
                credential is not None
                and await asyncio.to_thread(
                    bcrypt.checkpw,
                    current_password.encode("utf-8"),
                    credential.password_hash.encode("ascii"),
                )
            )
            if not valid:
                raise UserAuthenticationError(
                    "INVALID_CREDENTIALS", "当前密码错误"
                )
            password_hash = await asyncio.to_thread(
                bcrypt.hashpw,
                new_password.encode("utf-8"),
                bcrypt.gensalt(rounds=12),
            )
            await uow.access.set_user_password(
                credential=credential,
                password_hash=password_hash.decode("ascii"),
            )
            user = await uow.access.get_user(claims.user_id)
            domain = await uow.domains.get(domain_id=claims.domain_id)
            display_name = user.display_name if user is not None else None
            domain_name = domain.name if domain is not None else None
            password_version = self._timestamp_version(
                credential.password_updated_at
            )
            await uow.commit()
        return self._token_response(
            user_id=claims.user_id,
            display_name=display_name,
            domain_id=claims.domain_id,
            domain_name=domain_name,
            must_change_password=False,
            password_version=password_version,
        )

    async def validate_session(self, *, claims: UserTokenClaims) -> None:
        """在每次请求时校验用户状态、成员关系和密码版本。"""
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(claims.user_id)
            credential = await uow.access.get_user_credential(claims.user_id)
            domain_ids = await uow.access.list_active_domain_ids(claims.user_id)
            current_version = (
                self._timestamp_version(credential.password_updated_at)
                if credential is not None
                else None
            )
            user_status = user.status if user is not None else None
        if user_status != "ACTIVE":
            raise UserAuthenticationError("USER_DISABLED", "用户不存在或已停用")
        if claims.domain_id not in domain_ids:
            raise UserAuthenticationError(
                "DOMAIN_ACCESS_DENIED", "用户已失去当前 Domain 的访问权限"
            )
        if current_version != claims.password_version:
            raise UserAuthenticationError(
                "USER_SESSION_REVOKED", "密码已更新，请重新登录"
            )

    @staticmethod
    def _timestamp_version(value: datetime) -> int:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1_000_000)

    def authenticate_request(self, authorization: str | None) -> AuthContext | None:
        if not authorization:
            return None
        token = authorization.partition(" ")[2].strip()
        if token.startswith("kbot_sk_"):
            return None
        try:
            claims = self._codec.verify_authorization(authorization)
        except UserAuthenticationError as exc:
            raise PortalApiKeyError(exc.code, str(exc)) from exc
        return AuthContext(
            principal_kind=PrincipalKind.PORTAL,
            client_id="user-session",
            api_key_id="user-jwt",
            domain_id=str(claims.domain_id),
            asserted_user_id=claims.user_id,
            request_id=str(uuid4()),
            trace_id=str(uuid4()),
        )

    def verify(self, authorization: str | None) -> UserTokenClaims:
        return self._codec.verify_authorization(authorization)


def create_user_token_codec(*, settings) -> UserTokenCodec:
    """从部署 Secret 创建平台用户令牌编解码器。"""
    env_name = settings.security.user_jwt_secret_env
    secret = os.getenv(env_name)
    if not secret:
        raise RuntimeError(f"平台用户 JWT 密钥环境变量 {env_name} 未设置")
    return UserTokenCodec(
        secret=secret,
        issuer=settings.security.user_jwt_issuer,
        ttl_seconds=settings.security.user_jwt_ttl_seconds,
    )


__all__ = [
    "KM_PORTAL_DOMAIN_NAME",
    "UserAuthenticationError",
    "UserAuthService",
    "UserTokenClaims",
    "UserTokenCodec",
    "create_user_token_codec",
]
