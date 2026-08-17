"""平台入口与 App 入口登录、密码更新及短期令牌。"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import bcrypt
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError, JWTError

from platform_core.contracts import AuthContext, IdentityEntryKind, PrincipalKind
from platform_core.security import PortalApiKeyError, extract_bearer_token
from main_api.application.access_control import is_reserved_global_admin


_ALGORITHM = "HS256"
_AUDIENCE = "kbot-public-api"
_TOKEN_TYPE = "kbot-user+jwt"
KM_PORTAL_DOMAIN_NAME = "km_portal"


class UserAuthenticationError(ValueError):
    """用户认证、入口选择或密码更新失败。"""

    def __init__(self, code: str, message: str, *, status_code: int = 401):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class UserTokenClaims:
    user_id: str
    entry_kind: str
    app_id: str | None
    domain_id: int | None
    must_change_password: bool
    password_version: int
    expires_at: datetime


class UserTokenCodec:
    """签发并校验绑定平台或业务入口的用户 JWT。"""

    def __init__(self, *, secret: str, issuer: str, ttl_seconds: int):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("平台用户 JWT 密钥至少需要 32 字节")
        self._secret = secret
        self._issuer = issuer
        self._ttl_seconds = ttl_seconds

    def issue(
        self, *, user_id: str, entry_kind: str, app_id: str | None,
        domain_id: int | None, must_change_password: bool,
        password_version: int,
    ) -> tuple[str, datetime]:
        if entry_kind == "PLATFORM" and (app_id is not None or domain_id is not None):
            raise ValueError("平台入口不能包含 App 或 Domain")
        if entry_kind == "BUSINESS" and (not app_id or domain_id is None):
            raise ValueError("业务入口必须包含 App 和 Domain")
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self._ttl_seconds)
        token = jwt.encode(
            {
                "iss": self._issuer, "aud": _AUDIENCE, "sub": user_id,
                "entry": entry_kind, "app_id": app_id, "domain_id": domain_id,
                "must_change_password": must_change_password,
                "pwd": password_version, "typ": _TOKEN_TYPE,
                "iat": int(now.timestamp()), "nbf": int(now.timestamp()),
                "exp": int(expires_at.timestamp()), "jti": str(uuid4()),
            },
            self._secret,
            algorithm=_ALGORITHM,
        )
        return token, expires_at

    def verify_authorization(self, authorization: str | None) -> UserTokenClaims:
        token = extract_bearer_token(authorization)
        try:
            claims = jwt.decode(
                token, self._secret, algorithms=[_ALGORITHM], audience=_AUDIENCE,
                issuer=self._issuer,
                options={
                    "require_aud": True, "require_exp": True, "require_iat": True,
                    "require_iss": True, "require_jti": True, "require_nbf": True,
                    "require_sub": True,
                },
            )
            if claims.get("typ") != _TOKEN_TYPE:
                raise UserAuthenticationError("INVALID_USER_TOKEN", "用户令牌类型无效")
            entry_kind = str(claims["entry"])
            app_id = str(claims["app_id"]) if claims.get("app_id") else None
            domain_id = int(claims["domain_id"]) if claims.get("domain_id") is not None else None
            if entry_kind not in {"PLATFORM", "BUSINESS"}:
                raise ValueError("入口类型无效")
            if entry_kind == "PLATFORM" and (app_id or domain_id is not None):
                raise ValueError("平台令牌上下文无效")
            if entry_kind == "BUSINESS" and (not app_id or domain_id is None):
                raise ValueError("业务令牌上下文无效")
            return UserTokenClaims(
                user_id=str(claims["sub"]), entry_kind=entry_kind,
                app_id=app_id, domain_id=domain_id,
                must_change_password=bool(claims.get("must_change_password")),
                password_version=int(claims["pwd"]),
                expires_at=datetime.fromtimestamp(int(claims["exp"]), timezone.utc),
            )
        except UserAuthenticationError:
            raise
        except ExpiredSignatureError as exc:
            raise UserAuthenticationError("USER_TOKEN_EXPIRED", "用户登录已过期，请重新登录") from exc
        except (JWTClaimsError, JWTError, KeyError, TypeError, ValueError) as exc:
            raise UserAuthenticationError("INVALID_USER_TOKEN", "用户令牌无效") from exc


class UserAuthService:
    """校验账号凭据并签发入口绑定的公开 API 会话。"""

    def __init__(self, *, uow_factory, codec: UserTokenCodec):
        self._uow_factory = uow_factory
        self._codec = codec

    async def _verify_credentials(self, *, user_id: str, password: str):
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
            credential = await uow.access.get_user_credential(user_id)
            snapshot = (
                user.status if user else None,
                user.display_name if user else None,
                user.account_origin if user else None,
                user.owner_app_id if user else None,
                credential.password_hash if credential else None,
                bool(credential and credential.must_change_password == "Y"),
                self._timestamp_version(credential.password_updated_at) if credential else None,
            )
        status, display_name, origin, owner_app_id, password_hash, must_change, password_version = snapshot
        valid = bool(
            status == "ACTIVE" and password_hash
            and await asyncio.to_thread(
                bcrypt.checkpw, password.encode("utf-8"), password_hash.encode("ascii")
            )
        )
        if not valid:
            if is_reserved_global_admin(user_id) and (status is None or password_hash is None):
                raise UserAuthenticationError(
                    "SYSTEM_NOT_INITIALIZED",
                    "系统尚未初始化：ADMIN 用户或登录凭据不存在",
                    status_code=503,
                )
            raise UserAuthenticationError("INVALID_CREDENTIALS", "用户名或密码错误")
        return display_name, origin, owner_app_id, must_change, int(password_version)

    async def platform_login(self, *, user_id: str, password: str) -> dict[str, object]:
        display_name, origin, _, must_change, password_version = await self._verify_credentials(user_id=user_id, password=password)
        if origin != "PLATFORM":
            raise UserAuthenticationError("PLATFORM_ACCOUNT_REQUIRED", "该账号不是平台来源账号", status_code=403)
        return self._token_response(
            user_id=user_id, display_name=display_name, entry_kind="PLATFORM",
            app_id=None, domain_id=None, domain_name=None,
            must_change_password=must_change, password_version=password_version,
        )

    async def list_login_apps(self, *, user_id: str, password: str) -> dict[str, object]:
        display_name, origin, owner_app_id, must_change, _ = await self._verify_credentials(user_id=user_id, password=password)
        async with self._uow_factory() as uow:
            app_ids = await uow.access.list_active_app_ids(user_id)
            apps = {
                row.app_id: (row.display_name, row.status)
                for row in await uow.access.list_applications()
            }
        if origin == "APP":
            app_ids = (
                (owner_app_id,)
                if owner_app_id is not None and owner_app_id in app_ids
                else ()
            )
        app_items = [
            {"app_id": app_id, "display_name": apps[app_id][0], "status": apps[app_id][1]}
            for app_id in app_ids if app_id in apps and apps[app_id][1] == "ACTIVE"
        ]
        return {
            "user_id": user_id, "display_name": display_name,
            "account_origin": origin, "must_change_password": must_change,
            "apps": app_items,
        }

    async def list_login_domains(self, *, user_id: str, password: str, app_id: str) -> dict[str, object]:
        display_name, origin, owner_app_id, must_change, _ = await self._verify_credentials(user_id=user_id, password=password)
        if origin == "APP" and owner_app_id != app_id:
            raise UserAuthenticationError("APP_ACCESS_DENIED", "App 用户不能进入其他 App", status_code=403)
        async with self._uow_factory() as uow:
            domain_ids = await uow.access.list_active_domain_ids(user_id, app_id=app_id)
            domains = await uow.domains.list_by_ids(domain_ids=domain_ids)
            domain_items = [
                {"domain_id": int(row.domain_id), "name": row.name, "status": row.status}
                for row in domains if row.status == "ACTIVE"
            ]
        return {
            "user_id": user_id, "display_name": display_name, "app_id": app_id,
            "must_change_password": must_change,
            "domains": domain_items,
        }

    async def app_login(self, *, user_id: str, password: str, app_id: str, domain_id: int) -> dict[str, object]:
        display_name, origin, owner_app_id, must_change, password_version = await self._verify_credentials(user_id=user_id, password=password)
        if origin == "APP" and owner_app_id != app_id:
            raise UserAuthenticationError("APP_ACCESS_DENIED", "App 用户不能进入其他 App", status_code=403)
        async with self._uow_factory() as uow:
            app = await uow.access.get_application(app_id)
            domain_ids = await uow.access.list_active_domain_ids(user_id, app_id=app_id)
            domain = await uow.domains.get(domain_id=domain_id)
            app_status = app.status if app is not None else None
            domain_snapshot = (domain.name, domain.status) if domain is not None else None
        if app_status != "ACTIVE":
            raise UserAuthenticationError("APP_DISABLED", "App 不存在或已停用", status_code=403)
        if domain_snapshot is None or domain_snapshot[1] != "ACTIVE" or domain_id not in domain_ids:
            raise UserAuthenticationError("DOMAIN_ACCESS_DENIED", "用户没有所选 App Domain 的有效访问权限", status_code=403)
        return self._token_response(
            user_id=user_id, display_name=display_name, entry_kind="BUSINESS",
            app_id=app_id, domain_id=domain_id, domain_name=domain_snapshot[0],
            must_change_password=must_change, password_version=password_version,
        )

    async def login_for_domain_name(self, *, user_id: str, password: str, domain_name: str, app_id: str = "km_asset") -> dict[str, object]:
        async with self._uow_factory() as uow:
            domain = await uow.domains.get_by_name(name=domain_name)
            domain_id = int(domain.domain_id) if domain and domain.status == "ACTIVE" else None
        if domain_id is None:
            raise UserAuthenticationError("DOMAIN_UNAVAILABLE", "固定 Domain 尚未初始化或未启用")
        return await self.app_login(user_id=user_id, password=password, app_id=app_id, domain_id=domain_id)

    def _token_response(
        self, *, user_id: str, display_name: str | None, entry_kind: str,
        app_id: str | None, domain_id: int | None, domain_name: str | None,
        must_change_password: bool, password_version: int,
    ) -> dict[str, object]:
        token, expires_at = self._codec.issue(
            user_id=user_id, entry_kind=entry_kind, app_id=app_id, domain_id=domain_id,
            must_change_password=must_change_password, password_version=password_version,
        )
        return {
            "access_token": token, "token_type": "Bearer", "expires_at": expires_at,
            "user_id": user_id, "display_name": display_name, "entry_kind": entry_kind,
            "app_id": app_id, "domain_id": domain_id, "domain_name": domain_name,
            "must_change_password": must_change_password,
        }

    async def profile(self, *, claims: UserTokenClaims) -> dict[str, object]:
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(claims.user_id)
            memberships = await uow.access.list_user_memberships(user_id=claims.user_id)
            user_snapshot = (
                user.user_id, user.display_name, user.account_origin,
                user.owner_app_id, user.status,
            ) if user is not None else None
            membership_items = [
                {"app_id": row.app_id, "status": row.status, "member_source": row.member_source, "initial_admin": row.is_initial_admin == "Y"}
                for row in memberships
            ]
        if user_snapshot is None or user_snapshot[4] != "ACTIVE":
            raise UserAuthenticationError("USER_DISABLED", "用户不存在或已停用")
        return {
            "user_id": user_snapshot[0], "display_name": user_snapshot[1],
            "account_origin": user_snapshot[2], "owner_app_id": user_snapshot[3],
            "status": user_snapshot[4], "entry_kind": claims.entry_kind,
            "app_id": claims.app_id, "domain_id": claims.domain_id,
            "must_change_password": claims.must_change_password,
            "app_memberships": membership_items,
        }

    async def switch_domain(self, *, claims: UserTokenClaims, domain_id: int) -> dict[str, object]:
        if claims.entry_kind != "BUSINESS" or not claims.app_id:
            raise UserAuthenticationError("BUSINESS_SESSION_REQUIRED", "平台会话不能切换 Domain", status_code=409)
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(claims.user_id)
            domain_ids = await uow.access.list_active_domain_ids(claims.user_id, app_id=claims.app_id)
            domain = await uow.domains.get(domain_id=domain_id)
            user_snapshot = (user.user_id, user.display_name, user.status) if user else None
            domain_snapshot = (domain.name, domain.status) if domain else None
        if user_snapshot is None or user_snapshot[2] != "ACTIVE" or domain_snapshot is None or domain_snapshot[1] != "ACTIVE" or domain_id not in domain_ids:
            raise UserAuthenticationError("DOMAIN_ACCESS_DENIED", "用户没有所选 App Domain 的有效访问权限")
        return self._token_response(
            user_id=user_snapshot[0], display_name=user_snapshot[1], entry_kind="BUSINESS",
            app_id=claims.app_id, domain_id=domain_id, domain_name=domain_snapshot[0],
            must_change_password=claims.must_change_password,
            password_version=claims.password_version,
        )

    async def change_password(self, *, claims: UserTokenClaims, current_password: str, new_password: str) -> dict[str, object]:
        if current_password == new_password:
            raise UserAuthenticationError("PASSWORD_REUSED", "新密码不能与当前密码相同")
        async with self._uow_factory() as uow:
            credential = await uow.access.get_user_credential(claims.user_id)
            valid = bool(credential and await asyncio.to_thread(
                bcrypt.checkpw, current_password.encode("utf-8"), credential.password_hash.encode("ascii")
            ))
            if not valid:
                raise UserAuthenticationError("INVALID_CREDENTIALS", "当前密码错误")
            password_hash = await asyncio.to_thread(
                bcrypt.hashpw, new_password.encode("utf-8"), bcrypt.gensalt(rounds=12)
            )
            await uow.access.set_user_password(credential=credential, password_hash=password_hash.decode("ascii"))
            user = await uow.access.get_user(claims.user_id)
            domain = await uow.domains.get(domain_id=claims.domain_id) if claims.domain_id is not None else None
            display_name = user.display_name if user else None
            domain_name = domain.name if domain else None
            password_version = self._timestamp_version(credential.password_updated_at)
            await uow.commit()
        return self._token_response(
            user_id=claims.user_id, display_name=display_name,
            entry_kind=claims.entry_kind, app_id=claims.app_id, domain_id=claims.domain_id,
            domain_name=domain_name, must_change_password=False,
            password_version=password_version,
        )

    async def validate_session(self, *, claims: UserTokenClaims) -> None:
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(claims.user_id)
            credential = await uow.access.get_user_credential(claims.user_id)
            current_version = self._timestamp_version(credential.password_updated_at) if credential else None
            user_status = user.status if user else None
            account_origin = user.account_origin if user else None
            if claims.entry_kind == "BUSINESS" and claims.app_id and claims.domain_id is not None:
                domain_ids = await uow.access.list_active_domain_ids(claims.user_id, app_id=claims.app_id)
            else:
                domain_ids = ()
        if user_status != "ACTIVE":
            raise UserAuthenticationError("USER_DISABLED", "用户不存在或已停用")
        if claims.entry_kind == "PLATFORM" and account_origin != "PLATFORM":
            raise UserAuthenticationError("PLATFORM_ACCOUNT_REQUIRED", "App 用户不能使用平台会话")
        if claims.entry_kind == "BUSINESS" and claims.domain_id not in domain_ids:
            raise UserAuthenticationError("DOMAIN_ACCESS_DENIED", "用户已失去当前 App Domain 的访问权限")
        if current_version != claims.password_version:
            raise UserAuthenticationError("USER_SESSION_REVOKED", "密码已更新，请重新登录")

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
            principal_kind=PrincipalKind.PORTAL, client_id="user-session",
            api_key_id="user-jwt",
            entry_kind=IdentityEntryKind(claims.entry_kind), app_id=claims.app_id,
            domain_id=str(claims.domain_id) if claims.domain_id is not None else None,
            asserted_user_id=claims.user_id, request_id=str(uuid4()), trace_id=str(uuid4()),
        )

    def verify(self, authorization: str | None) -> UserTokenClaims:
        return self._codec.verify_authorization(authorization)


def create_user_token_codec(*, settings) -> UserTokenCodec:
    """从部署 Secret 创建用户令牌编解码器。"""
    env_name = settings.security.user_jwt_secret_env
    secret = os.getenv(env_name)
    if not secret:
        raise RuntimeError(f"平台用户 JWT 密钥环境变量 {env_name} 未设置")
    return UserTokenCodec(secret=secret, issuer=settings.security.user_jwt_issuer, ttl_seconds=settings.security.user_jwt_ttl_seconds)


__all__ = [
    "KM_PORTAL_DOMAIN_NAME", "UserAuthenticationError", "UserAuthService",
    "UserTokenClaims", "UserTokenCodec", "create_user_token_codec",
]
