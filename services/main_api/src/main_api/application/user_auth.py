"""统一工作台与 App 登录、密码更新及短期令牌。"""

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
from main_api.application.access_control import MODEL_MANAGEMENT_PERMISSION, is_reserved_global_admin
from main_api.application.domain_management import DomainConflictError, DomainLifecycleError, DomainManagementService


_ALGORITHM = "HS256"
_AUDIENCE = "kbot-public-api"
_TOKEN_TYPE = "kbot-user+jwt"
_KNOWLEDGE_ENTRY_AUDIENCE = "kbot-knowledge-retrieval-entry"
_KNOWLEDGE_ENTRY_TTL_SECONDS = 300
_KNOWLEDGE_APP = "knowledge_retrieval"
_ROOT_ACCOUNT_ORIGINS = {"PLATFORM": "PLATFORM", "WORKSPACE": "APP"}
_BCRYPT_PASSWORD_MAX_BYTES = 72
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


@dataclass(frozen=True, slots=True)
class KnowledgeEntryClaims:
    user_id: str
    password_version: int


class UserTokenCodec:
    """签发并校验工作台根会话和业务入口的用户 JWT。"""

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
        if entry_kind not in _ROOT_ACCOUNT_ORIGINS and entry_kind != "BUSINESS":
            raise ValueError("入口类型无效")
        if entry_kind in _ROOT_ACCOUNT_ORIGINS and (app_id is not None or domain_id is not None):
            raise ValueError("工作台根会话不能包含 App 或 Domain")
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
            if entry_kind not in _ROOT_ACCOUNT_ORIGINS and entry_kind != "BUSINESS":
                raise ValueError("入口类型无效")
            if entry_kind in _ROOT_ACCOUNT_ORIGINS and (app_id or domain_id is not None):
                raise ValueError("工作台根会话上下文无效")
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

    def issue_knowledge_entry(self, *, user_id: str, password_version: int, action: str) -> str:
        """签发只用于知识检索入口的短期票据，不能作为业务 API 的 Bearer Token。"""
        now = int(datetime.now(timezone.utc).timestamp())
        return jwt.encode({
            "iss": self._issuer, "aud": _KNOWLEDGE_ENTRY_AUDIENCE, "sub": user_id,
            "action": action, "pwd": password_version,
            "iat": now, "exp": now + _KNOWLEDGE_ENTRY_TTL_SECONDS,
        }, self._secret, algorithm=_ALGORITHM)

    def verify_knowledge_entry(self, entry_ticket: str, *, action: str) -> KnowledgeEntryClaims:
        try:
            claims = jwt.decode(
                entry_ticket, self._secret, algorithms=[_ALGORITHM],
                audience=_KNOWLEDGE_ENTRY_AUDIENCE, issuer=self._issuer,
                options={"require_exp": True, "require_iss": True, "require_aud": True,
                         "require_iat": True, "require_sub": True},
            )
            if claims["action"] != action or not claims["sub"]:
                raise ValueError("票据用途不匹配")
            return KnowledgeEntryClaims(user_id=claims["sub"], password_version=int(claims["pwd"]))
        except ExpiredSignatureError as exc:
            raise UserAuthenticationError("ENTRY_TICKET_EXPIRED", "入口已过期，请重新登录") from exc
        except (JWTError, KeyError, TypeError, ValueError) as exc:
            raise UserAuthenticationError("INVALID_ENTRY_TICKET", "入口票据无效，请重新登录") from exc


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
        password_bytes = password.encode("utf-8")
        valid = bool(
            status == "ACTIVE" and password_hash and len(password_bytes) <= _BCRYPT_PASSWORD_MAX_BYTES
            and await asyncio.to_thread(
                bcrypt.checkpw, password_bytes, password_hash.encode("ascii")
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

    async def workspace_login(self, *, user_id: str, password: str) -> dict[str, object]:
        """只校验一次凭据，由服务端选择对应账号来源的根会话。"""
        display_name, origin, _, must_change, password_version = await self._verify_credentials(
            user_id=user_id, password=password,
        )
        if origin not in _ROOT_ACCOUNT_ORIGINS.values():
            raise UserAuthenticationError("INVALID_ACCOUNT_ORIGIN", "账号来源无效", status_code=403)
        return self._token_response(
            user_id=user_id, display_name=display_name,
            entry_kind="PLATFORM" if origin == "PLATFORM" else "WORKSPACE",
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
        return await self._issue_business_session(
            user_id=user_id,
            display_name=display_name,
            app_id=app_id,
            domain_id=domain_id,
            must_change_password=must_change,
            password_version=password_version,
        )

    async def _issue_business_session(
        self,
        *,
        user_id: str,
        display_name: str | None,
        app_id: str,
        domain_id: int,
        must_change_password: bool,
        password_version: int,
    ) -> dict[str, object]:
        """在已认证用户的显式 App 与 Domain 授权范围内签发业务会话。"""
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
            must_change_password=must_change_password,
            password_version=password_version,
        )

    async def exchange_app_session(
        self, *, claims: UserTokenClaims, app_id: str, domain_id: int
    ) -> dict[str, object]:
        """以根会话换取业务会话，不接受隐式平台管理员权限。"""
        await self._require_root_session(claims)
        async with self._uow_factory() as uow:
            user = await self._get_root_user(uow=uow, claims=claims)
            self._require_app_owner_scope(user, app_id=app_id)
            display_name = user.display_name
        return await self._issue_business_session(
            user_id=claims.user_id,
            display_name=display_name,
            app_id=app_id,
            domain_id=domain_id,
            must_change_password=claims.must_change_password,
            password_version=claims.password_version,
        )

    async def _require_root_session(self, claims: UserTokenClaims) -> None:
        if claims.entry_kind not in _ROOT_ACCOUNT_ORIGINS:
            raise UserAuthenticationError(
                "WORKSPACE_SESSION_REQUIRED", "只有工作台根会话可以发现或进入应用入口", status_code=409,
            )
        await self.validate_session(claims=claims)
        if claims.must_change_password:
            raise UserAuthenticationError("PASSWORD_CHANGE_REQUIRED", "首次登录必须先修改密码", status_code=403)

    @staticmethod
    async def _get_root_user(*, uow, claims: UserTokenClaims):
        user = await uow.access.get_user(claims.user_id)
        if user is None or user.status != "ACTIVE":
            raise UserAuthenticationError("USER_DISABLED", "用户不存在或已停用")
        if user.account_origin != _ROOT_ACCOUNT_ORIGINS.get(claims.entry_kind):
            raise UserAuthenticationError("USER_SESSION_REVOKED", "登录会话已失效，请重新登录")
        return user

    @staticmethod
    def _require_app_owner_scope(user, *, app_id: str) -> None:
        if user.account_origin == "APP" and user.owner_app_id != app_id:
            raise UserAuthenticationError("APP_ACCESS_DENIED", "App 用户不能进入其他 App", status_code=403)

    async def create_knowledge_first_domain(
        self, *, claims: UserTokenClaims, name: str, description: str | None,
    ) -> dict[str, object]:
        """用根会话开通知识检索首域，复用既有业务管理员授权与事务条件。"""
        await self._require_root_session(claims)
        async with self._uow_factory() as uow:
            user = await self._get_root_user(uow=uow, claims=claims)
            self._require_app_owner_scope(user, app_id=_KNOWLEDGE_APP)
        try:
            domain = await DomainManagementService(uow_factory=self._uow_factory).create_for_app(
                app_id=_KNOWLEDGE_APP, name=name, description=description,
                actor_id=claims.user_id, first_domain=True,
            )
        except DomainConflictError as exc:
            raise UserAuthenticationError("DOMAIN_NAME_CONFLICT", str(exc), status_code=409) from exc
        except DomainLifecycleError as exc:
            raise UserAuthenticationError(exc.code, str(exc), status_code=exc.status_code) from exc
        return await self.exchange_app_session(
            claims=claims, app_id=_KNOWLEDGE_APP, domain_id=domain["domain_id"],
        )

    async def refresh_session(
        self, *, claims: UserTokenClaims
    ) -> dict[str, object]:
        """续签仍有效的当前入口会话，并再次校验账号和授权状态。"""
        await self.validate_session(claims=claims)
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(claims.user_id)
            domain = (
                await uow.domains.get(domain_id=claims.domain_id)
                if claims.domain_id is not None
                else None
            )
            user_snapshot = (
                user.display_name,
                user.status,
            ) if user is not None else None
            domain_name = domain.name if domain is not None else None
        if user_snapshot is None or user_snapshot[1] != "ACTIVE":
            raise UserAuthenticationError(
                "USER_DISABLED", "用户不存在或已停用"
            )
        return self._token_response(
            user_id=claims.user_id,
            display_name=user_snapshot[0],
            entry_kind=claims.entry_kind,
            app_id=claims.app_id,
            domain_id=claims.domain_id,
            domain_name=domain_name,
            must_change_password=claims.must_change_password,
            password_version=claims.password_version,
        )

    async def login_for_domain_name(self, *, user_id: str, password: str, domain_name: str, app_id: str) -> dict[str, object]:
        async with self._uow_factory() as uow:
            domain = await uow.domains.get_by_name(name=domain_name)
            domain_id = int(domain.domain_id) if domain and domain.status == "ACTIVE" else None
        if domain_id is None:
            raise UserAuthenticationError("DOMAIN_UNAVAILABLE", "固定 Domain 尚未初始化或未启用")
        return await self.app_login(user_id=user_id, password=password, app_id=app_id, domain_id=domain_id)

    async def begin_knowledge_entry(self, *, user_id: str, password: str) -> dict[str, object]:
        """认证后按当前授权决定直接进入、选域或创建首域。"""
        display_name, origin, owner_app_id, must_change, password_version = await self._verify_credentials(
            user_id=user_id, password=password,
        )
        self._check_knowledge_account(origin=origin, owner_app_id=owner_app_id)
        async with self._uow_factory() as uow:
            app = await uow.access.get_application(_KNOWLEDGE_APP)
            if app is None or app.status != "ACTIVE":
                raise UserAuthenticationError("APP_DISABLED", "知识检索 App 不存在或已停用", status_code=403)
            ids = await uow.access.list_active_domain_ids(user_id, app_id=_KNOWLEDGE_APP)
            domains = [
                {"domain_id": int(row.domain_id), "name": row.name}
                for row in await uow.domains.list_by_ids(domain_ids=ids)
                if row.status == "ACTIVE"
            ]
            if not domains:
                try:
                    await DomainManagementService.require_knowledge_first_domain_creator(uow=uow, actor_id=user_id)
                except DomainLifecycleError as exc:
                    raise UserAuthenticationError("DOMAIN_ACCESS_DENIED", "没有可进入的工作域，请联系知识检索管理员", status_code=403) from exc
        if len(domains) == 1:
            return await self._issue_knowledge_session(
                user_id=user_id, display_name=display_name, domain_id=domains[0]["domain_id"],
                must_change_password=must_change, password_version=password_version,
            )
        action = "SELECT_DOMAIN" if domains else ("CHANGE_PASSWORD" if must_change else "CREATE_FIRST_DOMAIN")
        return self._knowledge_challenge(
            user_id=user_id, password_version=password_version, action=action, domains=domains,
        )

    def _knowledge_challenge(
        self, *, user_id: str, password_version: int, action: str,
        domains: list[dict[str, int | str]],
    ) -> dict[str, object]:
        return {
            "action": action, "domains": domains,
            "entry_ticket": self._codec.issue_knowledge_entry(
                user_id=user_id, password_version=password_version, action=action,
            ),
        }

    @staticmethod
    def _check_knowledge_account(*, origin: str, owner_app_id: str | None) -> None:
        if origin == "APP" and owner_app_id != _KNOWLEDGE_APP:
            raise UserAuthenticationError("APP_ACCESS_DENIED", "App 用户不能进入其他 App", status_code=403)

    async def _issue_knowledge_session(
        self, *, user_id: str, display_name: str | None, domain_id: int,
        must_change_password: bool, password_version: int,
    ) -> dict[str, object]:
        business = await self._issue_business_session(
            user_id=user_id, display_name=display_name, app_id=_KNOWLEDGE_APP, domain_id=domain_id,
            must_change_password=must_change_password, password_version=password_version,
        )
        return {"action": "ENTERED", **business}

    async def complete_knowledge_entry(
        self, *, entry_ticket: str, action: str, domain_id: int | None = None,
        name: str | None = None, description: str | None = None,
        current_password: str | None = None, new_password: str | None = None,
    ) -> dict[str, object]:
        claims = self._codec.verify_knowledge_entry(entry_ticket, action=action)
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(claims.user_id)
            credential = await uow.access.get_user_credential(claims.user_id)
            if user is None or user.status != "ACTIVE":
                raise UserAuthenticationError("USER_DISABLED", "用户不存在或已停用", status_code=403)
            if credential is None or self._timestamp_version(credential.password_updated_at) != claims.password_version:
                raise UserAuthenticationError("USER_SESSION_REVOKED", "密码已更新，请重新登录")
            self._check_knowledge_account(origin=user.account_origin, owner_app_id=user.owner_app_id)
            display_name = user.display_name
            must_change = credential.must_change_password == "Y"
            if action == "CHANGE_PASSWORD" and current_password is not None and new_password is not None:
                await DomainManagementService.require_knowledge_first_domain_creator(uow=uow, actor_id=user.user_id)
                version = await self._update_password(
                    uow=uow, credential=credential, current_password=current_password, new_password=new_password,
                )
                await uow.commit()
                return self._knowledge_challenge(
                    user_id=claims.user_id, password_version=version, action="CREATE_FIRST_DOMAIN", domains=[],
                )
        if action == "CREATE_FIRST_DOMAIN" and name is not None:
            if must_change:
                raise UserAuthenticationError("PASSWORD_CHANGE_REQUIRED", "请先更新初始密码再创建工作域", status_code=403)
            domain = await DomainManagementService(uow_factory=self._uow_factory).create_for_app(
                app_id=_KNOWLEDGE_APP, name=name, description=description,
                actor_id=claims.user_id, first_domain=True,
            )
            domain_id = domain["domain_id"]
        elif action != "SELECT_DOMAIN" or domain_id is None:
            raise UserAuthenticationError("INVALID_ENTRY_ACTION", "工作域入口请求无效", status_code=422)
        return await self._issue_knowledge_session(
            user_id=claims.user_id, display_name=display_name, domain_id=domain_id,
            must_change_password=must_change, password_version=claims.password_version,
        )

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

    async def list_session_entries(
        self, *, claims: UserTokenClaims
    ) -> dict[str, object]:
        """使用有效会话列出用户当前可进入的 App 与 Domain。"""
        await self._require_root_session(claims)
        async with self._uow_factory() as uow:
            user = await self._get_root_user(uow=uow, claims=claims)
            display_name, origin = user.display_name, user.account_origin
            app_ids = set(await uow.access.list_active_app_ids(claims.user_id))
            # 首次部署没有 App-Domain 绑定时，入口仍应对有资格的管理员可见。
            app_ids.add(_KNOWLEDGE_APP)
            if origin == "APP":
                app_ids.intersection_update({user.owner_app_id})
            can_manage_models = origin == "PLATFORM" and MODEL_MANAGEMENT_PERMISSION in (
                await uow.access.platform_permissions_for(user_id=claims.user_id)
            )
            entries: list[dict[str, object]] = []
            for application in await uow.access.list_applications():
                app_id = application.app_id
                if app_id not in app_ids or application.status != "ACTIVE":
                    continue
                domain_ids = await uow.access.list_active_domain_ids(
                    claims.user_id, app_id=app_id
                )
                domains = await uow.domains.list_by_ids(
                    domain_ids=domain_ids
                )
                active_domains = [
                    {"domain_id": int(domain.domain_id), "name": domain.name, "status": domain.status}
                    for domain in domains
                    if domain.status == "ACTIVE"
                ]
                can_create_first_domain = False
                if not active_domains and app_id == _KNOWLEDGE_APP:
                    try:
                        await DomainManagementService.require_knowledge_first_domain_creator(
                            uow=uow, actor_id=claims.user_id,
                        )
                    except DomainLifecycleError:
                        pass
                    else:
                        can_create_first_domain = True
                if active_domains or can_create_first_domain:
                    entries.append({
                        "app_id": app_id,
                        "display_name": application.display_name,
                        "domains": active_domains,
                        "can_create_first_domain": can_create_first_domain,
                    })
        return {
            "user_id": claims.user_id,
            "display_name": display_name,
            "account_origin": origin,
            "apps": entries,
            "system_management": can_manage_models,
        }

    async def list_session_domains(self, *, claims: UserTokenClaims) -> dict[str, object]:
        """列出当前业务会话所属 App 中已授权且启用的工作域，无需管理权限。"""
        if claims.entry_kind != "BUSINESS" or not claims.app_id or claims.domain_id is None:
            raise UserAuthenticationError(
                "BUSINESS_SESSION_REQUIRED", "只有业务会话可以读取可切换工作域", status_code=409,
            )
        async with self._uow_factory() as uow:
            domain_ids = await uow.access.list_active_domain_ids(claims.user_id, app_id=claims.app_id)
            domains = await uow.domains.list_by_ids(domain_ids=domain_ids)
            items = [
                {"domain_id": int(row.domain_id), "name": row.name, "status": row.status}
                for row in domains if row.status == "ACTIVE"
            ]
        return {"app_id": claims.app_id, "domain_id": claims.domain_id, "domains": items}

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
        async with self._uow_factory() as uow:
            credential = await uow.access.get_user_credential(claims.user_id)
            password_version = await self._update_password(
                uow=uow, credential=credential, current_password=current_password, new_password=new_password,
            )
            user = await uow.access.get_user(claims.user_id)
            domain = await uow.domains.get(domain_id=claims.domain_id) if claims.domain_id is not None else None
            display_name = user.display_name if user else None
            domain_name = domain.name if domain else None
            await uow.commit()
        return self._token_response(
            user_id=claims.user_id, display_name=display_name,
            entry_kind=claims.entry_kind, app_id=claims.app_id, domain_id=claims.domain_id,
            domain_name=domain_name, must_change_password=False,
            password_version=password_version,
        )

    async def _update_password(self, *, uow, credential, current_password: str, new_password: str) -> int:
        if current_password == new_password:
            raise UserAuthenticationError("PASSWORD_REUSED", "新密码不能与当前密码相同")
        current_bytes, new_bytes = current_password.encode("utf-8"), new_password.encode("utf-8")
        if len(new_bytes) > _BCRYPT_PASSWORD_MAX_BYTES:
            raise UserAuthenticationError("PASSWORD_TOO_LONG", "新密码不能超过 72 字节", status_code=422)
        valid = bool(credential and len(current_bytes) <= _BCRYPT_PASSWORD_MAX_BYTES and await asyncio.to_thread(
            bcrypt.checkpw, current_bytes, credential.password_hash.encode("ascii")
        ))
        if not valid:
            raise UserAuthenticationError("INVALID_CREDENTIALS", "当前密码错误")
        password_hash = await asyncio.to_thread(
            bcrypt.hashpw, new_bytes, bcrypt.gensalt(rounds=12)
        )
        await uow.access.set_user_password(credential=credential, password_hash=password_hash.decode("ascii"))
        return self._timestamp_version(credential.password_updated_at)

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
        if claims.entry_kind == "WORKSPACE" and account_origin != "APP":
            raise UserAuthenticationError("USER_SESSION_REVOKED", "登录会话已失效，请重新登录")
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
        if token.startswith(("kbot_sk_", "kbot_ak_")):
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
