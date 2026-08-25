"""App 独立 API Client 的签发、轮换、验证与授权。"""

from __future__ import annotations

import hmac
import re
import secrets
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from uuid import UUID

from main_api.entities import AppApiClientEntity, AppApiCredentialEntity
from platform_core.contracts import (
    AuthContext,
    IdentityEntryKind,
    PrincipalKind,
)
from platform_core.identity import uuid7
from platform_core.security import digest_portal_api_key, extract_bearer_token


APP_API_KEY_PREFIX = "kbot_ak_"
APP_URL_SLUGS = {
    "knowledge_retrieval": "knowledge-retrieval",
    "km_asset": "km-asset",
    "aiops": "aiops",
}
APP_API_SCOPE_PERMISSIONS = {
    "knowledge_retrieval": {
        "knowledge:agent:read": "knowledge_retrieval:use",
        "knowledge:chat:write": "knowledge_retrieval:use",
        "knowledge:conversation:read": "knowledge_retrieval:use",
        "knowledge:run:read": "knowledge_retrieval:use",
    },
    "km_asset": {
        "km:agent:read": "km_asset:use",
        "km:chat:write": "km_asset:use",
        "km:conversation:read": "km_asset:use",
        "km:conversation:update": "km_asset:use",
        "km:conversation:delete": "km_asset:use",
        "km:run:read": "km_asset:use",
        "km:reference:read": "km_asset:use",
    },
    "aiops": {
        "aiops:agent:read": "aiops:use",
        "aiops:chat:write": "aiops:use",
        "aiops:conversation:read": "aiops:use",
        "aiops:run:read": "aiops:use",
    },
}
_FORBIDDEN_IDENTITY_HEADERS = {
    "x-kbot-user-id",
    "x-kbot-domain-id",
    "x-kbot-tenant-id",
    "x-kbot-app-id",
    "x-kbot-auth-context",
    "x-kbot-internal-token",
}


class AppApiKeyError(ValueError):
    """App API Key 配置、认证或授权失败。"""

    def __init__(self, code: str, message: str, *, status_code: int = 403):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class AppApiKeyService:
    """管理数据库型 App API Key，并生成不可伪造的业务身份上下文。"""

    def __init__(self, *, uow_factory, pepper: str):
        if not pepper:
            raise ValueError("App API Key Pepper 不能为空")
        self._uow_factory = uow_factory
        self._pepper = pepper
        self._requests: dict[UUID, deque[float]] = defaultdict(deque)

    @staticmethod
    def scope_catalog(app_id: str) -> tuple[dict[str, str], ...]:
        return tuple(
            {"scope_code": code, "required_permission": permission}
            for code, permission in sorted(
                APP_API_SCOPE_PERMISSIONS.get(app_id, {}).items()
            )
        )

    async def create_client(
        self,
        *,
        app_id: str,
        domain_id: int,
        subject_user_id: str,
        display_name: str,
        scopes: tuple[str, ...],
        agent_ids: tuple[UUID, ...],
        expires_at: datetime,
        rate_limit_per_minute: int,
        actor_id: str,
    ) -> dict[str, object]:
        normalized_scopes = tuple(sorted(set(scopes)))
        normalized_agents = tuple(sorted(set(agent_ids), key=str))
        self._validate_expiration(expires_at)
        await self._validate_grant(
            app_id=app_id,
            domain_id=domain_id,
            subject_user_id=subject_user_id,
            scopes=normalized_scopes,
            agent_ids=normalized_agents,
        )
        client = AppApiClientEntity(
            client_id=uuid7(),
            app_id=app_id,
            domain_id=domain_id,
            subject_user_id=subject_user_id,
            display_name=display_name,
            status="ACTIVE",
            rate_limit_per_minute=rate_limit_per_minute,
            row_version=1,
            created_by=actor_id,
        )
        raw_key, credential = self._new_credential(
            client_id=client.client_id,
            expires_at=expires_at,
            actor_id=actor_id,
        )
        async with self._uow_factory() as uow:
            await uow.app_api_keys.add_client(client)
            await uow.app_api_keys.replace_scopes(
                client_id=client.client_id, scopes=normalized_scopes
            )
            await uow.app_api_keys.replace_agents(
                client_id=client.client_id, agent_ids=normalized_agents
            )
            await uow.app_api_keys.add_credential(credential)
            client_values = self._client_values(client)
            credential_values = self._credential_values(credential)
            await uow.commit()
        return {
            **self._client_view_values(
                client_values,
                normalized_scopes,
                normalized_agents,
                [credential_values],
            ),
            "credential_id": str(credential.credential_id),
            "api_key": raw_key,
            "warning": "该密钥只显示一次，关闭后无法恢复",
        }

    async def list_clients(
        self, *, app_id: str, domain_id: int
    ) -> list[dict[str, object]]:
        async with self._uow_factory() as uow:
            clients = await uow.app_api_keys.list_clients(
                app_id=app_id, domain_id=domain_id
            )
            snapshots = []
            for client in clients:
                values = self._client_values(client)
                scopes = await uow.app_api_keys.list_scopes(
                    client_id=client.client_id
                )
                agents = await uow.app_api_keys.list_agents(
                    client_id=client.client_id
                )
                credentials = await uow.app_api_keys.list_credentials(
                    client_id=client.client_id
                )
                snapshots.append((values, scopes, agents, [
                    self._credential_values(row) for row in credentials
                ]))
        return [
            self._client_view_values(*snapshot) for snapshot in snapshots
        ]

    async def get_client(
        self, *, app_id: str, domain_id: int, client_id: UUID
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            client = await uow.app_api_keys.get_client(client_id)
            if (
                client is None
                or client.app_id != app_id
                or int(client.domain_id) != domain_id
            ):
                raise AppApiKeyError(
                    "APP_API_CLIENT_NOT_FOUND", "API Client 不存在", status_code=404
                )
            values = self._client_values(client)
            scopes = await uow.app_api_keys.list_scopes(client_id=client_id)
            agents = await uow.app_api_keys.list_agents(client_id=client_id)
            credentials = [
                self._credential_values(row)
                for row in await uow.app_api_keys.list_credentials(
                    client_id=client_id
                )
            ]
        return self._client_view_values(values, scopes, agents, credentials)

    async def set_status(
        self, *, app_id: str, domain_id: int, client_id: UUID, status: str
    ) -> dict[str, object]:
        if status != "DISABLED":
            raise AppApiKeyError(
                "APP_API_CLIENT_STATUS_INVALID",
                "已停用的 API Client 不能重新启用，请创建新的 API Client",
                status_code=422,
            )
        async with self._uow_factory() as uow:
            client = await uow.app_api_keys.get_client(client_id)
            if (
                client is None
                or client.app_id != app_id
                or int(client.domain_id) != domain_id
            ):
                raise AppApiKeyError(
                    "APP_API_CLIENT_NOT_FOUND", "API Client 不存在", status_code=404
                )
            client.status = status
            client.row_version = int(client.row_version) + 1
            client.updated_at = datetime.now(timezone.utc)
            if status == "DISABLED":
                await uow.app_api_keys.revoke_active_credentials(
                    client_id=client_id
                )
            await uow.commit()
        return await self.get_client(
            app_id=app_id, domain_id=domain_id, client_id=client_id
        )

    async def rotate(
        self,
        *,
        app_id: str,
        domain_id: int,
        client_id: UUID,
        expires_at: datetime,
        actor_id: str,
    ) -> dict[str, object]:
        self._validate_expiration(expires_at)
        async with self._uow_factory() as uow:
            client = await uow.app_api_keys.get_client(client_id)
            if (
                client is None
                or client.app_id != app_id
                or int(client.domain_id) != domain_id
            ):
                raise AppApiKeyError(
                    "APP_API_CLIENT_NOT_FOUND", "API Client 不存在", status_code=404
                )
            if client.status != "ACTIVE":
                raise AppApiKeyError(
                    "APP_API_CLIENT_DISABLED", "已停用的 API Client 不能轮换密钥"
                )
            await uow.app_api_keys.revoke_active_credentials(client_id=client_id)
            raw_key, credential = self._new_credential(
                client_id=client_id,
                expires_at=expires_at,
                actor_id=actor_id,
            )
            await uow.app_api_keys.add_credential(credential)
            await uow.commit()
        return {
            "client_id": str(client_id),
            "credential_id": str(credential.credential_id),
            "api_key": raw_key,
            "expires_at": expires_at,
            "warning": "该密钥只显示一次，旧密钥已立即撤销",
        }

    async def authenticate_request(
        self, *, authorization: str | None, path: str, headers
    ) -> AuthContext | None:
        if not authorization:
            return None
        raw_key = extract_bearer_token(authorization)
        if not raw_key.startswith(APP_API_KEY_PREFIX):
            return None
        forbidden = sorted(
            name for name in _FORBIDDEN_IDENTITY_HEADERS if headers.get(name)
        )
        if forbidden:
            raise AppApiKeyError(
                "APP_API_KEY_IDENTITY_HEADER_FORBIDDEN",
                "App API Key 请求不得提交用户、App、Domain 或内部身份 Header",
                status_code=400,
            )
        public_key_id = self._extract_public_id(raw_key)
        async with self._uow_factory() as uow:
            credential = await uow.app_api_keys.get_credential_by_public_id(
                public_key_id
            )
            if credential is None:
                raise AppApiKeyError(
                    "INVALID_APP_API_KEY", "App API Key 无效", status_code=401
                )
            client = await uow.app_api_keys.get_client(credential.client_id)
            if client is None:
                raise AppApiKeyError(
                    "INVALID_APP_API_KEY", "App API Key 无效", status_code=401
                )
            credential_values = self._credential_values(credential)
            client_values = self._client_values(client)
            configured_scopes = await uow.app_api_keys.list_scopes(
                client_id=client.client_id
            )
            agent_ids = await uow.app_api_keys.list_agents(
                client_id=client.client_id
            )
            permissions = await uow.access.permissions_for(
                app_id=client.app_id,
                domain_id=int(client.domain_id),
                user_id=client.subject_user_id,
            )
            self._validate_authentication(
                raw_key=raw_key,
                path=path,
                credential=credential_values,
                client=client_values,
            )
            await uow.app_api_keys.touch_credential(credential)
            await uow.commit()
        mapping = APP_API_SCOPE_PERMISSIONS.get(str(client_values[1]), {})
        effective_scopes = tuple(
            scope for scope in configured_scopes
            if mapping.get(scope) in permissions
        )
        if not effective_scopes:
            raise AppApiKeyError(
                "APP_API_KEY_PERMISSION_REVOKED",
                "API Client 绑定用户已失去所需权限",
            )
        required_scope = self._required_scope(
            app_id=str(client_values[1]), method=str(headers.get(":method") or ""),
            path=path,
        )
        if required_scope is None or required_scope not in effective_scopes:
            raise AppApiKeyError(
                "APP_API_KEY_SCOPE_DENIED",
                "API Client Scope 不允许访问该公开业务接口",
            )
        self._check_rate_limit(
            client_id=client_values[0],
            limit=int(client_values[6]),
        )
        return AuthContext(
            principal_kind=PrincipalKind.APP_API_CLIENT,
            client_id=str(client_values[0]),
            api_key_id=str(credential_values[0]),
            entry_kind=IdentityEntryKind.BUSINESS,
            app_id=str(client_values[1]),
            domain_id=str(client_values[2]),
            asserted_user_id=str(client_values[3]),
            scopes=effective_scopes,
            authorized_agent_ids=agent_ids,
            request_id=secrets.token_hex(16),
            trace_id=secrets.token_hex(16),
        )

    async def _validate_grant(
        self,
        *,
        app_id: str,
        domain_id: int,
        subject_user_id: str,
        scopes: tuple[str, ...],
        agent_ids: tuple[UUID, ...],
    ) -> None:
        catalog = APP_API_SCOPE_PERMISSIONS.get(app_id)
        if not catalog:
            raise AppApiKeyError(
                "APP_API_KEY_UNSUPPORTED_APP", "该 App 尚未开放 API Client"
            )
        if not scopes or any(scope not in catalog for scope in scopes):
            raise AppApiKeyError(
                "APP_API_KEY_SCOPE_INVALID", "API Client Scope 不属于当前 App"
            )
        if not agent_ids:
            raise AppApiKeyError(
                "APP_API_KEY_AGENT_REQUIRED", "API Client 至少需要绑定一个 Agent"
            )
        async with self._uow_factory() as uow:
            permissions = await uow.access.permissions_for(
                app_id=app_id,
                domain_id=domain_id,
                user_id=subject_user_id,
            )
        missing = sorted({catalog[scope] for scope in scopes} - permissions)
        if missing:
            raise AppApiKeyError(
                "APP_API_KEY_SUBJECT_PERMISSION_DENIED",
                "绑定服务账号没有 Scope 所需的 App 权限",
            )

    def _new_credential(
        self, *, client_id: UUID, expires_at: datetime, actor_id: str
    ) -> tuple[str, AppApiCredentialEntity]:
        credential_id = uuid7()
        public_key_id = credential_id.hex
        raw_key = f"{APP_API_KEY_PREFIX}{public_key_id}.{secrets.token_urlsafe(32)}"
        return raw_key, AppApiCredentialEntity(
            credential_id=credential_id,
            client_id=client_id,
            public_key_id=public_key_id,
            key_digest=digest_portal_api_key(raw_key, self._pepper),
            status="ACTIVE",
            expires_at=expires_at,
            created_by=actor_id,
        )

    def _validate_authentication(
        self, *, raw_key: str, path: str, credential: tuple, client: tuple
    ) -> None:
        now = datetime.now(timezone.utc)
        expires_at = credential[4]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if not hmac.compare_digest(
            digest_portal_api_key(raw_key, self._pepper), credential[3]
        ):
            raise AppApiKeyError(
                "INVALID_APP_API_KEY", "App API Key 无效", status_code=401
            )
        if credential[2] != "ACTIVE" or now >= expires_at:
            raise AppApiKeyError(
                "APP_API_KEY_EXPIRED", "App API Key 已撤销或过期", status_code=401
            )
        if client[5] != "ACTIVE":
            raise AppApiKeyError(
                "APP_API_CLIENT_DISABLED", "API Client 已停用", status_code=401
            )
        expected_prefix = (
            f"/api/v1/apps/{APP_URL_SLUGS.get(str(client[1]), '')}/"
        )
        relative = path.removeprefix(expected_prefix)
        forbidden = (
            not relative
            or relative == "access"
            or relative.startswith("auth/")
            or relative.startswith("api-clients")
        )
        if (
            not expected_prefix.endswith("//")
            and path.startswith(expected_prefix)
            and not forbidden
        ):
            return
        raise AppApiKeyError(
            "APP_API_KEY_CONTEXT_MISMATCH",
            "App API Key 只能访问绑定 App 的公开 Main API",
        )

    def _check_rate_limit(self, *, client_id: UUID, limit: int) -> None:
        now = datetime.now(timezone.utc).timestamp()
        bucket = self._requests[client_id]
        while bucket and bucket[0] <= now - 60:
            bucket.popleft()
        if len(bucket) >= limit:
            raise AppApiKeyError(
                "APP_API_KEY_RATE_LIMITED",
                "API Client 请求频率超过限制",
                status_code=429,
            )
        bucket.append(now)

    @staticmethod
    def _required_scope(
        *, app_id: str, method: str, path: str
    ) -> str | None:
        """以公开路径白名单确定机器请求所需 Scope，未登记路径默认拒绝。"""
        slug = APP_URL_SLUGS.get(app_id)
        if not slug:
            return None
        relative = path.removeprefix(f"/api/v1/apps/{slug}/").strip("/")
        method = method.upper()
        rules: dict[str, tuple[tuple[str, str, str], ...]] = {
            "km_asset": (
                ("GET", r"agents(?:/[0-9a-fA-F-]{36})?", "km:agent:read"),
                ("POST", r"conversations", "km:chat:write"),
                ("POST", r"conversations/[0-9a-fA-F-]{36}/turns", "km:chat:write"),
                ("GET", r"conversations(?:/.*)?", "km:conversation:read"),
                ("PATCH", r"conversations/[0-9a-fA-F-]{36}", "km:conversation:update"),
                ("DELETE", r"conversations/[0-9a-fA-F-]{36}", "km:conversation:delete"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/references/.*", "km:reference:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}(?:/(?:result|events))?", "km:run:read"),
            ),
            "knowledge_retrieval": (
                ("GET", r"agents(?:/[0-9a-fA-F-]{36})?", "knowledge:agent:read"),
                ("POST", r"runs(?:/[0-9a-fA-F-]{36}/cancel)?", "knowledge:chat:write"),
                ("GET", r"runs/[0-9a-fA-F-]{36}(?:/.*)?", "knowledge:run:read"),
                ("POST", r"conversations", "knowledge:chat:write"),
                ("POST", r"conversations/[0-9a-fA-F-]{36}/turns(?:/multipart)?", "knowledge:chat:write"),
                ("GET", r"conversations(?:/.*)?", "knowledge:conversation:read"),
                ("PATCH", r"conversations/[0-9a-fA-F-]{36}", "knowledge:chat:write"),
                ("DELETE", r"conversations/[0-9a-fA-F-]{36}", "knowledge:chat:write"),
                ("GET", r"memories", "knowledge:conversation:read"),
            ),
            "aiops": (
                ("GET", r"agents(?:/[0-9a-fA-F-]{36})?", "aiops:agent:read"),
                ("POST", r"conversations(?:/.*)?", "aiops:chat:write"),
                ("GET", r"conversations(?:/.*)?", "aiops:conversation:read"),
                ("POST", r"runs(?:/[0-9a-fA-F-]{36}/cancel)?", "aiops:chat:write"),
                ("GET", r"runs/[0-9a-fA-F-]{36}(?:/.*)?", "aiops:run:read"),
                ("GET", r"reports(?:/[0-9a-fA-F-]{36})?", "aiops:run:read"),
            ),
        }
        for expected_method, pattern, scope in rules.get(app_id, ()):
            if method == expected_method and re.fullmatch(pattern, relative):
                return scope
        return None

    @staticmethod
    def _extract_public_id(raw_key: str) -> str:
        identifier, separator, secret = raw_key[len(APP_API_KEY_PREFIX):].partition(".")
        if separator != "." or len(identifier) != 32 or len(secret) < 32:
            raise AppApiKeyError(
                "INVALID_APP_API_KEY", "App API Key 格式无效", status_code=401
            )
        try:
            UUID(hex=identifier)
        except ValueError as exc:
            raise AppApiKeyError(
                "INVALID_APP_API_KEY", "App API Key 格式无效", status_code=401
            ) from exc
        return identifier

    @staticmethod
    def _validate_expiration(expires_at: datetime) -> None:
        if not isinstance(expires_at, datetime):
            raise AppApiKeyError(
                "APP_API_KEY_EXPIRY_INVALID", "密钥过期时间不能为空", status_code=422
            )
        value = expires_at
        if value.tzinfo is None:
            raise AppApiKeyError(
                "APP_API_KEY_EXPIRY_INVALID", "密钥过期时间必须包含时区", status_code=422
            )
        now = datetime.now(timezone.utc)
        try:
            maximum = now.replace(year=now.year + 100)
        except ValueError:
            maximum = now.replace(year=now.year + 100, day=28)
        maximum += timedelta(days=1)
        if value <= now + timedelta(minutes=5) or value > maximum:
            raise AppApiKeyError(
                "APP_API_KEY_EXPIRY_INVALID",
                "密钥有效期必须在 5 分钟至 100 年之间",
                status_code=422,
            )

    @staticmethod
    def _client_values(client: AppApiClientEntity) -> tuple:
        return (
            client.client_id,
            client.app_id,
            int(client.domain_id),
            client.subject_user_id,
            client.display_name,
            client.status,
            int(client.rate_limit_per_minute),
            int(client.row_version),
            client.created_by,
            client.created_at,
            client.updated_at,
        )

    @staticmethod
    def _credential_values(credential: AppApiCredentialEntity) -> tuple:
        return (
            credential.credential_id,
            credential.public_key_id,
            credential.status,
            credential.key_digest,
            credential.expires_at,
            credential.last_used_at,
            credential.created_at,
            credential.revoked_at,
        )

    @classmethod
    def _client_view(
        cls, *, client, scopes, agent_ids, credentials
    ) -> dict[str, object]:
        return cls._client_view_values(
            cls._client_values(client), scopes, agent_ids,
            [cls._credential_values(item) for item in credentials],
        )

    @staticmethod
    def _client_view_values(
        client: tuple, scopes: tuple[str, ...], agent_ids, credentials
    ) -> dict[str, object]:
        return {
            "client_id": str(client[0]),
            "app_id": client[1],
            "domain_id": client[2],
            "subject_user_id": client[3],
            "display_name": client[4],
            "status": client[5],
            "rate_limit_per_minute": client[6],
            "row_version": client[7],
            "created_by": client[8],
            "created_at": client[9],
            "updated_at": client[10],
            "scopes": list(scopes),
            "agent_ids": [str(value) for value in agent_ids],
            "credentials": [{
                "credential_id": str(value[0]),
                "public_key_id": value[1],
                "status": value[2],
                "expires_at": value[4],
                "last_used_at": value[5],
                "created_at": value[6],
                "revoked_at": value[7],
            } for value in credentials],
        }


def require_app_api_scope(request, scope: str) -> None:
    """若当前主体是 App API Client，则强制检查机器 Scope。"""
    context = request.state.auth_context
    if context.principal_kind != PrincipalKind.APP_API_CLIENT:
        return
    if scope not in context.scopes:
        raise AppApiKeyError(
            "APP_API_KEY_SCOPE_DENIED", f"API Client 缺少 Scope：{scope}"
        )


def require_app_api_permission(request, permission: str) -> None:
    """确保 App API Client 至少拥有映射到业务权限的机器 Scope。"""
    context = request.state.auth_context
    if context.principal_kind != PrincipalKind.APP_API_CLIENT:
        return
    mapping = APP_API_SCOPE_PERMISSIONS.get(context.app_id or "", {})
    if not any(
        mapping.get(scope) == permission for scope in context.scopes
    ):
        raise AppApiKeyError(
            "APP_API_KEY_SCOPE_DENIED", "API Client Scope 不允许执行该操作"
        )


def require_app_api_agent(request, agent_id: UUID) -> None:
    """限制 App API Client 只能访问创建时绑定的 Agent。"""
    context = request.state.auth_context
    if context.principal_kind != PrincipalKind.APP_API_CLIENT:
        return
    if agent_id not in context.authorized_agent_ids:
        raise AppApiKeyError(
            "APP_API_KEY_AGENT_DENIED", "API Client 无权访问该 Agent", status_code=404
        )


__all__ = [
    "APP_API_KEY_PREFIX",
    "APP_API_SCOPE_PERMISSIONS",
    "AppApiKeyError",
    "AppApiKeyService",
    "require_app_api_agent",
    "require_app_api_permission",
    "require_app_api_scope",
]
