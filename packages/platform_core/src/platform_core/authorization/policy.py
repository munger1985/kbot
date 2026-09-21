"""定义所有业务 App 必须遵守的统一授权规则。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from types import MappingProxyType
from typing import Any, Collection, Mapping

from platform_core.contracts import PrincipalKind


class AgentAccessMode(str, Enum):
    """Agent 对人类用户和机器客户端的统一可见性规则。"""

    DOMAIN_ACTIVE = "DOMAIN_ACTIVE"
    CLIENT_ALLOWLIST = "CLIENT_ALLOWLIST"


@dataclass(frozen=True, slots=True)
class ApiRouteScope:
    """机器公开路由与 Scope 的唯一映射。"""

    method: str
    relative_path_pattern: str
    scope: str

    def matches(self, *, method: str, relative_path: str) -> bool:
        return self.method == method.upper() and re.fullmatch(
            self.relative_path_pattern, relative_path
        ) is not None


@dataclass(frozen=True, slots=True)
class AppAuthorizationPolicy:
    """一个业务 App 的核心授权合同。"""

    app_id: str
    public_slug: str
    use_permission: str
    human_agent_access: AgentAccessMode = AgentAccessMode.DOMAIN_ACTIVE
    machine_agent_access: AgentAccessMode = AgentAccessMode.CLIENT_ALLOWLIST
    api_scope_permissions: Mapping[str, str] = MappingProxyType({})
    api_route_scopes: tuple[ApiRouteScope, ...] = ()

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.app_id) is None:
            raise ValueError(f"App ID 格式无效：{self.app_id}")
        if re.fullmatch(r"[a-z][a-z0-9-]*", self.public_slug) is None:
            raise ValueError(f"App URL Slug 格式无效：{self.public_slug}")
        expected_use_permission = f"{self.app_id}:use"
        if self.use_permission != expected_use_permission:
            raise ValueError(
                f"App {self.app_id} 的入口权限必须是 {expected_use_permission}"
            )
        for scope, permission in self.api_scope_permissions.items():
            if not scope or not permission.startswith(f"{self.app_id}:"):
                raise ValueError(
                    f"App {self.app_id} 的机器 Scope 映射不属于当前 App：{scope}"
                )
        unknown_scopes = {
            route.scope for route in self.api_route_scopes
        } - set(self.api_scope_permissions)
        if unknown_scopes:
            raise ValueError(
                f"App {self.app_id} 的公开路由引用未登记 Scope："
                f"{sorted(unknown_scopes)}"
            )

    def required_scope(self, *, method: str, relative_path: str) -> str | None:
        """返回公开机器路由要求的 Scope，未登记路由默认拒绝。"""

        for route in self.api_route_scopes:
            if route.matches(method=method, relative_path=relative_path):
                return route.scope
        return None

    def assert_role_permissions(
        self, permission_codes: Collection[str]
    ) -> None:
        """确保业务角色属于当前 App，且不能绕过入口权限。"""

        foreign_permissions = sorted(
            permission
            for permission in permission_codes
            if not permission.startswith(f"{self.app_id}:")
        )
        if foreign_permissions:
            raise ValueError(
                f"App {self.app_id} 的角色包含其他 App 权限："
                f"{foreign_permissions}"
            )
        if self.use_permission not in permission_codes:
            raise ValueError(
                f"App {self.app_id} 的角色必须包含 {self.use_permission}"
            )


def _policy(
    *,
    app_id: str,
    public_slug: str,
    api_scope_permissions: Mapping[str, str] | None = None,
    api_route_scopes: tuple[tuple[str, str, str], ...] = (),
) -> AppAuthorizationPolicy:
    return AppAuthorizationPolicy(
        app_id=app_id,
        public_slug=public_slug,
        use_permission=f"{app_id}:use",
        api_scope_permissions=MappingProxyType(dict(api_scope_permissions or {})),
        api_route_scopes=tuple(ApiRouteScope(*route) for route in api_route_scopes),
    )


APP_AUTHORIZATION_POLICIES: Mapping[str, AppAuthorizationPolicy] = (
    MappingProxyType({
        "knowledge_retrieval": _policy(
            app_id="knowledge_retrieval",
            public_slug="knowledge-retrieval",
            api_scope_permissions={
                "knowledge:agent:read": "knowledge_retrieval:use",
                "knowledge:chat:write": "knowledge_retrieval:use",
                "knowledge:conversation:read": "knowledge_retrieval:use",
                "knowledge:run:read": "knowledge_retrieval:use",
            },
            api_route_scopes=(
                ("GET", r"agents", "knowledge:agent:read"),
                ("GET", r"agents/[0-9a-fA-F-]{36}", "knowledge:agent:read"),
                ("POST", r"runs", "knowledge:chat:write"),
                ("POST", r"runs/[0-9a-fA-F-]{36}/cancel", "knowledge:chat:write"),
                ("GET", r"runs/[0-9a-fA-F-]{36}", "knowledge:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/result", "knowledge:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/events", "knowledge:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/references/[^/]+/preview", "knowledge:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/references/[^/]+/content", "knowledge:run:read"),
                ("POST", r"conversations", "knowledge:chat:write"),
                ("POST", r"conversations/[0-9a-fA-F-]{36}/turns", "knowledge:chat:write"),
                ("POST", r"conversations/[0-9a-fA-F-]{36}/turns/multipart", "knowledge:chat:write"),
                ("GET", r"conversations", "knowledge:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}", "knowledge:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}/turns", "knowledge:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}/turns/[0-9a-fA-F-]{36}/trace", "knowledge:conversation:read"),
                ("PATCH", r"conversations/[0-9a-fA-F-]{36}", "knowledge:chat:write"),
                ("DELETE", r"conversations/[0-9a-fA-F-]{36}", "knowledge:chat:write"),
                ("GET", r"memories", "knowledge:conversation:read"),
            ),
        ),
        "km_asset": _policy(
            app_id="km_asset",
            public_slug="km-asset",
            api_scope_permissions={
                "km:agent:read": "km_asset:use",
                "km:chat:write": "km_asset:use",
                "km:conversation:read": "km_asset:use",
                "km:conversation:update": "km_asset:use",
                "km:conversation:delete": "km_asset:use",
                "km:run:read": "km_asset:use",
                "km:reference:read": "km_asset:use",
            },
            api_route_scopes=(
                ("GET", r"agents", "km:agent:read"),
                ("GET", r"agents/[0-9a-fA-F-]{36}", "km:agent:read"),
                ("POST", r"conversations", "km:chat:write"),
                ("POST", r"conversations/[0-9a-fA-F-]{36}/turns", "km:chat:write"),
                ("GET", r"conversations", "km:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}", "km:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}/turns", "km:conversation:read"),
                ("PATCH", r"conversations/[0-9a-fA-F-]{36}", "km:conversation:update"),
                ("DELETE", r"conversations/[0-9a-fA-F-]{36}", "km:conversation:delete"),
                ("GET", r"runs/[0-9a-fA-F-]{36}", "km:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/result", "km:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/events", "km:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/references/[^/]+/preview", "km:reference:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/references/[^/]+/files/[0-9a-fA-F-]{36}/content", "km:reference:read"),
            ),
        ),
        "media_studio": _policy(
            app_id="media_studio",
            public_slug="media-studio",
        ),
        "aiops": _policy(
            app_id="aiops",
            public_slug="aiops",
            api_scope_permissions={
                "aiops:agent:read": "aiops:use",
                "aiops:chat:write": "aiops:use",
                "aiops:conversation:read": "aiops:use",
                "aiops:conversation:delete": "aiops:use",
                "aiops:run:read": "aiops:use",
            },
            api_route_scopes=(
                ("GET", r"agents", "aiops:agent:read"),
                ("GET", r"agents/[0-9a-fA-F-]{36}", "aiops:agent:read"),
                ("POST", r"conversation-uploads", "aiops:chat:write"),
                ("POST", r"conversations", "aiops:chat:write"),
                ("POST", r"conversations/[0-9a-fA-F-]{36}/turns", "aiops:chat:write"),
                ("POST", r"conversations/[0-9a-fA-F-]{36}/turns/[0-9a-fA-F-]{36}/cancel", "aiops:chat:write"),
                ("GET", r"conversations", "aiops:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}", "aiops:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}/turns", "aiops:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}/turns/[0-9a-fA-F-]{36}", "aiops:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}/turns/[0-9a-fA-F-]{36}/events", "aiops:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}/turns/[0-9a-fA-F-]{36}/inputs/[0-9]+/content", "aiops:conversation:read"),
                ("GET", r"conversations/[0-9a-fA-F-]{36}/turns/[0-9a-fA-F-]{36}/workload-reports/[^/]+", "aiops:conversation:read"),
                ("DELETE", r"conversations/[0-9a-fA-F-]{36}", "aiops:conversation:delete"),
                ("POST", r"runs", "aiops:chat:write"),
                ("POST", r"runs/[0-9a-fA-F-]{36}/cancel", "aiops:chat:write"),
                ("GET", r"fleet", "aiops:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}", "aiops:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/result", "aiops:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/events", "aiops:run:read"),
                ("GET", r"runs/[0-9a-fA-F-]{36}/pending-input", "aiops:run:read"),
                ("GET", r"reports/[0-9a-fA-F-]{36}", "aiops:run:read"),
            ),
        ),
    })
)

_PUBLIC_SLUGS = tuple(
    policy.public_slug for policy in APP_AUTHORIZATION_POLICIES.values()
)
if len(_PUBLIC_SLUGS) != len(set(_PUBLIC_SLUGS)):
    raise RuntimeError("业务 App 的公开 URL Slug 不能重复")


def get_app_authorization_policy(app_id: str) -> AppAuthorizationPolicy:
    """返回已注册 App 的授权策略，未注册 App 一律拒绝。"""

    try:
        return APP_AUTHORIZATION_POLICIES[app_id]
    except KeyError as exc:
        raise ValueError(f"App 未注册统一授权策略：{app_id}") from exc


def resolve_business_app_id(value: str) -> str:
    """把稳定 App ID 或公开 URL Slug 解析为唯一 App ID。"""

    for app_id, policy in APP_AUTHORIZATION_POLICIES.items():
        if value in {app_id, policy.public_slug}:
            return app_id
    raise ValueError(f"App 未注册统一授权策略：{value}")


def registered_business_app_ids() -> tuple[str, ...]:
    """返回全部已注册业务 App。"""

    return tuple(APP_AUTHORIZATION_POLICIES)


def can_read_agent(
    *,
    app_id: str,
    domain_id: int,
    principal_kind: PrincipalKind,
    permissions: Collection[str],
    authorized_agent_ids: Collection[object],
    agent: Mapping[str, Any],
) -> bool:
    """按核心策略判定一个 Agent 是否可由当前主体读取。"""

    policy = get_app_authorization_policy(app_id)
    agent_id = str(agent.get("agent_id") or "")
    status = str(agent.get("status") or "")
    try:
        agent_domain_id = int(agent.get("domain_id"))
    except (TypeError, ValueError):
        return False
    if (
        domain_id < 1
        or agent_domain_id != domain_id
        or not agent_id
        or policy.use_permission not in permissions
    ):
        return False
    if principal_kind == PrincipalKind.APP_API_CLIENT:
        if policy.machine_agent_access != AgentAccessMode.CLIENT_ALLOWLIST:
            raise RuntimeError(
                f"App {app_id} 未实现机器 Agent 访问模式："
                f"{policy.machine_agent_access}"
        )
        allowed = {str(value) for value in authorized_agent_ids}
        return status == "ACTIVE" and agent_id in allowed

    if principal_kind != PrincipalKind.PORTAL:
        return False

    if policy.human_agent_access != AgentAccessMode.DOMAIN_ACTIVE:
        raise RuntimeError(
            f"App {app_id} 未实现人类 Agent 访问模式："
            f"{policy.human_agent_access}"
        )
    if f"{app_id}:agent_manage" in permissions:
        return True
    return status == "ACTIVE"


def filter_readable_agents(
    *,
    app_id: str,
    domain_id: int,
    principal_kind: PrincipalKind,
    permissions: Collection[str],
    authorized_agent_ids: Collection[object],
    agents: Collection[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """使用唯一核心规则过滤 Agent 目录。"""

    return [
        agent
        for agent in agents
        if can_read_agent(
            app_id=app_id,
            domain_id=domain_id,
            principal_kind=principal_kind,
            permissions=permissions,
            authorized_agent_ids=authorized_agent_ids,
            agent=agent,
        )
    ]
