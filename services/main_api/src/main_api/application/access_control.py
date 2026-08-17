"""应用成员权限判定。"""

from __future__ import annotations

from dataclasses import dataclass

GLOBAL_ADMIN_USER_ID = "ADMIN"


def is_reserved_global_admin(user_id: str) -> bool:
    """保留所有大小写形式的全局管理员标识。"""
    return (
        user_id.strip().casefold()
        == GLOBAL_ADMIN_USER_ID.casefold()
    )


class AccessDeniedError(PermissionError):
    def __init__(self, permission_code: str):
        super().__init__(f"缺少权限：{permission_code}")
        self.permission_code = permission_code


class AccessConfigurationError(ValueError):
    """应用成员或角色配置无效。"""


@dataclass(frozen=True, slots=True)
class AccessSnapshot:
    app_id: str
    domain_id: int
    user_id: str
    roles: tuple[str, ...]
    permissions: frozenset[str]


@dataclass(frozen=True, slots=True)
class PlatformAccessSnapshot:
    user_id: str
    permissions: frozenset[str]


class AccessControlService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def snapshot(
        self, *, app_id: str, domain_id: int, user_id: str
    ) -> AccessSnapshot:
        async with self._uow_factory() as uow:
            permissions = await uow.access.permissions_for(
                app_id=app_id, domain_id=domain_id, user_id=user_id
            )
            roles = await uow.access.list_roles(
                app_id=app_id, domain_id=domain_id, user_id=user_id
            )
        return AccessSnapshot(
            app_id=app_id,
            domain_id=domain_id,
            user_id=user_id,
            roles=roles,
            permissions=frozenset(permissions),
        )

    async def require(
        self,
        *,
        app_id: str,
        domain_id: int,
        user_id: str,
        permission_code: str,
    ) -> AccessSnapshot:
        snapshot = await self.snapshot(
            app_id=app_id, domain_id=domain_id, user_id=user_id
        )
        if permission_code not in snapshot.permissions:
            raise AccessDeniedError(permission_code)
        return snapshot

    async def platform_snapshot(self, *, user_id: str) -> PlatformAccessSnapshot:
        """读取不依赖 Domain 的平台权限。"""
        async with self._uow_factory() as uow:
            permissions = await uow.access.platform_permissions_for(user_id=user_id)
        return PlatformAccessSnapshot(
            user_id=user_id,
            permissions=frozenset(permissions),
        )

    async def require_platform(
        self, *, user_id: str, permission_code: str
    ) -> PlatformAccessSnapshot:
        snapshot = await self.platform_snapshot(user_id=user_id)
        if permission_code not in snapshot.permissions:
            raise AccessDeniedError(permission_code)
        return snapshot

    async def user_max_security_level(self, *, user_id: str) -> int:
        """从用户主数据读取受信的检索安全等级上限。"""
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
        if user is None or user.status != "ACTIVE":
            raise AccessConfigurationError("平台用户不存在或已停用")
        level = int(user.max_security_level)
        if level < 0 or level > 3:
            raise AccessConfigurationError("平台用户安全等级配置无效")
        return level

    async def ensure_user(
        self, *, user_id: str, display_name: str | None = None
    ) -> None:
        del display_name
        async with self._uow_factory() as uow:
            row = await uow.access.get_user(user_id)
            if row is None:
                raise AccessConfigurationError(
                    "用户不存在，必须通过平台用户或 App 用户管理接口创建"
                )

    async def list_members(
        self, *, app_id: str, domain_id: int
    ) -> list[dict[str, object]]:
        async with self._uow_factory() as uow:
            members = await uow.access.list_app_members(app_id=app_id)
            rows = await uow.access.list_member_roles(app_id=app_id, domain_id=domain_id)
            visible_user_ids = {row.user_id for row in rows}
            members = [row for row in members if row.user_id in visible_user_ids]
            user_ids = tuple(row.user_id for row in members)
            users = {
                row.user_id: row
                for row in await uow.access.list_users_by_ids(user_ids)
            }
        grouped: dict[str, list[dict[str, str]]] = {row.user_id: [] for row in members}
        for row in rows:
            grouped.setdefault(row.user_id, []).append({
                "role_code": row.role_code, "status": row.status
            })
        return [{
            "user_id": user_id,
            "display_name": users.get(user_id).display_name
            if user_id in users else None,
            "max_security_level": int(users[user_id].max_security_level)
            if user_id in users else 0,
            "status": users.get(user_id).status
            if user_id in users else "ACTIVE",
            "protected": (
                is_reserved_global_admin(user_id)
                or any(row.user_id == user_id and row.is_initial_admin == "Y" for row in members)
            ),
            "roles": roles,
        } for user_id, roles in grouped.items()]

    async def list_policy_subjects(
        self, *, app_id: str, domain_id: int
    ) -> dict[str, list[dict[str, object]]]:
        """返回策略可选择的当前 Domain 成员及应用角色目录。"""
        members = await self.list_members(app_id=app_id, domain_id=domain_id)
        async with self._uow_factory() as uow:
            roles = await uow.access.list_active_app_roles(app_id=app_id)
        return {
            "members": [
                {
                    "id": str(item["user_id"]),
                    "display_name": item.get("display_name"),
                    "username": str(item["user_id"]),
                }
                for item in members
                if item.get("status") == "ACTIVE"
                and any(
                    role.get("status") == "ACTIVE"
                    for role in item.get("roles", [])
                )
            ],
            "roles": [
                {
                    "code": row.role_code,
                    "display_name": row.display_name,
                }
                for row in roles
            ],
        }

__all__ = [
    "AccessConfigurationError", "AccessControlService",
    "AccessDeniedError", "AccessSnapshot", "GLOBAL_ADMIN_USER_ID",
    "is_reserved_global_admin", "PlatformAccessSnapshot",
]
