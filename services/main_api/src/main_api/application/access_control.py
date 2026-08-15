"""应用成员权限判定。"""

from __future__ import annotations

from dataclasses import dataclass

from main_api.entities.access_control import PlatformUserEntity


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

    async def ensure_user(
        self, *, user_id: str, display_name: str | None = None
    ) -> None:
        async with self._uow_factory() as uow:
            row = await uow.access.get_user(user_id)
            if is_reserved_global_admin(user_id) and row is None:
                raise AccessConfigurationError(
                    "ADMIN 是平台保留账号，只能通过项目初始化脚本创建"
                )
            if row is None:
                await uow.access.add_user(
                    PlatformUserEntity(
                        user_id=user_id,
                        display_name=display_name,
                        status="ACTIVE",
                    )
                )
                await uow.commit()

    async def list_members(
        self, *, app_id: str, domain_id: int
    ) -> list[dict[str, object]]:
        async with self._uow_factory() as uow:
            rows = await uow.access.list_member_roles(
                app_id=app_id, domain_id=domain_id
            )
            user_ids = tuple(dict.fromkeys(row.user_id for row in rows))
            users = {
                row.user_id: row
                for row in await uow.access.list_users_by_ids(user_ids)
            }
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row.user_id, []).append({
                "role_code": row.role_code, "status": row.status
            })
        return [{
            "user_id": user_id,
            "display_name": users.get(user_id).display_name
            if user_id in users else None,
            "status": users.get(user_id).status
            if user_id in users else "ACTIVE",
            "protected": is_reserved_global_admin(user_id),
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

    async def set_member_role(
        self, *, app_id: str, domain_id: int, user_id: str,
        display_name: str | None, role_code: str, status: str,
        actor_id: str,
    ) -> dict[str, str]:
        if status not in {"ACTIVE", "DISABLED"}:
            raise AccessConfigurationError("成员角色状态无效")
        if is_reserved_global_admin(user_id):
            raise AccessConfigurationError(
                "ADMIN 是平台保留账号，不能通过成员角色管理修改或删除"
            )
        async with self._uow_factory() as uow:
            role = await uow.access.get_role(
                app_id=app_id, role_code=role_code
            )
            if role is None or role.status != "ACTIVE":
                raise AccessConfigurationError("应用角色不存在或已停用")
            user = await uow.access.get_user(user_id)
            if user is None:
                await uow.access.add_user(PlatformUserEntity(
                    user_id=user_id, display_name=display_name, status="ACTIVE"
                ))
            row = await uow.access.upsert_member_role(
                app_id=app_id, domain_id=domain_id, user_id=user_id,
                role_code=role_code, status=status, actor_id=actor_id,
            )
            await uow.commit()
        return {
            "user_id": row.user_id,
            "role_code": row.role_code,
            "status": row.status,
        }


__all__ = [
    "AccessConfigurationError", "AccessControlService",
    "AccessDeniedError", "AccessSnapshot", "GLOBAL_ADMIN_USER_ID",
    "is_reserved_global_admin",
]
