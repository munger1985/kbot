"""平台用户、应用角色及成员授权管理。"""

from __future__ import annotations

import asyncio

import bcrypt

from main_api.application.access_control import (
    SYSTEM_ADMIN_ROLE_CODE,
    is_reserved_global_admin,
)
from main_api.entities.access_control import (
    AppRoleEntity,
    PlatformUserCredentialEntity,
    PlatformUserEntity,
)


class AccessManagementError(ValueError):
    """用户或角色管理请求不符合平台约束。"""

    def __init__(self, code: str, message: str, *, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _assert_status(value: str) -> None:
    if value not in {"ACTIVE", "DISABLED"}:
        raise AccessManagementError("INVALID_STATUS", "状态值无效")


def _assert_security_level(value: int) -> None:
    if value < 0 or value > 3:
        raise AccessManagementError(
            "INVALID_SECURITY_LEVEL", "用户安全等级必须在 0 到 3 之间"
        )


def _assert_mutable_user(user_id: str) -> None:
    if is_reserved_global_admin(user_id):
        raise AccessManagementError(
            "GLOBAL_ADMIN_PROTECTED",
            "ADMIN 是平台保留账号，不能通过用户或角色管理修改",
            status_code=409,
        )


def _assert_mutable_role(role_code: str) -> None:
    if role_code.casefold() == SYSTEM_ADMIN_ROLE_CODE:
        raise AccessManagementError(
            "SYSTEM_ADMIN_ROLE_PROTECTED",
            "system_admin 是平台保留角色，不能通过角色管理修改",
            status_code=409,
        )


class AccessManagementService:
    """集中维护用户、密码凭据、角色定义与 Domain 成员关系。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def list_users(
        self,
        *,
        offset: int,
        limit: int,
        search: str | None,
        status: str | None,
    ) -> dict[str, object]:
        if status:
            _assert_status(status)
        async with self._uow_factory() as uow:
            rows, total = await uow.access.list_users(
                offset=offset,
                limit=limit,
                search=search.strip() if search else None,
                status=status,
            )
            items = [self._user_item(row) for row in rows]
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "total": total,
        }

    async def get_user(self, *, user_id: str) -> dict[str, object]:
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
            credential = await uow.access.get_user_credential(user_id)
            memberships = await uow.access.list_user_memberships(user_id=user_id)
            if user is None:
                raise AccessManagementError(
                    "USER_NOT_FOUND", "用户不存在", status_code=404
                )
            item = self._user_item(user)
            item["credential_configured"] = credential is not None
            item["must_change_password"] = bool(
                credential is not None and credential.must_change_password == "Y"
            )
            item["memberships"] = [
                {
                    "app_id": row.app_id,
                    "domain_id": int(row.domain_id),
                    "role_code": row.role_code,
                    "status": row.status,
                }
                for row in memberships
            ]
        return item

    async def create_user(
        self,
        *,
        user_id: str,
        display_name: str | None,
        password: str,
        status: str,
        must_change_password: bool,
        max_security_level: int = 1,
    ) -> dict[str, object]:
        _assert_status(status)
        _assert_security_level(max_security_level)
        _assert_mutable_user(user_id)
        password_hash = await asyncio.to_thread(
            bcrypt.hashpw,
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12),
        )
        async with self._uow_factory() as uow:
            if await uow.access.get_user(user_id) is not None:
                raise AccessManagementError(
                    "USER_ALREADY_EXISTS", "用户已存在", status_code=409
                )
            user = PlatformUserEntity(
                user_id=user_id,
                display_name=display_name,
                max_security_level=max_security_level,
                status=status,
            )
            await uow.access.add_user(user)
            await uow.access.add_user_credential(
                PlatformUserCredentialEntity(
                    user_id=user_id,
                    password_hash=password_hash.decode("ascii"),
                    must_change_password="Y" if must_change_password else "N",
                )
            )
            result = self._user_item(user)
            await uow.commit()
        return result

    async def update_user(
        self,
        *,
        user_id: str,
        display_name: str | None,
        display_name_provided: bool,
        status: str | None,
        max_security_level: int | None = None,
    ) -> dict[str, object]:
        _assert_mutable_user(user_id)
        if status is not None:
            _assert_status(status)
        if max_security_level is not None:
            _assert_security_level(max_security_level)
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
            if user is None:
                raise AccessManagementError(
                    "USER_NOT_FOUND", "用户不存在", status_code=404
                )
            await uow.access.update_user(
                user=user,
                display_name=(
                    display_name if display_name_provided else user.display_name
                ),
                status=status or user.status,
                max_security_level=(
                    max_security_level
                    if max_security_level is not None
                    else int(user.max_security_level)
                ),
            )
            result = self._user_item(user)
            await uow.commit()
        return result

    async def reset_password(
        self,
        *,
        user_id: str,
        password: str,
        must_change_password: bool,
    ) -> dict[str, object]:
        password_hash = await asyncio.to_thread(
            bcrypt.hashpw,
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12),
        )
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
            if user is None:
                raise AccessManagementError(
                    "USER_NOT_FOUND", "用户不存在", status_code=404
                )
            credential = await uow.access.get_user_credential(user_id)
            if credential is None:
                await uow.access.add_user_credential(
                    PlatformUserCredentialEntity(
                        user_id=user_id,
                        password_hash=password_hash.decode("ascii"),
                        must_change_password=(
                            "Y" if must_change_password else "N"
                        ),
                    )
                )
            else:
                await uow.access.set_user_password(
                    credential=credential,
                    password_hash=password_hash.decode("ascii"),
                    must_change_password=must_change_password,
                )
            await uow.commit()
        return {
            "user_id": user_id,
            "must_change_password": must_change_password,
        }

    async def delete_user(self, *, user_id: str) -> dict[str, object]:
        """物理删除普通用户、登录凭据及全部 Domain 成员关系。"""
        _assert_mutable_user(user_id)
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
            if user is None:
                raise AccessManagementError(
                    "USER_NOT_FOUND", "用户不存在", status_code=404
                )
            await uow.access.delete_user(user=user)
            await uow.commit()
        return {"user_id": user_id, "deleted": True}

    async def set_memberships(
        self,
        *,
        app_id: str,
        domain_ids: tuple[int, ...],
        user_id: str,
        role_code: str,
        status: str,
        actor_id: str,
    ) -> dict[str, object]:
        _assert_mutable_user(user_id)
        _assert_mutable_role(role_code)
        _assert_status(status)
        normalized_domain_ids = tuple(dict.fromkeys(domain_ids))
        if not normalized_domain_ids or any(
            domain_id <= 0 for domain_id in normalized_domain_ids
        ):
            raise AccessManagementError(
                "INVALID_DOMAIN_IDS", "业务域列表不能为空且 ID 必须为正整数"
            )
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
            role = await uow.access.get_role(app_id=app_id, role_code=role_code)
            if user is None:
                raise AccessManagementError(
                    "USER_NOT_FOUND", "用户不存在", status_code=404
                )
            if role is None:
                raise AccessManagementError(
                    "ROLE_NOT_FOUND", "应用角色不存在", status_code=404
                )
            domains = await uow.domains.list_by_ids(
                domain_ids=normalized_domain_ids
            )
            existing_domain_ids = {int(domain.domain_id) for domain in domains}
            missing_domain_ids = [
                domain_id
                for domain_id in normalized_domain_ids
                if domain_id not in existing_domain_ids
            ]
            if missing_domain_ids:
                raise AccessManagementError(
                    "DOMAIN_NOT_FOUND",
                    "Domain 不存在："
                    + ", ".join(str(value) for value in missing_domain_ids),
                    status_code=404,
                )
            items = []
            for domain_id in normalized_domain_ids:
                row = await uow.access.upsert_member_role(
                    app_id=app_id,
                    domain_id=domain_id,
                    user_id=user_id,
                    role_code=role_code,
                    status=status,
                    actor_id=actor_id,
                )
                items.append(
                    {
                        "app_id": row.app_id,
                        "domain_id": int(row.domain_id),
                        "user_id": row.user_id,
                        "role_code": row.role_code,
                        "status": row.status,
                    }
                )
            await uow.commit()
        return {"items": items, "total": len(items)}

    async def list_permissions(
        self, *, app_id: str | None
    ) -> list[dict[str, str]]:
        async with self._uow_factory() as uow:
            rows = await uow.access.list_permissions(app_id=app_id)
            return [
                {
                    "app_id": row.app_id,
                    "permission_code": row.permission_code,
                    "display_name": row.display_name,
                }
                for row in rows
            ]

    async def list_roles(
        self, *, app_id: str | None
    ) -> list[dict[str, object]]:
        async with self._uow_factory() as uow:
            rows = await uow.access.list_all_roles(app_id=app_id)
            result = []
            for row in rows:
                permissions = await uow.access.list_role_permission_codes(
                    app_id=row.app_id, role_code=row.role_code
                )
                result.append(self._role_item(row, permissions))
        return result

    async def create_role(
        self,
        *,
        app_id: str,
        role_code: str,
        display_name: str,
        permission_codes: tuple[str, ...],
    ) -> dict[str, object]:
        _assert_mutable_role(role_code)
        async with self._uow_factory() as uow:
            if await uow.access.get_role(app_id=app_id, role_code=role_code):
                raise AccessManagementError(
                    "ROLE_ALREADY_EXISTS", "应用角色已存在", status_code=409
                )
            await self._validate_permissions(
                uow=uow, app_id=app_id, permission_codes=permission_codes
            )
            role = AppRoleEntity(
                app_id=app_id,
                role_code=role_code,
                display_name=display_name,
                status="ACTIVE",
            )
            await uow.access.add_role(role)
            await uow.access.replace_role_permissions(
                app_id=app_id,
                role_code=role_code,
                permission_codes=permission_codes,
            )
            result = self._role_item(role, permission_codes)
            await uow.commit()
        return result

    async def update_role(
        self,
        *,
        app_id: str,
        role_code: str,
        display_name: str,
        status: str,
        permission_codes: tuple[str, ...],
    ) -> dict[str, object]:
        _assert_mutable_role(role_code)
        _assert_status(status)
        async with self._uow_factory() as uow:
            role = await uow.access.get_role(app_id=app_id, role_code=role_code)
            if role is None:
                raise AccessManagementError(
                    "ROLE_NOT_FOUND", "应用角色不存在", status_code=404
                )
            await self._validate_permissions(
                uow=uow, app_id=app_id, permission_codes=permission_codes
            )
            await uow.access.update_role(
                role=role, display_name=display_name, status=status
            )
            await uow.access.replace_role_permissions(
                app_id=app_id,
                role_code=role_code,
                permission_codes=permission_codes,
            )
            result = self._role_item(role, permission_codes)
            await uow.commit()
        return result

    async def delete_role(
        self, *, app_id: str, role_code: str
    ) -> dict[str, object]:
        """逻辑删除应用角色，保留历史成员关系和权限定义。"""
        _assert_mutable_role(role_code)
        async with self._uow_factory() as uow:
            role = await uow.access.get_role(app_id=app_id, role_code=role_code)
            if role is None:
                raise AccessManagementError(
                    "ROLE_NOT_FOUND", "应用角色不存在", status_code=404
                )
            await uow.access.update_role(
                role=role,
                display_name=role.display_name,
                status="DISABLED",
            )
            await uow.commit()
        return {
            "app_id": app_id,
            "role_code": role_code,
            "status": "DISABLED",
            "deleted": True,
        }

    @staticmethod
    async def _validate_permissions(*, uow, app_id: str, permission_codes):
        if len(permission_codes) != len(set(permission_codes)):
            raise AccessManagementError(
                "DUPLICATE_PERMISSION", "角色权限不能重复"
            )
        rows = await uow.access.list_permissions(app_id=app_id)
        allowed = {row.permission_code for row in rows}
        unknown = sorted(set(permission_codes) - allowed)
        if unknown:
            raise AccessManagementError(
                "INVALID_ROLE_PERMISSION",
                f"权限不属于应用 {app_id}：{', '.join(unknown)}",
            )

    @staticmethod
    def _user_item(user: PlatformUserEntity) -> dict[str, object]:
        return {
            "user_id": user.user_id,
            "display_name": user.display_name,
            "max_security_level": int(user.max_security_level),
            "status": user.status,
            "protected": is_reserved_global_admin(user.user_id),
            "created_at": user.created_at,
            "updated_at": user.updated_at,
        }

    @staticmethod
    def _role_item(
        role: AppRoleEntity, permissions: tuple[str, ...]
    ) -> dict[str, object]:
        return {
            "app_id": role.app_id,
            "role_code": role.role_code,
            "display_name": role.display_name,
            "status": role.status,
            "protected": role.role_code.casefold() == SYSTEM_ADMIN_ROLE_CODE,
            "permissions": list(permissions),
        }


__all__ = [
    "AccessManagementError",
    "AccessManagementService",
    "SYSTEM_ADMIN_ROLE_CODE",
]
