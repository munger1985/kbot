"""平台账号、App 初始管理员与显式授权管理。"""

from __future__ import annotations

import asyncio
import json

import bcrypt

from main_api.application.access_control import is_reserved_global_admin
from main_api.application.notification_user_binding import personal_binding, set_personal_binding
from main_api.entities.access_control import (
    AccessAuditEventEntity,
    AppDomainEntity,
    AppMemberEntity,
    AppRoleEntity,
    PlatformUserCredentialEntity,
    PlatformUserEntity,
)
from platform_core.authorization import (
    BUILT_IN_APP_ROLE_CODES,
    app_permission_catalog,
    built_in_role_permissions,
    get_app_authorization_policy,
)
from platform_core.identity import uuid7
from platform_core.notifications.delivery import DeliveryScope


INITIAL_APP_ADMIN_ROLE = "app_admin"


class AccessManagementError(ValueError):
    """用户或授权管理请求不符合平台边界。"""

    def __init__(self, code: str, message: str, *, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _assert_status(value: str) -> None:
    if value not in {"ACTIVE", "DISABLED"}:
        raise AccessManagementError("INVALID_STATUS", "状态值无效")


def _assert_security_level(value: int) -> None:
    if value < 0 or value > 3:
        raise AccessManagementError("INVALID_SECURITY_LEVEL", "用户安全等级必须在 0 到 3 之间")


def _assert_mutable_user(user: PlatformUserEntity) -> None:
    if is_reserved_global_admin(user.user_id) or getattr(user, "is_protected", "N") == "Y":
        raise AccessManagementError(
            "PROTECTED_USER", "受保护账号不能修改、停用或删除", status_code=409
        )


class AccessManagementService:
    """在单一 UoW 中维护身份、凭据、App 资格、角色和 Domain 范围。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    @staticmethod
    async def _password_hash(password: str) -> str:
        value = await asyncio.to_thread(
            bcrypt.hashpw, password.encode("utf-8"), bcrypt.gensalt(rounds=12)
        )
        return value.decode("ascii")

    async def list_users(
        self, *, offset: int, limit: int, search: str | None, status: str | None,
    ) -> dict[str, object]:
        if status:
            _assert_status(status)
        async with self._uow_factory() as uow:
            rows, total = await uow.access.list_users(
                offset=offset, limit=limit, search=search.strip() if search else None,
                status=status,
            )
            assignments = await uow.access.list_user_app_assignments(tuple(row.user_id for row in rows))
            items = [{**self._user_item(row), "assigned_app_names": assignments.get(row.user_id, [])} for row in rows]
        return {"items": items, "offset": offset, "limit": limit, "total": total}

    async def get_user(self, *, user_id: str) -> dict[str, object]:
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(user_id)
            if user is None:
                raise AccessManagementError("USER_NOT_FOUND", "用户不存在", status_code=404)
            credential = await uow.access.get_user_credential(user_id)
            memberships = await uow.access.list_user_memberships(user_id=user_id)
            item = self._user_item(user)
            item["credential_configured"] = credential is not None
            item["must_change_password"] = bool(credential and credential.must_change_password == "Y")
            item["app_memberships"] = []
            for row in memberships:
                member_item = self._member_item(row)
                member_item["role_bindings"] = await self._binding_items(
                    uow=uow, app_id=row.app_id, user_id=user_id
                )
                item["app_memberships"].append(member_item)
            return item

    async def create_platform_user(
        self, *, user_id: str, display_name: str | None, password: str,
        status: str, must_change_password: bool, max_security_level: int,
        platform_role_codes: tuple[str, ...], actor_id: str,
        actor_security_level: int, actor_permissions: frozenset[str],
    ) -> dict[str, object]:
        _assert_status(status)
        _assert_security_level(max_security_level)
        if max_security_level > actor_security_level:
            raise AccessManagementError("USER_SECURITY_LEVEL_DENIED", "不能授予高于当前管理员的安全等级", status_code=403)
        if is_reserved_global_admin(user_id):
            raise AccessManagementError("GLOBAL_ADMIN_PROTECTED", "ADMIN 只能通过项目初始化创建", status_code=409)
        password_hash = await self._password_hash(password)
        async with self._uow_factory() as uow:
            if await uow.access.get_user(user_id) is not None:
                raise AccessManagementError("USER_ALREADY_EXISTS", "用户已存在", status_code=409)
            roles = []
            for role_code in tuple(dict.fromkeys(platform_role_codes)):
                role = await uow.access.get_role(app_id="platform", role_code=role_code)
                if role is None or role.status != "ACTIVE":
                    raise AccessManagementError("PLATFORM_ROLE_NOT_FOUND", f"平台角色不存在或已停用：{role_code}", status_code=404)
                permissions = frozenset(await uow.access.list_role_permission_codes(app_id="platform", role_code=role_code))
                if not permissions.issubset(actor_permissions):
                    raise AccessManagementError("PLATFORM_ROLE_ESCALATION", f"不能分配超出当前管理员权限的角色：{role_code}", status_code=403)
                roles.append(role_code)
            user = PlatformUserEntity(
                user_id=user_id, display_name=display_name, account_origin="PLATFORM",
                owner_app_id=None, is_protected="N", max_security_level=max_security_level,
                status=status,
            )
            await uow.access.add_user(user)
            await uow.access.add_user_credential(PlatformUserCredentialEntity(
                user_id=user_id, password_hash=password_hash,
                must_change_password="Y" if must_change_password else "N",
            ))
            for role_code in roles:
                await uow.access.upsert_platform_user_role(
                    user_id=user_id, role_code=role_code, status="ACTIVE", actor_id=actor_id
                )
            await uow.commit()
            result = self._user_item(user)
            result["platform_roles"] = roles
            return result

    async def create_initial_app_admin(
        self, *, app_id: str, user_id: str, display_name: str | None,
        password: str, must_change_password: bool, max_security_level: int,
        actor_id: str, actor_security_level: int,
    ) -> dict[str, object]:
        _assert_security_level(max_security_level)
        if max_security_level > actor_security_level:
            raise AccessManagementError("USER_SECURITY_LEVEL_DENIED", "不能授予高于当前管理员的安全等级", status_code=403)
        if is_reserved_global_admin(user_id):
            raise AccessManagementError("GLOBAL_ADMIN_PROTECTED", "ADMIN 不能作为 App 初始管理员", status_code=409)
        password_hash = await self._password_hash(password)
        async with self._uow_factory() as uow:
            app = await self._require_app(uow=uow, app_id=app_id, require_active=False)
            if app.member_assignable != "Y":
                raise AccessManagementError("APP_MEMBER_DISABLED", "该 App 不允许配置成员", status_code=409)
            app_domains = await uow.access.list_app_domains(app_id=app_id)
            if app_id != "knowledge_retrieval" and not any(row.status == "ACTIVE" for row in app_domains):
                raise AccessManagementError(
                    "APP_DOMAIN_REQUIRED",
                    "请先为 App 配置至少一个启用的 Domain，再创建初始管理员",
                    status_code=409,
                )
            existing_members = await uow.access.list_app_members(app_id=app_id)
            if any(row.is_initial_admin == "Y" for row in existing_members):
                raise AccessManagementError("INITIAL_APP_ADMIN_EXISTS", "该 App 已存在初始管理员", status_code=409)
            if await uow.access.get_user(user_id) is not None:
                raise AccessManagementError("USER_ALREADY_EXISTS", "初始管理员必须使用尚未存在的 App 账号", status_code=409)
            role = await uow.access.get_role(app_id=app_id, role_code=INITIAL_APP_ADMIN_ROLE)
            if role is None or role.status != "ACTIVE" or role.is_system != "Y":
                raise AccessManagementError("INITIAL_ADMIN_ROLE_MISSING", "App 初始管理员系统角色尚未初始化", status_code=503)
            user = PlatformUserEntity(
                user_id=user_id, display_name=display_name, account_origin="APP",
                owner_app_id=app_id, is_protected="Y", max_security_level=max_security_level,
                status="ACTIVE",
            )
            await uow.access.add_user(user)
            await uow.access.add_user_credential(PlatformUserCredentialEntity(
                user_id=user_id, password_hash=password_hash,
                must_change_password="Y" if must_change_password else "N",
            ))
            member = AppMemberEntity(
                app_id=app_id, user_id=user_id, member_source="APP_INITIAL_ADMIN",
                is_initial_admin="Y", status="ACTIVE", granted_by=actor_id,
            )
            await uow.access.add_app_member(member)
            await uow.access.upsert_member_role(
                app_id=app_id, user_id=user_id, role_code=INITIAL_APP_ADMIN_ROLE,
                scope_mode="ALL_APP_DOMAINS", status="ACTIVE", actor_id=actor_id,
            )
            await uow.access.replace_member_role_scopes(
                app_id=app_id, user_id=user_id, role_code=INITIAL_APP_ADMIN_ROLE, domain_ids=()
            )
            await uow.commit()
            return {**self._user_item(user), "membership": self._member_item(member)}

    async def set_platform_app_grant(
        self, *, user_id: str, app_id: str,
        role_bindings: tuple[dict[str, object], ...], actor_id: str,
        actor_platform_permissions: frozenset[str],
    ) -> dict[str, object]:
        if user_id.casefold() == actor_id.casefold():
            raise AccessManagementError(
                "APP_GRANT_SELF_ESCALATION",
                "平台授权管理员不能给自己授予业务 App 权限",
                status_code=403,
            )
        assignable_permissions = self._app_grant_permission_limit(app_id, actor_platform_permissions)
        async with self._uow_factory() as uow:
            user = await uow.access.get_user_for_update(user_id)
            if user is None:
                raise AccessManagementError("USER_NOT_FOUND", "平台用户不存在", status_code=404)
            if user.account_origin != "PLATFORM":
                raise AccessManagementError("PLATFORM_USER_REQUIRED", "只有平台来源账号可以使用显式 App Grant")
            _assert_mutable_user(user)
            app = await self._require_app(uow=uow, app_id=app_id)
            if app.member_assignable != "Y":
                raise AccessManagementError("APP_MEMBER_DISABLED", "该 App 不允许配置成员", status_code=409)
            member = await uow.access.get_app_member(app_id=app_id, user_id=user_id)
            if member is None:
                member = AppMemberEntity(
                    app_id=app_id, user_id=user_id, member_source="PLATFORM_GRANT",
                    is_initial_admin="N", status="ACTIVE", granted_by=actor_id,
                )
                await uow.access.add_app_member(member)
            else:
                if member.member_source != "PLATFORM_GRANT":
                    raise AccessManagementError("APP_MEMBER_SOURCE_CONFLICT", "该成员不是平台显式授权产生的记录", status_code=409)
                await self._assert_existing_app_roles_assignable(
                    uow=uow, app_id=app_id, user_id=user_id, assignable_permissions=assignable_permissions,
                )
                await uow.access.update_app_member_status(member=member, status="ACTIVE")
            bindings = await self._replace_role_bindings(
                uow=uow, app_id=app_id, user_id=user_id,
                role_bindings=role_bindings, actor_id=actor_id,
                assignable_permissions=assignable_permissions,
            )
            await uow.commit()
            return {"user_id": user_id, "app_id": app_id, "status": "ACTIVE", "role_bindings": bindings}

    async def set_platform_user_roles(
        self, *, user_id: str, role_codes: tuple[str, ...], actor_id: str,
        assignable_permissions: frozenset[str],
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            user = await uow.access.get_user_for_update(user_id)
            if user is None or user.account_origin != "PLATFORM":
                raise AccessManagementError("PLATFORM_USER_REQUIRED", "平台用户不存在", status_code=404)
            await self._assert_platform_target_dominated(uow=uow, actor_id=actor_id, target=user)
            normalized = tuple(dict.fromkeys(role_codes))
            for role_code in normalized:
                role = await uow.access.get_role(app_id="platform", role_code=role_code)
                if role is None or role.status != "ACTIVE":
                    raise AccessManagementError("PLATFORM_ROLE_NOT_FOUND", f"平台角色不存在或已停用：{role_code}", status_code=404)
                permissions = frozenset(await uow.access.list_role_permission_codes(app_id="platform", role_code=role_code))
                if not permissions.issubset(assignable_permissions):
                    raise AccessManagementError("PLATFORM_ROLE_ESCALATION", f"不能分配超出当前管理员权限的角色：{role_code}", status_code=403)
            await uow.access.delete_platform_user_roles(user_id=user_id)
            for role_code in normalized:
                await uow.access.upsert_platform_user_role(
                    user_id=user_id, role_code=role_code, status="ACTIVE", actor_id=actor_id
                )
            await uow.commit()
            return {"user_id": user_id, "platform_roles": list(normalized)}

    async def revoke_platform_app_grant(
        self, *, user_id: str, app_id: str, actor_id: str, actor_platform_permissions: frozenset[str],
    ) -> dict[str, object]:
        if user_id.casefold() == actor_id.casefold():
            raise AccessManagementError("APP_GRANT_SELF_ESCALATION", "不能管理自己的 App 授权", status_code=403)
        limit = self._app_grant_permission_limit(app_id, actor_platform_permissions)
        async with self._uow_factory() as uow:
            user = await uow.access.get_user_for_update(user_id)
            member = await uow.access.get_app_member(app_id=app_id, user_id=user_id)
            if user is None or member is None:
                raise AccessManagementError("APP_GRANT_NOT_FOUND", "平台用户 App Grant 不存在", status_code=404)
            if user.account_origin != "PLATFORM" or member.member_source != "PLATFORM_GRANT":
                raise AccessManagementError("PLATFORM_GRANT_REQUIRED", "只能撤销平台用户的显式 App Grant", status_code=409)
            _assert_mutable_user(user)
            await self._assert_existing_app_roles_assignable(
                uow=uow, app_id=app_id, user_id=user_id,
                assignable_permissions=limit,
            )
            await uow.access.delete_app_member(member=member)
            await uow.commit()
            return {"user_id": user_id, "app_id": app_id, "revoked": True}

    async def assign_app_domain(self, *, app_id: str, domain_id: int, actor_id: str) -> dict[str, object]:
        async with self._uow_factory() as uow:
            if app_id == "knowledge_retrieval":
                await uow.access.lock_application(app_id)
            await self._require_app(uow=uow, app_id=app_id, require_active=False)
            domain = await uow.domains.get(domain_id=domain_id)
            if domain is None:
                raise AccessManagementError("DOMAIN_NOT_FOUND", "Domain 不存在", status_code=404)
            row = await uow.access.get_app_domain(app_id=app_id, domain_id=domain_id)
            if row is None:
                row = AppDomainEntity(app_id=app_id, domain_id=domain_id, status="ACTIVE", created_by=actor_id)
                await uow.access.add_app_domain(row)
            else:
                row.status = "ACTIVE"
            await uow.commit()
            return {"app_id": app_id, "domain_id": domain_id, "status": row.status}

    async def set_application_status(self, *, app_id: str, status: str) -> dict[str, object]:
        _assert_status(status)
        async with self._uow_factory() as uow:
            app = await self._require_app(uow=uow, app_id=app_id, require_active=False)
            await uow.access.update_application_status(app=app, status=status)
            await uow.commit()
            return self._app_item(app)

    async def list_applications(self) -> list[dict[str, object]]:
        async with self._uow_factory() as uow:
            return [self._app_item(row) for row in await uow.access.list_applications()]

    async def list_app_grant_options(self, *, actor_permissions: frozenset[str]) -> list[dict[str, object]]:
        """读取真实 App、角色和业务空间；仅标出当前管理员可授予的角色。"""
        async with self._uow_factory() as uow:
            result = []
            can_grant = "platform:app_grant_manage" in actor_permissions
            for app in await uow.access.list_applications():
                if app.app_id == "platform":
                    continue
                app_id = app.app_id
                limit = self._app_grant_permission_limit(app_id, actor_permissions) if can_grant else frozenset()
                roles = []
                for role in await uow.access.list_all_roles(app_id=app_id):
                    manageable = can_grant and (
                        limit is None or await self._effective_role_permissions(uow=uow, app_id=app_id, role_code=role.role_code) <= limit
                    )
                    roles.append({
                        "role_code": role.role_code, "display_name": role.display_name,
                        "scope_policy": role.scope_policy, "status": role.status,
                        "manageable": manageable,
                    })
                app_domains = {int(row.domain_id): row.status for row in await uow.access.list_app_domains(app_id=app_id)}
                domains = await uow.domains.list_by_ids(domain_ids=tuple(app_domains)) if app_domains else []
                result.append({
                    **self._app_item(app), "manageable": can_grant, "roles": roles,
                    "domains": [{"domain_id": str(row.domain_id), "name": row.name,
                                 "active": row.status == app_domains[int(row.domain_id)] == "ACTIVE"} for row in domains],
                })
            return result

    async def update_user(
        self, *, user_id: str, display_name: str | None, display_name_provided: bool,
        status: str | None, max_security_level: int | None = None,
        actor_security_level: int = 3, actor_id: str,
    ) -> dict[str, object]:
        if is_reserved_global_admin(user_id):
            raise AccessManagementError("GLOBAL_ADMIN_PROTECTED", "ADMIN 是平台保留账号，不能修改", status_code=409)
        if status is not None:
            _assert_status(status)
        if max_security_level is not None:
            _assert_security_level(max_security_level)
            if max_security_level > actor_security_level:
                raise AccessManagementError("USER_SECURITY_LEVEL_DENIED", "不能授予高于当前管理员的安全等级", status_code=403)
        async with self._uow_factory() as uow:
            user = await uow.access.get_user_for_update(user_id)
            if user is None:
                raise AccessManagementError("USER_NOT_FOUND", "用户不存在", status_code=404)
            _assert_mutable_user(user)
            if user.account_origin != "PLATFORM":
                raise AccessManagementError("USER_OWNERSHIP_DENIED", "用户不属于当前管理范围", status_code=403)
            await self._assert_platform_target_dominated(uow=uow, actor_id=actor_id, target=user)
            await uow.access.update_user(
                user=user,
                display_name=display_name if display_name_provided else user.display_name,
                status=status or user.status,
                max_security_level=max_security_level if max_security_level is not None else int(user.max_security_level),
            )
            await uow.commit()
            return self._user_item(user)

    async def reset_platform_user_password(
        self, *, actor_id: str, user_id: str, password: str,
        must_change_password: bool,
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            target = await uow.access.get_user_for_update(user_id)
            if target is None or target.account_origin != "PLATFORM":
                raise AccessManagementError(
                    "PLATFORM_USER_REQUIRED", "平台用户不存在", status_code=404
                )
            await self._assert_platform_target_dominated(
                uow=uow,
                actor_id=actor_id,
                target=target,
            )
            result = await self._set_password(
                uow=uow,
                user_id=user_id,
                password=password,
                must_change_password=must_change_password,
            )
            await uow.commit()
            return result

    async def reset_initial_app_admin_password(
        self, *, app_id: str, password: str, must_change_password: bool
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            members = await uow.access.list_app_members(app_id=app_id)
            initial = next((row for row in members if row.is_initial_admin == "Y"), None)
            if initial is None:
                raise AccessManagementError("INITIAL_APP_ADMIN_NOT_FOUND", "App 初始管理员不存在", status_code=404)
            result = await self._set_password(
                uow=uow,
                user_id=initial.user_id,
                password=password,
                must_change_password=must_change_password,
            )
            await uow.commit()
            return result

    async def delete_user(
        self, *, user_id: str, actor_id: str,
    ) -> dict[str, object]:
        if is_reserved_global_admin(user_id):
            raise AccessManagementError("GLOBAL_ADMIN_PROTECTED", "ADMIN 是平台保留账号，不能删除", status_code=409)
        async with self._uow_factory() as uow:
            user = await uow.access.get_user_for_update(user_id)
            if user is None:
                raise AccessManagementError("USER_NOT_FOUND", "用户不存在", status_code=404)
            _assert_mutable_user(user)
            if user.account_origin != "PLATFORM":
                raise AccessManagementError("USER_OWNERSHIP_DENIED", "用户不属于当前管理范围", status_code=403)
            await self._assert_platform_target_dominated(uow=uow, actor_id=actor_id, target=user)
            binding = await personal_binding(uow.external_delivery, scope=DeliveryScope(), user_id=user_id)
            if binding:
                await set_personal_binding(
                    uow.external_delivery, scope=DeliveryScope(), user_id=user_id, actor_id=actor_id, open_id=None,
                    expected_binding_id=binding["id"], expected_version=binding["row_version"],
                )
            await uow.access.delete_user(user=user)
            await uow.commit()
            return {"user_id": user_id, "deleted": True}

    async def list_permissions(self, *, app_id: str | None) -> list[dict[str, object]]:
        async with self._uow_factory() as uow:
            rows = await uow.access.list_permissions(app_id=app_id)
        metadata = {
            item.code: item
            for item in app_permission_catalog(app_id)
        } if app_id and app_id != "platform" else {}
        return [{
            "app_id": row.app_id,
            "permission_code": row.permission_code,
            "display_name": row.display_name,
            **({
                "default_tier": metadata[row.permission_code].default_tier.value,
                "custom_assignable": metadata[row.permission_code].custom_assignable,
            } if row.permission_code in metadata else {}),
        } for row in rows]

    async def list_roles(self, *, app_id: str | None) -> list[dict[str, object]]:
        async with self._uow_factory() as uow:
            result = []
            for row in await uow.access.list_all_roles(app_id=app_id):
                permissions = await uow.access.list_role_permission_codes(app_id=row.app_id, role_code=row.role_code)
                member_count = await uow.access.count_active_role_members(
                    app_id=row.app_id,
                    role_code=row.role_code,
                )
                result.append(self._role_item(
                    row, permissions, member_count=member_count
                ))
            return result

    async def create_role(
        self, *, app_id: str, role_code: str, display_name: str,
        permission_codes: tuple[str, ...], assignable_permissions: frozenset[str],
        actor_id: str,
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            await self._require_app(uow=uow, app_id=app_id)
            if app_id != "platform" and role_code in BUILT_IN_APP_ROLE_CODES:
                raise AccessManagementError(
                    "SYSTEM_ROLE_CODE_RESERVED",
                    "统一系统角色编码不能用于自定义角色",
                    status_code=409,
                )
            if await uow.access.get_role(app_id=app_id, role_code=role_code):
                raise AccessManagementError("ROLE_ALREADY_EXISTS", "应用角色已存在", status_code=409)
            await self._validate_permissions(uow=uow, app_id=app_id, permission_codes=permission_codes)
            self._assert_assignable_permissions(
                app_id=app_id,
                permission_codes=permission_codes,
                assignable_permissions=assignable_permissions,
            )
            role = AppRoleEntity(
                app_id=app_id, role_code=role_code, display_name=display_name,
                is_system="N",
                scope_policy="PLATFORM" if app_id == "platform" else "SELECTABLE",
                status="ACTIVE", row_version=1,
            )
            await uow.access.add_role(role)
            await uow.access.replace_role_permissions(app_id=app_id, role_code=role_code, permission_codes=permission_codes)
            await self._audit_role_change(
                uow=uow, actor_id=actor_id, action="ROLE_CREATED", role=role,
                permission_codes=permission_codes,
            )
            await uow.commit()
            return self._role_item(role, permission_codes)

    async def update_role(
        self, *, app_id: str, role_code: str, display_name: str, status: str,
        permission_codes: tuple[str, ...], assignable_permissions: frozenset[str],
        actor_id: str, expected_version: int | None = None,
    ) -> dict[str, object]:
        _assert_status(status)
        if app_id != "platform" and status != "ACTIVE":
            raise AccessManagementError(
                "ROLE_STATUS_INVALID", "App 自定义角色只能通过物理删除下线"
            )
        async with self._uow_factory() as uow:
            role = await uow.access.get_role_for_update(
                app_id=app_id, role_code=role_code
            )
            if role is None:
                raise AccessManagementError("ROLE_NOT_FOUND", "应用角色不存在", status_code=404)
            self._assert_mutable_role(role)
            self._assert_expected_version(role, expected_version)
            await self._validate_permissions(uow=uow, app_id=app_id, permission_codes=permission_codes)
            self._assert_assignable_permissions(
                app_id=app_id,
                permission_codes=permission_codes,
                assignable_permissions=assignable_permissions,
            )
            await uow.access.update_role(
                role=role,
                display_name=display_name,
                status=status if app_id == "platform" else None,
            )
            await uow.access.replace_role_permissions(app_id=app_id, role_code=role_code, permission_codes=permission_codes)
            await self._audit_role_change(
                uow=uow, actor_id=actor_id, action="ROLE_UPDATED", role=role,
                permission_codes=permission_codes,
            )
            await uow.commit()
            return self._role_item(role, permission_codes)

    async def delete_role(
        self, *, app_id: str, role_code: str,
        actor_id: str, expected_version: int | None = None,
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            role = await uow.access.get_role_for_update(
                app_id=app_id, role_code=role_code
            )
            if role is None:
                raise AccessManagementError("ROLE_NOT_FOUND", "应用角色不存在", status_code=404)
            self._assert_mutable_role(role)
            self._assert_expected_version(role, expected_version)
            permission_codes = await uow.access.list_role_permission_codes(
                app_id=app_id, role_code=role_code
            )
            if app_id == "platform":
                await uow.access.update_role(
                    role=role,
                    display_name=role.display_name,
                    status="DISABLED",
                )
                affected_member_count = 0
            else:
                affected_member_count = await uow.access.delete_role(role=role)
            await self._audit_role_change(
                uow=uow, actor_id=actor_id, action="ROLE_DELETED", role=role,
                permission_codes=permission_codes,
                affected_member_count=affected_member_count,
            )
            await uow.commit()
            return {
                "app_id": app_id,
                "role_code": role_code,
                "status": "DISABLED" if app_id == "platform" else "DELETED",
                "deleted": True,
                "affected_member_count": affected_member_count,
            }

    @staticmethod
    async def _audit_role_change(
        *, uow, actor_id: str, action: str, role: AppRoleEntity,
        permission_codes: tuple[str, ...], affected_member_count: int = 0,
    ) -> None:
        await uow.access.add_access_audit_event(AccessAuditEventEntity(
            event_id=str(uuid7()), actor_id=actor_id, app_id=role.app_id,
            action=action, resource_type="APP_ROLE", resource_id=role.role_code,
            summary_json=json.dumps({
                "display_name": role.display_name,
                "permissions": sorted(permission_codes),
                "row_version": int(role.row_version),
            }, ensure_ascii=False, separators=(",", ":")),
            affected_member_count=affected_member_count,
        ))

    @staticmethod
    async def _require_app(*, uow, app_id: str, require_active: bool = True):
        app = await uow.access.get_application(app_id)
        if app is None:
            raise AccessManagementError("APP_NOT_FOUND", "App 不存在", status_code=404)
        if require_active and app.status != "ACTIVE":
            raise AccessManagementError("APP_DISABLED", "App 已停用", status_code=409)
        return app

    @staticmethod
    def _app_grant_permission_limit(app_id: str, actor_permissions: frozenset[str]) -> frozenset[str] | None:
        if "platform:app_grant_manage" not in actor_permissions:
            raise AccessManagementError("PLATFORM_PERMISSION_DENIED", "缺少平台 App 授权管理权限", status_code=403)
        return None if "platform:app_manage" in actor_permissions else frozenset({f"{app_id}:use"})

    @staticmethod
    async def _effective_role_permissions(*, uow, app_id: str, role_code: str) -> frozenset[str]:
        permissions = set(await uow.access.list_role_permission_codes(app_id=app_id, role_code=role_code))
        if role_code in BUILT_IN_APP_ROLE_CODES:
            permissions.update(built_in_role_permissions(app_id, role_code))
        return frozenset(permissions)

    async def _assert_existing_app_roles_assignable(
        self, *, uow, app_id: str, user_id: str, assignable_permissions: frozenset[str] | None,
    ) -> None:
        if assignable_permissions is None:
            return
        for binding in await uow.access.list_member_roles(app_id=app_id, user_id=user_id):
            permissions = await self._effective_role_permissions(uow=uow, app_id=app_id, role_code=binding.role_code)
            if not permissions <= assignable_permissions:
                raise AccessManagementError("ROLE_ASSIGNMENT_ESCALATION", "现有角色超出当前管理员的可授权范围", status_code=403)

    async def _replace_role_bindings(
        self, *, uow, app_id: str, user_id: str,
        role_bindings: tuple[dict[str, object], ...], actor_id: str,
        assignable_permissions: frozenset[str] | None = None,
    ) -> list[dict[str, object]]:
        if not role_bindings:
            raise AccessManagementError("ROLE_BINDING_REQUIRED", "至少需要分配一个 App 角色")
        role_codes = [str(item["role_code"]) for item in role_bindings]
        if len(role_codes) != len(set(role_codes)):
            raise AccessManagementError("DUPLICATE_ROLE_BINDING", "App 角色不能重复")
        validated = []
        for item in role_bindings:
            role_code = str(item["role_code"])
            scope_mode = str(item.get("scope_mode") or "SELECTED_DOMAINS")
            domain_ids = tuple(dict.fromkeys(int(value) for value in item.get("domain_ids", ())))
            if scope_mode not in {"ALL_APP_DOMAINS", "SELECTED_DOMAINS"}:
                raise AccessManagementError("INVALID_SCOPE_MODE", "角色 Domain 范围模式无效")
            if scope_mode == "SELECTED_DOMAINS" and not domain_ids:
                raise AccessManagementError("DOMAIN_SCOPE_REQUIRED", "指定 Domain 模式至少需要一个 Domain")
            if scope_mode == "ALL_APP_DOMAINS":
                domain_ids = ()
            role = await uow.access.get_role(app_id=app_id, role_code=role_code)
            if role is None or role.status != "ACTIVE":
                raise AccessManagementError("ROLE_NOT_FOUND", f"App 角色不存在或已停用：{role_code}", status_code=404)
            if role.scope_policy == "ALL_APP_DOMAINS" and scope_mode != "ALL_APP_DOMAINS":
                raise AccessManagementError(
                    "ROLE_SCOPE_REQUIRED", "该角色必须覆盖全部 App Domain"
                )
            if role.scope_policy == "SELECTABLE" and scope_mode != "SELECTED_DOMAINS":
                raise AccessManagementError(
                    "ROLE_SCOPE_ESCALATION", "该角色只能选择具体 Domain"
                )
            if assignable_permissions is not None:
                role_permissions = await self._effective_role_permissions(uow=uow, app_id=app_id, role_code=role_code)
                if not role_permissions.issubset(assignable_permissions):
                    raise AccessManagementError(
                        "ROLE_ASSIGNMENT_ESCALATION",
                        f"不能分配超出当前管理员权限的角色：{role_code}",
                        status_code=403,
                    )
            for domain_id in domain_ids:
                app_domain = await uow.access.get_app_domain(app_id=app_id, domain_id=domain_id)
                domain = await uow.domains.get(domain_id=domain_id) if app_domain and app_domain.status == "ACTIVE" else None
                if domain is None or domain.status != "ACTIVE":
                    raise AccessManagementError("APP_DOMAIN_NOT_FOUND", f"Domain 不属于当前 App 或已停用：{domain_id}", status_code=404)
            validated.append((role_code, scope_mode, domain_ids))
        await uow.access.delete_member_authorizations(app_id=app_id, user_id=user_id)
        result = []
        for role_code, scope_mode, domain_ids in validated:
            await uow.access.upsert_member_role(
                app_id=app_id, user_id=user_id, role_code=role_code,
                scope_mode=scope_mode, status="ACTIVE", actor_id=actor_id,
            )
            await uow.access.replace_member_role_scopes(
                app_id=app_id, user_id=user_id, role_code=role_code, domain_ids=domain_ids
            )
            result.append({"role_code": role_code, "scope_mode": scope_mode, "domain_ids": list(domain_ids), "status": "ACTIVE"})
        return result

    async def _binding_items(self, *, uow, app_id: str, user_id: str) -> list[dict[str, object]]:
        rows = await uow.access.list_member_roles(app_id=app_id, user_id=user_id)
        result = []
        for row in rows:
            scopes = await uow.access.list_member_role_scopes(app_id=app_id, user_id=user_id, role_code=row.role_code)
            result.append({"role_code": row.role_code, "scope_mode": row.scope_mode, "domain_ids": [str(scope) for scope in scopes], "status": row.status})
        return result

    async def _set_password(
        self, *, uow, user_id: str, password: str,
        must_change_password: bool,
    ) -> dict[str, object]:
        """在调用方事务内写入密码，避免校验与修改分离。"""

        password_hash = await self._password_hash(password)
        credential = await uow.access.get_user_credential(user_id)
        if credential is None:
            await uow.access.add_user_credential(PlatformUserCredentialEntity(
                user_id=user_id,
                password_hash=password_hash,
                must_change_password="Y" if must_change_password else "N",
            ))
        else:
            await uow.access.set_user_password(
                credential=credential,
                password_hash=password_hash,
                must_change_password=must_change_password,
            )
        return {
            "user_id": user_id,
            "must_change_password": must_change_password,
        }

    async def _assert_app_target_dominated(
        self, *, uow, app_id: str, actor_id: str, target
    ) -> None:
        """按保留的角色范围校验，避免借助停用或暂不可用的授权重置高权限账号。"""
        target_bindings = await uow.access.list_member_roles(app_id=app_id, user_id=target.user_id)
        if not target_bindings:
            return
        actor_domains = set(await uow.access.list_active_domain_ids(actor_id, app_id=app_id))
        actor_bindings = await uow.access.list_member_roles(app_id=app_id, user_id=actor_id)
        for binding in target_bindings:
            permissions = await self._effective_role_permissions(uow=uow, app_id=app_id, role_code=binding.role_code)
            if binding.scope_mode == "ALL_APP_DOMAINS":
                domain_ids = tuple(row.domain_id for row in await uow.access.list_app_domains(app_id=app_id))
                candidate_roles = {row.role_code for row in actor_bindings
                                   if row.status == "ACTIVE" and row.scope_mode == "ALL_APP_DOMAINS"}
            else:
                domain_ids = await uow.access.list_member_role_scopes(
                    app_id=app_id, user_id=target.user_id, role_code=binding.role_code,
                )
                candidate_roles = None
            if not domain_ids or not set(domain_ids) <= actor_domains:
                raise AccessManagementError(
                    "PASSWORD_RESET_SCOPE_ESCALATION", "目标账号包含当前管理员无权管理的业务空间", status_code=403,
                )
            for domain_id in domain_ids:
                roles = await uow.access.list_roles(app_id=app_id, domain_id=domain_id, user_id=actor_id)
                allowed: set[str] = set()
                for code in roles:
                    if candidate_roles is None or code in candidate_roles:
                        allowed.update(await self._effective_role_permissions(uow=uow, app_id=app_id, role_code=code))
                if not permissions <= allowed:
                    raise AccessManagementError(
                        "PASSWORD_RESET_PRIVILEGE_ESCALATION", "不能操作业务权限高于当前管理员的账号", status_code=403,
                    )

    async def _assert_platform_target_dominated(
        self, *, uow, actor_id: str, target
    ) -> None:
        """确保平台管理员完整覆盖目标平台账号的全部有效权限。"""

        _assert_mutable_user(target)
        if actor_id.casefold() == target.user_id.casefold():
            raise AccessManagementError(
                "PASSWORD_RESET_SELF_DENIED",
                "管理员重置接口不能用于修改自己的密码",
                status_code=403,
            )
        if is_reserved_global_admin(actor_id):
            return
        actor = await uow.access.get_user(actor_id)
        if actor is None or actor.status != "ACTIVE":
            raise AccessManagementError(
                "ACTOR_NOT_FOUND", "当前管理员账号不可用", status_code=403
            )
        if int(target.max_security_level) > int(actor.max_security_level):
            raise AccessManagementError(
                "PASSWORD_RESET_PRIVILEGE_ESCALATION",
                "不能重置安全等级高于当前管理员的账号密码",
                status_code=403,
            )
        actor_platform = await uow.access.platform_permissions_for(
            user_id=actor_id
        )
        target_platform: set[str] = set()
        for binding in await uow.access.list_platform_user_roles(user_id=target.user_id):
            target_platform.update(await uow.access.list_role_permission_codes(app_id="platform", role_code=binding.role_code))
        if not set(target_platform).issubset(set(actor_platform)):
            raise AccessManagementError(
                "PASSWORD_RESET_PRIVILEGE_ESCALATION",
                "不能重置平台权限高于当前管理员的账号密码",
                status_code=403,
            )
        for membership in await uow.access.list_user_memberships(
            user_id=target.user_id
        ):
            await self._assert_app_target_dominated(
                uow=uow,
                app_id=membership.app_id,
                actor_id=actor_id,
                target=target,
            )

    @staticmethod
    async def _validate_permissions(*, uow, app_id: str, permission_codes):
        if len(permission_codes) != len(set(permission_codes)):
            raise AccessManagementError("DUPLICATE_PERMISSION", "角色权限不能重复")
        if app_id != "platform":
            try:
                policy = get_app_authorization_policy(app_id)
            except ValueError as exc:
                raise AccessManagementError(
                    "APP_AUTHORIZATION_POLICY_MISSING", str(exc)
                ) from exc
            try:
                policy.assert_role_permissions(permission_codes)
            except ValueError as exc:
                raise AccessManagementError(
                    "APP_USE_PERMISSION_REQUIRED", str(exc)
                ) from exc
        allowed = {row.permission_code for row in await uow.access.list_permissions(app_id=app_id)}
        unknown = sorted(set(permission_codes) - allowed)
        if unknown:
            raise AccessManagementError("INVALID_ROLE_PERMISSION", f"权限不属于应用 {app_id}：{', '.join(unknown)}")

    @staticmethod
    def _assert_mutable_role(role: AppRoleEntity) -> None:
        if role.is_system == "Y":
            raise AccessManagementError("SYSTEM_ROLE_PROTECTED", "系统角色不能修改或删除", status_code=409)

    @staticmethod
    def _assert_assignable_permissions(
        *, app_id: str, permission_codes: tuple[str, ...],
        assignable_permissions: frozenset[str],
    ) -> None:
        catalog_assignable = (
            {
                item.code
                for item in app_permission_catalog(app_id)
                if item.custom_assignable
            }
            if app_id != "platform"
            else set(assignable_permissions)
        )
        excess = sorted(
            set(permission_codes)
            - assignable_permissions.intersection(catalog_assignable)
        )
        if excess:
            raise AccessManagementError(
                "ROLE_PERMISSION_ESCALATION",
                "不能配置当前管理员不具备的权限：" + ", ".join(excess),
                status_code=403,
            )

    @staticmethod
    def _assert_expected_version(
        role: AppRoleEntity, expected_version: int | None
    ) -> None:
        if expected_version is not None and int(role.row_version) != expected_version:
            raise AccessManagementError(
                "ROLE_VERSION_CONFLICT",
                "角色已被其他操作修改，请刷新后重试",
                status_code=412,
            )

    @staticmethod
    def _user_item(user: PlatformUserEntity) -> dict[str, object]:
        return {
            "user_id": user.user_id, "display_name": user.display_name,
            "account_origin": user.account_origin, "owner_app_id": user.owner_app_id,
            "max_security_level": int(user.max_security_level), "status": user.status,
            "protected": user.is_protected == "Y" or is_reserved_global_admin(user.user_id),
            "created_at": user.created_at, "updated_at": user.updated_at,
        }

    @staticmethod
    def _member_item(member: AppMemberEntity) -> dict[str, object]:
        return {
            "app_id": member.app_id, "status": member.status,
            "member_source": member.member_source,
            "initial_admin": member.is_initial_admin == "Y",
        }

    @staticmethod
    def _role_item(
        role: AppRoleEntity, permissions: tuple[str, ...],
        *, member_count: int = 0,
    ) -> dict[str, object]:
        return {
            "app_id": role.app_id, "role_code": role.role_code,
            "display_name": role.display_name, "status": role.status,
            "protected": role.is_system == "Y", "scope_policy": role.scope_policy,
            "permissions": list(permissions), "row_version": int(role.row_version),
            "member_count": member_count,
        }

    @staticmethod
    def _app_item(app) -> dict[str, object]:
        return {
            "app_id": app.app_id, "display_name": app.display_name,
            "status": app.status, "member_assignable": app.member_assignable == "Y",
            "row_version": int(app.row_version),
        }


__all__ = ["AccessManagementError", "AccessManagementService", "INITIAL_APP_ADMIN_ROLE"]
