"""平台 Domain 生命周期应用服务。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy.exc import IntegrityError

from main_api.entities import AppDomainEntity, PlatformDomainEntity

BOOTSTRAP_DOMAIN_NAMES = frozenset({
    "knowledge_retrieval_portal",
    "media_studio_portal",
})


class DomainConflictError(RuntimeError):
    """Domain 名称与当前应用内已有记录冲突。"""


class DomainLifecycleError(RuntimeError):
    """App 委托的 Domain 生命周期请求不符合平台边界。"""

    def __init__(self, code: str, message: str, *, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class DomainManagementService:
    """管理不隶属于具体 Domain 的平台级 Domain 注册信息。"""

    def __init__(self, *, uow_factory: Callable[[], Any]):
        self._uow_factory = uow_factory

    async def create(
        self,
        *,
        name: str,
        description: str | None,
        actor_id: str,
    ) -> dict[str, Any]:
        normalized_name = name.strip()
        async with self._uow_factory() as uow:
            actor = await uow.access.get_user(actor_id)
            if actor is None:
                if actor_id.strip().casefold() == "admin":
                    raise DomainConflictError(
                        "ADMIN 是平台保留账号，只能通过项目初始化脚本创建"
                    )
                raise DomainConflictError("平台用户不存在或已停用")
            if actor.status != "ACTIVE" or actor.account_origin != "PLATFORM":
                raise DomainConflictError("只有启用的平台用户可以创建 Domain")
            entity = await self._insert_domain(
                uow=uow,
                name=normalized_name,
                description=description,
                actor_id=actor_id,
            )
            await uow.commit()
            return self._view(entity)

    async def create_for_app(
        self,
        *,
        app_id: str,
        name: str,
        description: str | None,
        actor_id: str,
    ) -> dict[str, Any]:
        """由 App 委托平台原子创建 Domain、绑定 App-Domain，并保证创建人可访问。"""
        normalized_name = _normalized_name(name)
        normalized_description = _normalized_description(description)
        async with self._uow_factory() as uow:
            await self._require_app_actor(uow=uow, app_id=app_id, actor_id=actor_id)
            entity = await self._insert_domain(
                uow=uow,
                name=normalized_name,
                description=normalized_description,
                actor_id=actor_id,
            )
            await self._bind_app_domain(
                uow=uow,
                app_id=app_id,
                domain_id=int(entity.domain_id),
                actor_id=actor_id,
            )
            await self._ensure_creator_access(
                uow=uow,
                app_id=app_id,
                user_id=actor_id,
                domain_id=int(entity.domain_id),
            )
            await uow.commit()
            return self._view(entity)

    async def list_for_app(
        self,
        *,
        app_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """列出当前用户在指定 App 中已获授权的 Domain，含已停用记录。"""
        async with self._uow_factory() as uow:
            domain_ids = await self._authorized_domain_ids(
                uow=uow, app_id=app_id, user_id=user_id
            )
            domains = await uow.domains.list_by_ids(domain_ids=domain_ids)
            return {"items": [self._view(row) for row in domains]}

    async def get_for_app(
        self,
        *,
        app_id: str,
        domain_id: int,
        user_id: str,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            entity = await self._require_managed_domain(
                uow=uow, app_id=app_id, domain_id=domain_id, user_id=user_id
            )
            return self._view(entity)

    async def update_for_app(
        self,
        *,
        app_id: str,
        domain_id: int,
        user_id: str,
        actor_id: str,
        expected_row_version: int,
        name: str | None = None,
        description: str | None = None,
        description_set: bool = False,
        status: str | None = None,
    ) -> dict[str, Any]:
        """更新 App 已授权 Domain 的名称、说明或启停状态。"""
        if status is not None and status not in {"ACTIVE", "DISABLED"}:
            raise DomainLifecycleError("DOMAIN_STATUS_INVALID", "Domain 状态只能是 ACTIVE 或 DISABLED")
        normalized_name = _normalized_name(name) if name is not None else None
        async with self._uow_factory() as uow:
            await self._require_app_actor(uow=uow, app_id=app_id, actor_id=actor_id)
            entity = await self._require_managed_domain(
                uow=uow, app_id=app_id, domain_id=domain_id, user_id=user_id
            )
            if int(entity.row_version) != expected_row_version:
                raise DomainLifecycleError(
                    "DOMAIN_VERSION_CONFLICT",
                    "Domain 已被其他请求修改",
                    status_code=409,
                )
            changed = False
            if normalized_name is not None and normalized_name != entity.name:
                self._reject_bootstrap_mutation(entity, action="rename")
                existing = await uow.domains.get_by_name(name=normalized_name)
                if existing is not None and int(existing.domain_id) != int(entity.domain_id):
                    raise DomainConflictError("当前应用已存在同名 Domain")
                entity.name = normalized_name
                changed = True
            if description_set:
                normalized_description = _normalized_description(description)
                if normalized_description != entity.description:
                    entity.description = normalized_description
                    changed = True
            if status is not None and status != entity.status:
                if status == "DISABLED":
                    self._reject_bootstrap_mutation(entity, action="disable")
                entity.status = status
                await self._set_app_domain_status(
                    uow=uow,
                    app_id=app_id,
                    domain_id=int(entity.domain_id),
                    status=status,
                    actor_id=actor_id,
                )
                changed = True
            if changed:
                entity.row_version = int(entity.row_version) + 1
                entity.updated_by = actor_id
            await uow.commit()
            return self._view(entity)

    async def disable_for_app(
        self,
        *,
        app_id: str,
        domain_id: int,
        user_id: str,
        actor_id: str,
        expected_row_version: int,
    ) -> dict[str, Any]:
        """停用 App Domain；引导域不可停用。"""
        return await self.update_for_app(
            app_id=app_id,
            domain_id=domain_id,
            user_id=user_id,
            actor_id=actor_id,
            expected_row_version=expected_row_version,
            status="DISABLED",
        )

    async def _insert_domain(
        self,
        *,
        uow: Any,
        name: str,
        description: str | None,
        actor_id: str,
    ) -> PlatformDomainEntity:
        existing = await uow.domains.get_by_name(name=name)
        if existing is not None:
            raise DomainConflictError("当前应用已存在同名 Domain")
        entity = PlatformDomainEntity(
            name=name,
            status="ACTIVE",
            description=description,
            row_version=1,
            created_by=actor_id,
            updated_by=actor_id,
        )
        try:
            await uow.domains.add(entity)
        except IntegrityError as exc:
            raise DomainConflictError("当前应用已存在同名 Domain") from exc
        if getattr(entity, "domain_id", None) is None:
            raise DomainLifecycleError(
                "DOMAIN_ID_REQUIRED",
                "平台未能生成 Domain 标识",
                status_code=500,
            )
        return entity

    @staticmethod
    async def _require_app_actor(*, uow: Any, app_id: str, actor_id: str) -> None:
        app = await uow.access.get_application(app_id)
        if app is None:
            raise DomainLifecycleError("APP_NOT_FOUND", "App 不存在", status_code=404)
        if app.status != "ACTIVE":
            raise DomainLifecycleError("APP_DISABLED", "App 已停用", status_code=409)
        actor = await uow.access.get_user(actor_id)
        if actor is None or actor.status != "ACTIVE":
            raise DomainLifecycleError(
                "USER_DISABLED", "用户不存在或已停用", status_code=403
            )
        member = await uow.access.get_app_member(app_id=app_id, user_id=actor_id)
        if member is None or member.status != "ACTIVE":
            raise DomainLifecycleError(
                "APP_MEMBER_REQUIRED",
                "当前用户不是该 App 的有效成员",
                status_code=403,
            )

    @classmethod
    async def _require_managed_domain(
        cls,
        *,
        uow: Any,
        app_id: str,
        domain_id: int,
        user_id: str,
    ) -> PlatformDomainEntity:
        domain_ids = await cls._authorized_domain_ids(
            uow=uow, app_id=app_id, user_id=user_id
        )
        if domain_id not in domain_ids:
            raise DomainLifecycleError(
                "DOMAIN_NOT_FOUND", "Domain 不存在或无权访问", status_code=404
            )
        entity = await uow.domains.get(domain_id=domain_id)
        if entity is None:
            raise DomainLifecycleError(
                "DOMAIN_NOT_FOUND", "Domain 不存在或无权访问", status_code=404
            )
        return entity

    @staticmethod
    async def _authorized_domain_ids(
        *,
        uow: Any,
        app_id: str,
        user_id: str,
    ) -> tuple[int, ...]:
        member = await uow.access.get_app_member(app_id=app_id, user_id=user_id)
        if member is None or member.status != "ACTIVE":
            return ()
        roles = [
            row
            for row in await uow.access.list_member_roles(app_id=app_id)
            if row.user_id == user_id and row.status == "ACTIVE"
        ]
        if any(row.scope_mode == "ALL_APP_DOMAINS" for row in roles):
            return tuple(
                int(row.domain_id)
                for row in await uow.access.list_app_domains(app_id=app_id)
            )
        domain_ids: list[int] = []
        for row in roles:
            if row.scope_mode != "SELECTED_DOMAINS":
                continue
            domain_ids.extend(
                int(value)
                for value in await uow.access.list_member_role_scopes(
                    app_id=app_id, user_id=user_id, role_code=row.role_code
                )
            )
        return tuple(dict.fromkeys(domain_ids))

    @staticmethod
    async def _bind_app_domain(
        *,
        uow: Any,
        app_id: str,
        domain_id: int,
        actor_id: str,
    ) -> None:
        existing_link = await uow.access.get_app_domain(
            app_id=app_id, domain_id=domain_id
        )
        if existing_link is None:
            await uow.access.add_app_domain(
                AppDomainEntity(
                    app_id=app_id,
                    domain_id=domain_id,
                    status="ACTIVE",
                    created_by=actor_id,
                )
            )
            return
        if existing_link.status != "ACTIVE":
            existing_link.status = "ACTIVE"

    @staticmethod
    async def _set_app_domain_status(
        *,
        uow: Any,
        app_id: str,
        domain_id: int,
        status: str,
        actor_id: str,
    ) -> None:
        existing_link = await uow.access.get_app_domain(
            app_id=app_id, domain_id=domain_id
        )
        if existing_link is None:
            if status != "ACTIVE":
                return
            await uow.access.add_app_domain(
                AppDomainEntity(
                    app_id=app_id,
                    domain_id=domain_id,
                    status="ACTIVE",
                    created_by=actor_id,
                )
            )
            return
        existing_link.status = status

    @staticmethod
    async def _ensure_creator_access(
        *,
        uow: Any,
        app_id: str,
        user_id: str,
        domain_id: int,
    ) -> None:
        roles = [
            row
            for row in await uow.access.list_member_roles(app_id=app_id)
            if row.user_id == user_id and row.status == "ACTIVE"
        ]
        if any(row.scope_mode == "ALL_APP_DOMAINS" for row in roles):
            return
        selected = [
            row for row in roles if row.scope_mode == "SELECTED_DOMAINS"
        ]
        if not selected:
            raise DomainLifecycleError(
                "DOMAIN_SCOPE_REQUIRED",
                "当前用户没有可写入的 Domain 授权范围",
                status_code=403,
            )
        for row in selected:
            scopes = await uow.access.list_member_role_scopes(
                app_id=app_id, user_id=user_id, role_code=row.role_code
            )
            if domain_id in scopes:
                continue
            await uow.access.replace_member_role_scopes(
                app_id=app_id,
                user_id=user_id,
                role_code=row.role_code,
                domain_ids=(*scopes, domain_id),
            )

    @staticmethod
    def _reject_bootstrap_mutation(entity: PlatformDomainEntity, *, action: str) -> None:
        if entity.name not in BOOTSTRAP_DOMAIN_NAMES:
            return
        if action == "rename":
            raise DomainLifecycleError(
                "DOMAIN_BOOTSTRAP_PROTECTED",
                "引导 Domain 的名称由初始化脚本管理，不能修改",
                status_code=409,
            )
        raise DomainLifecycleError(
            "DOMAIN_BOOTSTRAP_PROTECTED",
            "引导 Domain 不能停用或删除",
            status_code=409,
        )

    @staticmethod
    def _view(entity: PlatformDomainEntity) -> dict[str, Any]:
        return {
            "domain_id": int(entity.domain_id),
            "name": entity.name,
            "status": entity.status,
            "description": entity.description,
            "row_version": int(entity.row_version),
            "created_at": _isoformat(getattr(entity, "created_at", None)),
            "updated_at": _isoformat(getattr(entity, "updated_at", None)),
        }


def _normalized_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise DomainLifecycleError("DOMAIN_NAME_REQUIRED", "Domain 名称不能为空")
    return normalized


def _normalized_description(description: str | None) -> str | None:
    if description is None:
        return None
    normalized = description.strip()
    return normalized or None


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
