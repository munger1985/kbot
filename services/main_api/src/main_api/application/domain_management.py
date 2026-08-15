"""平台 Domain 生命周期应用服务。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.exc import IntegrityError

from main_api.entities import PlatformDomainEntity, PlatformUserEntity
from .access_control import is_reserved_global_admin


class DomainConflictError(RuntimeError):
    """Domain 名称与当前应用内已有记录冲突。"""


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
            existing = await uow.domains.get_by_name(
                name=normalized_name,
            )
            if existing is not None:
                raise DomainConflictError("当前应用已存在同名 Domain")
            entity = PlatformDomainEntity(
                name=normalized_name,
                status="ACTIVE",
                description=description,
                row_version=1,
                created_by=actor_id,
                updated_by=actor_id,
            )
            try:
                await uow.domains.add(entity)
                user = await uow.access.get_user(actor_id)
                if is_reserved_global_admin(actor_id) and user is None:
                    raise DomainConflictError(
                        "ADMIN 是平台保留账号，只能通过项目初始化脚本创建"
                    )
                if user is None:
                    await uow.access.add_user(
                        PlatformUserEntity(
                            user_id=actor_id,
                            display_name=actor_id,
                            status="ACTIVE",
                        )
                    )
                if is_reserved_global_admin(actor_id):
                    roles = await uow.access.list_all_roles()
                    for role in roles:
                        if role.role_code != "system_admin":
                            continue
                        await uow.access.upsert_member_role(
                            app_id=role.app_id,
                            domain_id=int(entity.domain_id),
                            user_id=actor_id,
                            role_code=role.role_code,
                            status="ACTIVE",
                            actor_id=actor_id,
                        )
                else:
                    for app_id in ("knowledge_retrieval", "aiops"):
                        await uow.access.upsert_member_role(
                            app_id=app_id,
                            domain_id=int(entity.domain_id),
                            user_id=actor_id,
                            role_code="manager",
                            status="ACTIVE",
                            actor_id=actor_id,
                        )
                await uow.commit()
            except IntegrityError as exc:
                raise DomainConflictError(
                    "当前应用已存在同名 Domain"
                ) from exc
            return {
                "domain_id": int(entity.domain_id),
                "name": entity.name,
                "status": entity.status,
                "description": entity.description,
                "row_version": int(entity.row_version),
            }
