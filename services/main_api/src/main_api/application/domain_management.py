"""平台 Domain 生命周期应用服务。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.exc import IntegrityError

from main_api.entities import PlatformDomainEntity


class DomainConflictError(RuntimeError):
    """Domain 名称与当前应用内已有记录冲突。"""


class DomainManagementService:
    """管理不隶属于具体 Domain 的平台级 Domain 注册信息。"""

    def __init__(self, *, app_id: int, uow_factory: Callable[[], Any]):
        self._app_id = app_id
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
                app_id=self._app_id,
                name=normalized_name,
            )
            if existing is not None:
                raise DomainConflictError("当前应用已存在同名 Domain")
            entity = PlatformDomainEntity(
                app_id=self._app_id,
                name=normalized_name,
                status="ACTIVE",
                description=description,
                row_version=1,
                created_by=actor_id,
                updated_by=actor_id,
            )
            try:
                await uow.domains.add(entity)
                await uow.commit()
            except IntegrityError as exc:
                raise DomainConflictError(
                    "当前应用已存在同名 Domain"
                ) from exc
            return {
                "domain_id": int(entity.domain_id),
                "app_id": int(entity.app_id),
                "name": entity.name,
                "status": entity.status,
                "description": entity.description,
                "row_version": int(entity.row_version),
            }
