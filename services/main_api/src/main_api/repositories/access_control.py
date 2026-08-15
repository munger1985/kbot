"""应用成员与权限 Repository。"""

from datetime import datetime, timezone

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from main_api.entities.access_control import (
    AppMemberRoleEntity,
    AppRoleEntity,
    AppRolePermissionEntity,
    PermissionEntity,
    PlatformUserCredentialEntity,
    PlatformUserEntity,
)


class AccessControlRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_user(self, user_id: str) -> PlatformUserEntity | None:
        return await self._session.get(PlatformUserEntity, user_id)

    async def add_user(self, row: PlatformUserEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get_user_credential(
        self, user_id: str
    ) -> PlatformUserCredentialEntity | None:
        return await self._session.get(PlatformUserCredentialEntity, user_id)

    async def set_user_password(
        self,
        *,
        credential: PlatformUserCredentialEntity,
        password_hash: str,
        must_change_password: bool = False,
    ) -> None:
        credential.password_hash = password_hash
        credential.must_change_password = "Y" if must_change_password else "N"
        now = datetime.now(timezone.utc)
        credential.password_updated_at = now
        credential.updated_at = now
        await self._session.flush()

    async def add_user_credential(
        self, row: PlatformUserCredentialEntity
    ) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list_active_domain_ids(self, user_id: str) -> tuple[int, ...]:
        rows = await self._session.scalars(
            select(AppMemberRoleEntity.domain_id)
            .join(
                AppRoleEntity,
                (AppRoleEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppRoleEntity.role_code == AppMemberRoleEntity.role_code),
            )
            .join(
                PlatformUserEntity,
                PlatformUserEntity.user_id == AppMemberRoleEntity.user_id,
            )
            .where(
                AppMemberRoleEntity.user_id == user_id,
                AppMemberRoleEntity.status == "ACTIVE",
                AppRoleEntity.status == "ACTIVE",
                PlatformUserEntity.status == "ACTIVE",
            )
            .distinct()
            .order_by(AppMemberRoleEntity.domain_id)
        )
        return tuple(int(value) for value in rows)

    async def list_active_km_domain_ids(self, user_id: str) -> tuple[int, ...]:
        rows = await self._session.scalars(
            select(AppMemberRoleEntity.domain_id)
            .join(
                AppRoleEntity,
                (AppRoleEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppRoleEntity.role_code == AppMemberRoleEntity.role_code),
            )
            .join(
                PlatformUserEntity,
                PlatformUserEntity.user_id == AppMemberRoleEntity.user_id,
            )
            .where(
                AppMemberRoleEntity.app_id == "km_asset",
                AppMemberRoleEntity.user_id == user_id,
                AppMemberRoleEntity.status == "ACTIVE",
                AppRoleEntity.status == "ACTIVE",
                PlatformUserEntity.status == "ACTIVE",
            )
            .distinct()
            .order_by(AppMemberRoleEntity.domain_id)
        )
        return tuple(int(value) for value in rows)

    async def permissions_for(
        self, *, app_id: str, domain_id: int, user_id: str
    ) -> set[str]:
        rows = await self._session.scalars(
            select(AppRolePermissionEntity.permission_code)
            .join(
                AppMemberRoleEntity,
                (AppMemberRoleEntity.app_id == AppRolePermissionEntity.app_id)
                & (AppMemberRoleEntity.role_code == AppRolePermissionEntity.role_code),
            )
            .join(
                PermissionEntity,
                PermissionEntity.permission_code
                == AppRolePermissionEntity.permission_code,
            )
            .join(
                AppRoleEntity,
                (AppRoleEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppRoleEntity.role_code == AppMemberRoleEntity.role_code),
            )
            .join(
                PlatformUserEntity,
                PlatformUserEntity.user_id == AppMemberRoleEntity.user_id,
            )
            .where(
                AppMemberRoleEntity.app_id == app_id,
                AppMemberRoleEntity.domain_id == domain_id,
                AppMemberRoleEntity.user_id == user_id,
                AppMemberRoleEntity.status == "ACTIVE",
                AppRoleEntity.status == "ACTIVE",
                PlatformUserEntity.status == "ACTIVE",
                PermissionEntity.app_id == app_id,
            )
        )
        return set(rows)

    async def list_roles(
        self, *, app_id: str, domain_id: int, user_id: str
    ) -> tuple[str, ...]:
        rows = await self._session.scalars(
            select(AppMemberRoleEntity.role_code)
            .join(
                AppRoleEntity,
                (AppRoleEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppRoleEntity.role_code == AppMemberRoleEntity.role_code),
            )
            .join(
                PlatformUserEntity,
                PlatformUserEntity.user_id == AppMemberRoleEntity.user_id,
            )
            .where(
                AppMemberRoleEntity.app_id == app_id,
                AppMemberRoleEntity.domain_id == domain_id,
                AppMemberRoleEntity.user_id == user_id,
                AppMemberRoleEntity.status == "ACTIVE",
                AppRoleEntity.status == "ACTIVE",
                PlatformUserEntity.status == "ACTIVE",
            )
            .order_by(AppMemberRoleEntity.role_code)
        )
        return tuple(rows)

    async def list_active_app_roles(self, *, app_id: str) -> list[AppRoleEntity]:
        rows = await self._session.scalars(
            select(AppRoleEntity).where(
                AppRoleEntity.app_id == app_id,
                AppRoleEntity.status == "ACTIVE",
            ).order_by(AppRoleEntity.display_name, AppRoleEntity.role_code)
        )
        return list(rows)

    async def upsert_member_role(
        self,
        *,
        app_id: str,
        domain_id: int,
        user_id: str,
        role_code: str,
        status: str,
        actor_id: str,
    ) -> AppMemberRoleEntity:
        key = {
            "app_id": app_id,
            "domain_id": domain_id,
            "user_id": user_id,
            "role_code": role_code,
        }
        row = await self._session.get(AppMemberRoleEntity, key)
        if row is None:
            row = AppMemberRoleEntity(
                **key, status=status, created_by=actor_id
            )
            self._session.add(row)
        else:
            row.status = status
        await self._session.flush()
        return row

    async def get_role(
        self, *, app_id: str, role_code: str
    ) -> AppRoleEntity | None:
        return await self._session.get(AppRoleEntity, (app_id, role_code))

    async def list_member_roles(
        self, *, app_id: str, domain_id: int
    ) -> list[AppMemberRoleEntity]:
        rows = await self._session.scalars(
            select(AppMemberRoleEntity).where(
                AppMemberRoleEntity.app_id == app_id,
                AppMemberRoleEntity.domain_id == domain_id,
            ).order_by(
                AppMemberRoleEntity.user_id,
                AppMemberRoleEntity.role_code,
            )
        )
        return list(rows)

    async def list_users_by_ids(
        self, user_ids: tuple[str, ...]
    ) -> list[PlatformUserEntity]:
        if not user_ids:
            return []
        rows = await self._session.scalars(
            select(PlatformUserEntity).where(
                PlatformUserEntity.user_id.in_(user_ids)
            )
        )
        return list(rows)

    async def list_users(
        self,
        *,
        offset: int,
        limit: int,
        search: str | None,
        status: str | None,
    ) -> tuple[list[PlatformUserEntity], int]:
        filters = []
        if search:
            pattern = f"%{search.casefold()}%"
            filters.append(
                or_(
                    func.lower(PlatformUserEntity.user_id).like(pattern),
                    func.lower(PlatformUserEntity.display_name).like(pattern),
                )
            )
        if status:
            filters.append(PlatformUserEntity.status == status)
        total = int(
            await self._session.scalar(
                select(func.count()).select_from(PlatformUserEntity).where(*filters)
            )
            or 0
        )
        rows = await self._session.scalars(
            select(PlatformUserEntity)
            .where(*filters)
            .order_by(PlatformUserEntity.user_id)
            .offset(offset)
            .limit(limit)
        )
        return list(rows), total

    async def update_user(
        self,
        *,
        user: PlatformUserEntity,
        display_name: str | None,
        status: str,
    ) -> None:
        user.display_name = display_name
        user.status = status
        user.updated_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def list_user_memberships(
        self, *, user_id: str
    ) -> list[AppMemberRoleEntity]:
        rows = await self._session.scalars(
            select(AppMemberRoleEntity)
            .where(AppMemberRoleEntity.user_id == user_id)
            .order_by(
                AppMemberRoleEntity.domain_id,
                AppMemberRoleEntity.app_id,
                AppMemberRoleEntity.role_code,
            )
        )
        return list(rows)

    async def delete_user(
        self, *, user: PlatformUserEntity
    ) -> None:
        """按外键依赖顺序物理删除普通用户及其认证、授权数据。"""
        await self._session.execute(
            delete(AppMemberRoleEntity).where(
                AppMemberRoleEntity.user_id == user.user_id
            )
        )
        await self._session.execute(
            delete(PlatformUserCredentialEntity).where(
                PlatformUserCredentialEntity.user_id == user.user_id
            )
        )
        await self._session.delete(user)
        await self._session.flush()

    async def list_permissions(
        self, *, app_id: str | None = None
    ) -> list[PermissionEntity]:
        statement = select(PermissionEntity)
        if app_id:
            statement = statement.where(PermissionEntity.app_id == app_id)
        rows = await self._session.scalars(
            statement.order_by(PermissionEntity.app_id, PermissionEntity.permission_code)
        )
        return list(rows)

    async def list_all_roles(
        self, *, app_id: str | None = None
    ) -> list[AppRoleEntity]:
        statement = select(AppRoleEntity)
        if app_id:
            statement = statement.where(AppRoleEntity.app_id == app_id)
        rows = await self._session.scalars(
            statement.order_by(AppRoleEntity.app_id, AppRoleEntity.role_code)
        )
        return list(rows)

    async def list_role_permission_codes(
        self, *, app_id: str, role_code: str
    ) -> tuple[str, ...]:
        rows = await self._session.scalars(
            select(AppRolePermissionEntity.permission_code)
            .where(
                AppRolePermissionEntity.app_id == app_id,
                AppRolePermissionEntity.role_code == role_code,
            )
            .order_by(AppRolePermissionEntity.permission_code)
        )
        return tuple(rows)

    async def add_role(self, row: AppRoleEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def update_role(
        self, *, role: AppRoleEntity, display_name: str, status: str
    ) -> None:
        role.display_name = display_name
        role.status = status
        await self._session.flush()

    async def replace_role_permissions(
        self,
        *,
        app_id: str,
        role_code: str,
        permission_codes: tuple[str, ...],
    ) -> None:
        await self._session.execute(
            delete(AppRolePermissionEntity).where(
                AppRolePermissionEntity.app_id == app_id,
                AppRolePermissionEntity.role_code == role_code,
            )
        )
        self._session.add_all(
            AppRolePermissionEntity(
                app_id=app_id,
                role_code=role_code,
                permission_code=permission_code,
            )
            for permission_code in permission_codes
        )
        await self._session.flush()


__all__ = ["AccessControlRepository"]
