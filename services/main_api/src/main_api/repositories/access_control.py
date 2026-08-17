"""平台身份、应用成员与分层授权 Repository。"""

from datetime import datetime, timezone

from sqlalchemy import delete, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from main_api.entities.access_control import (
    AppDomainEntity,
    AppMemberEntity,
    AppMemberRoleEntity,
    AppMemberRoleScopeEntity,
    AppRoleEntity,
    AppRolePermissionEntity,
    PermissionEntity,
    PlatformApplicationEntity,
    PlatformUserCredentialEntity,
    PlatformUserEntity,
    PlatformUserRoleEntity,
)


class AccessControlRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_application(self, app_id: str) -> PlatformApplicationEntity | None:
        return await self._session.get(PlatformApplicationEntity, app_id)

    async def list_applications(self) -> list[PlatformApplicationEntity]:
        rows = await self._session.scalars(
            select(PlatformApplicationEntity).order_by(PlatformApplicationEntity.app_id)
        )
        return list(rows)

    async def update_application_status(self, *, app: PlatformApplicationEntity, status: str) -> None:
        app.status = status
        app.row_version = int(app.row_version) + 1
        app.updated_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def get_user(self, user_id: str) -> PlatformUserEntity | None:
        return await self._session.get(PlatformUserEntity, user_id)

    async def add_user(self, row: PlatformUserEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get_user_credential(self, user_id: str) -> PlatformUserCredentialEntity | None:
        return await self._session.get(PlatformUserCredentialEntity, user_id)

    async def add_user_credential(self, row: PlatformUserCredentialEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def set_user_password(
        self, *, credential: PlatformUserCredentialEntity, password_hash: str,
        must_change_password: bool = False,
    ) -> None:
        credential.password_hash = password_hash
        credential.must_change_password = "Y" if must_change_password else "N"
        now = datetime.now(timezone.utc)
        credential.password_updated_at = now
        credential.updated_at = now
        await self._session.flush()

    @staticmethod
    def _domain_scope_clause(*, app_id: str, domain_id_column):
        return or_(
            AppMemberRoleEntity.scope_mode == "ALL_APP_DOMAINS",
            exists().where(
                AppMemberRoleScopeEntity.app_id == AppMemberRoleEntity.app_id,
                AppMemberRoleScopeEntity.user_id == AppMemberRoleEntity.user_id,
                AppMemberRoleScopeEntity.role_code == AppMemberRoleEntity.role_code,
                AppMemberRoleScopeEntity.app_id == app_id,
                AppMemberRoleScopeEntity.domain_id == domain_id_column,
            ),
        )

    async def list_active_domain_ids(
        self, user_id: str, app_id: str | None = None
    ) -> tuple[int, ...]:
        statement = (
            select(AppDomainEntity.domain_id)
            .join(PlatformApplicationEntity, PlatformApplicationEntity.app_id == AppDomainEntity.app_id)
            .join(AppMemberEntity, AppMemberEntity.app_id == AppDomainEntity.app_id)
            .join(
                AppMemberRoleEntity,
                (AppMemberRoleEntity.app_id == AppMemberEntity.app_id)
                & (AppMemberRoleEntity.user_id == AppMemberEntity.user_id),
            )
            .join(
                AppRoleEntity,
                (AppRoleEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppRoleEntity.role_code == AppMemberRoleEntity.role_code),
            )
            .join(PlatformUserEntity, PlatformUserEntity.user_id == AppMemberEntity.user_id)
            .where(
                AppMemberEntity.user_id == user_id,
                AppMemberEntity.status == "ACTIVE",
                AppMemberRoleEntity.status == "ACTIVE",
                AppRoleEntity.status == "ACTIVE",
                AppDomainEntity.status == "ACTIVE",
                PlatformApplicationEntity.status == "ACTIVE",
                PlatformUserEntity.status == "ACTIVE",
                self._domain_scope_clause(app_id=AppDomainEntity.app_id, domain_id_column=AppDomainEntity.domain_id),
            )
        )
        if app_id:
            statement = statement.where(AppDomainEntity.app_id == app_id)
        rows = await self._session.scalars(statement.distinct().order_by(AppDomainEntity.domain_id))
        return tuple(int(value) for value in rows)

    async def list_active_km_domain_ids(self, user_id: str) -> tuple[int, ...]:
        return await self.list_active_domain_ids(user_id, app_id="km_asset")

    async def list_active_app_ids(self, user_id: str) -> tuple[str, ...]:
        rows = await self._session.scalars(
            select(AppDomainEntity.app_id)
            .join(
                AppMemberEntity,
                AppMemberEntity.app_id == AppDomainEntity.app_id,
            )
            .join(
                AppMemberRoleEntity,
                (AppMemberRoleEntity.app_id == AppMemberEntity.app_id)
                & (AppMemberRoleEntity.user_id == AppMemberEntity.user_id),
            )
            .join(
                AppRoleEntity,
                (AppRoleEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppRoleEntity.role_code == AppMemberRoleEntity.role_code),
            )
            .join(
                PlatformApplicationEntity,
                PlatformApplicationEntity.app_id == AppDomainEntity.app_id,
            )
            .join(
                PlatformUserEntity,
                PlatformUserEntity.user_id == AppMemberEntity.user_id,
            )
            .where(
                AppMemberEntity.user_id == user_id,
                AppMemberEntity.status == "ACTIVE",
                AppMemberRoleEntity.status == "ACTIVE",
                AppRoleEntity.status == "ACTIVE",
                AppDomainEntity.status == "ACTIVE",
                PlatformApplicationEntity.status == "ACTIVE",
                PlatformUserEntity.status == "ACTIVE",
                self._domain_scope_clause(
                    app_id=AppDomainEntity.app_id,
                    domain_id_column=AppDomainEntity.domain_id,
                ),
            )
            .distinct()
            .order_by(AppDomainEntity.app_id)
        )
        return tuple(rows)

    async def platform_permissions_for(self, *, user_id: str) -> set[str]:
        rows = await self._session.scalars(
            select(AppRolePermissionEntity.permission_code)
            .join(
                PlatformUserRoleEntity,
                (PlatformUserRoleEntity.app_id == AppRolePermissionEntity.app_id)
                & (PlatformUserRoleEntity.role_code == AppRolePermissionEntity.role_code),
            )
            .join(
                AppRoleEntity,
                (AppRoleEntity.app_id == AppRolePermissionEntity.app_id)
                & (AppRoleEntity.role_code == AppRolePermissionEntity.role_code),
            )
            .join(PlatformUserEntity, PlatformUserEntity.user_id == PlatformUserRoleEntity.user_id)
            .where(
                AppRolePermissionEntity.app_id == "platform",
                PlatformUserRoleEntity.app_id == "platform",
                PlatformUserRoleEntity.user_id == user_id,
                PlatformUserRoleEntity.status == "ACTIVE",
                AppRoleEntity.status == "ACTIVE",
                PlatformUserEntity.status == "ACTIVE",
                PlatformUserEntity.account_origin == "PLATFORM",
            )
        )
        return set(rows)

    async def permissions_for(self, *, app_id: str, domain_id: int, user_id: str) -> set[str]:
        rows = await self._session.scalars(
            select(AppRolePermissionEntity.permission_code)
            .join(
                AppMemberRoleEntity,
                (AppMemberRoleEntity.app_id == AppRolePermissionEntity.app_id)
                & (AppMemberRoleEntity.role_code == AppRolePermissionEntity.role_code),
            )
            .join(
                AppMemberEntity,
                (AppMemberEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppMemberEntity.user_id == AppMemberRoleEntity.user_id),
            )
            .join(PlatformApplicationEntity, PlatformApplicationEntity.app_id == AppMemberEntity.app_id)
            .join(AppDomainEntity, AppDomainEntity.app_id == AppMemberEntity.app_id)
            .join(
                AppRoleEntity,
                (AppRoleEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppRoleEntity.role_code == AppMemberRoleEntity.role_code),
            )
            .join(PlatformUserEntity, PlatformUserEntity.user_id == AppMemberEntity.user_id)
            .where(
                AppMemberEntity.app_id == app_id,
                AppMemberEntity.user_id == user_id,
                AppDomainEntity.domain_id == domain_id,
                AppMemberEntity.status == "ACTIVE",
                AppMemberRoleEntity.status == "ACTIVE",
                AppRoleEntity.status == "ACTIVE",
                AppDomainEntity.status == "ACTIVE",
                PlatformApplicationEntity.status == "ACTIVE",
                PlatformUserEntity.status == "ACTIVE",
                self._domain_scope_clause(app_id=app_id, domain_id_column=domain_id),
            )
        )
        return set(rows)

    async def list_roles(self, *, app_id: str, domain_id: int, user_id: str) -> tuple[str, ...]:
        rows = await self._session.scalars(
            select(AppMemberRoleEntity.role_code)
            .join(
                AppMemberEntity,
                (AppMemberEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppMemberEntity.user_id == AppMemberRoleEntity.user_id),
            )
            .join(PlatformApplicationEntity, PlatformApplicationEntity.app_id == AppMemberEntity.app_id)
            .join(AppDomainEntity, AppDomainEntity.app_id == AppMemberEntity.app_id)
            .join(
                AppRoleEntity,
                (AppRoleEntity.app_id == AppMemberRoleEntity.app_id)
                & (AppRoleEntity.role_code == AppMemberRoleEntity.role_code),
            )
            .join(PlatformUserEntity, PlatformUserEntity.user_id == AppMemberEntity.user_id)
            .where(
                AppMemberRoleEntity.app_id == app_id,
                AppMemberRoleEntity.user_id == user_id,
                AppDomainEntity.domain_id == domain_id,
                AppMemberRoleEntity.status == "ACTIVE",
                AppMemberEntity.status == "ACTIVE",
                AppRoleEntity.status == "ACTIVE",
                AppDomainEntity.status == "ACTIVE",
                PlatformApplicationEntity.status == "ACTIVE",
                PlatformUserEntity.status == "ACTIVE",
                self._domain_scope_clause(app_id=app_id, domain_id_column=domain_id),
            )
            .distinct()
            .order_by(AppMemberRoleEntity.role_code)
        )
        return tuple(rows)

    async def list_active_app_roles(self, *, app_id: str) -> list[AppRoleEntity]:
        rows = await self._session.scalars(
            select(AppRoleEntity).where(AppRoleEntity.app_id == app_id, AppRoleEntity.status == "ACTIVE").order_by(AppRoleEntity.display_name, AppRoleEntity.role_code)
        )
        return list(rows)

    async def get_role(self, *, app_id: str, role_code: str) -> AppRoleEntity | None:
        return await self._session.get(AppRoleEntity, (app_id, role_code))

    async def get_app_member(self, *, app_id: str, user_id: str) -> AppMemberEntity | None:
        return await self._session.get(AppMemberEntity, (app_id, user_id))

    async def add_app_member(self, row: AppMemberEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def update_app_member_status(self, *, member: AppMemberEntity, status: str) -> None:
        member.status = status
        member.updated_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def get_app_domain(self, *, app_id: str, domain_id: int) -> AppDomainEntity | None:
        return await self._session.get(AppDomainEntity, (app_id, domain_id))

    async def add_app_domain(self, row: AppDomainEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list_app_domains(self, *, app_id: str) -> list[AppDomainEntity]:
        rows = await self._session.scalars(
            select(AppDomainEntity).where(AppDomainEntity.app_id == app_id).order_by(AppDomainEntity.domain_id)
        )
        return list(rows)

    async def upsert_platform_user_role(
        self, *, user_id: str, role_code: str, status: str, actor_id: str
    ) -> PlatformUserRoleEntity:
        row = await self._session.get(PlatformUserRoleEntity, (user_id, role_code))
        if row is None:
            row = PlatformUserRoleEntity(
                user_id=user_id, role_code=role_code, app_id="platform",
                status=status, created_by=actor_id
            )
            self._session.add(row)
        else:
            row.status = status
        await self._session.flush()
        return row

    async def list_platform_user_roles(self, *, user_id: str) -> list[PlatformUserRoleEntity]:
        rows = await self._session.scalars(
            select(PlatformUserRoleEntity).where(PlatformUserRoleEntity.user_id == user_id).order_by(PlatformUserRoleEntity.role_code)
        )
        return list(rows)

    async def delete_platform_user_roles(self, *, user_id: str) -> None:
        await self._session.execute(
            delete(PlatformUserRoleEntity).where(PlatformUserRoleEntity.user_id == user_id)
        )
        await self._session.flush()

    async def upsert_member_role(
        self, *, app_id: str, user_id: str, role_code: str,
        scope_mode: str, status: str, actor_id: str,
    ) -> AppMemberRoleEntity:
        key = {"app_id": app_id, "user_id": user_id, "role_code": role_code}
        row = await self._session.get(AppMemberRoleEntity, key)
        if row is None:
            row = AppMemberRoleEntity(**key, scope_mode=scope_mode, status=status, created_by=actor_id)
            self._session.add(row)
        else:
            row.scope_mode = scope_mode
            row.status = status
        await self._session.flush()
        return row

    async def replace_member_role_scopes(
        self, *, app_id: str, user_id: str, role_code: str, domain_ids: tuple[int, ...]
    ) -> None:
        await self._session.execute(
            delete(AppMemberRoleScopeEntity).where(
                AppMemberRoleScopeEntity.app_id == app_id,
                AppMemberRoleScopeEntity.user_id == user_id,
                AppMemberRoleScopeEntity.role_code == role_code,
            )
        )
        self._session.add_all(
            AppMemberRoleScopeEntity(app_id=app_id, user_id=user_id, role_code=role_code, domain_id=domain_id)
            for domain_id in domain_ids
        )
        await self._session.flush()

    async def delete_member_authorizations(self, *, app_id: str, user_id: str) -> None:
        await self._session.execute(
            delete(AppMemberRoleScopeEntity).where(
                AppMemberRoleScopeEntity.app_id == app_id,
                AppMemberRoleScopeEntity.user_id == user_id,
            )
        )
        await self._session.execute(
            delete(AppMemberRoleEntity).where(
                AppMemberRoleEntity.app_id == app_id,
                AppMemberRoleEntity.user_id == user_id,
            )
        )
        await self._session.flush()

    async def delete_app_member(self, *, member: AppMemberEntity) -> None:
        await self.delete_member_authorizations(app_id=member.app_id, user_id=member.user_id)
        await self._session.delete(member)
        await self._session.flush()

    async def list_member_role_scopes(self, *, app_id: str, user_id: str, role_code: str) -> tuple[int, ...]:
        rows = await self._session.scalars(
            select(AppMemberRoleScopeEntity.domain_id).where(
                AppMemberRoleScopeEntity.app_id == app_id,
                AppMemberRoleScopeEntity.user_id == user_id,
                AppMemberRoleScopeEntity.role_code == role_code,
            ).order_by(AppMemberRoleScopeEntity.domain_id)
        )
        return tuple(int(value) for value in rows)

    async def list_member_roles(self, *, app_id: str, domain_id: int | None = None) -> list[AppMemberRoleEntity]:
        statement = select(AppMemberRoleEntity).where(AppMemberRoleEntity.app_id == app_id)
        if domain_id is not None:
            statement = statement.where(self._domain_scope_clause(app_id=app_id, domain_id_column=domain_id))
        rows = await self._session.scalars(statement.order_by(AppMemberRoleEntity.user_id, AppMemberRoleEntity.role_code))
        return list(rows)

    async def list_app_members(self, *, app_id: str) -> list[AppMemberEntity]:
        rows = await self._session.scalars(
            select(AppMemberEntity).where(AppMemberEntity.app_id == app_id).order_by(AppMemberEntity.user_id)
        )
        return list(rows)

    async def list_users_by_ids(self, user_ids: tuple[str, ...]) -> list[PlatformUserEntity]:
        if not user_ids:
            return []
        rows = await self._session.scalars(select(PlatformUserEntity).where(PlatformUserEntity.user_id.in_(user_ids)))
        return list(rows)

    async def list_users(
        self, *, offset: int, limit: int, search: str | None, status: str | None,
        account_origin: str | None = None, owner_app_id: str | None = None,
    ) -> tuple[list[PlatformUserEntity], int]:
        filters = []
        if search:
            pattern = f"%{search.casefold()}%"
            filters.append(or_(func.lower(PlatformUserEntity.user_id).like(pattern), func.lower(PlatformUserEntity.display_name).like(pattern)))
        if status:
            filters.append(PlatformUserEntity.status == status)
        if account_origin:
            filters.append(PlatformUserEntity.account_origin == account_origin)
        if owner_app_id:
            filters.append(PlatformUserEntity.owner_app_id == owner_app_id)
        total = int(await self._session.scalar(select(func.count()).select_from(PlatformUserEntity).where(*filters)) or 0)
        rows = await self._session.scalars(select(PlatformUserEntity).where(*filters).order_by(PlatformUserEntity.user_id).offset(offset).limit(limit))
        return list(rows), total

    async def update_user(
        self, *, user: PlatformUserEntity, display_name: str | None,
        status: str, max_security_level: int,
    ) -> None:
        user.display_name = display_name
        user.status = status
        user.max_security_level = max_security_level
        user.updated_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def list_user_memberships(self, *, user_id: str) -> list[AppMemberEntity]:
        rows = await self._session.scalars(
            select(AppMemberEntity).where(AppMemberEntity.user_id == user_id).order_by(AppMemberEntity.app_id)
        )
        return list(rows)

    async def delete_user(self, *, user: PlatformUserEntity) -> None:
        await self._session.execute(delete(AppMemberRoleScopeEntity).where(AppMemberRoleScopeEntity.user_id == user.user_id))
        await self._session.execute(delete(AppMemberRoleEntity).where(AppMemberRoleEntity.user_id == user.user_id))
        await self._session.execute(delete(AppMemberEntity).where(AppMemberEntity.user_id == user.user_id))
        await self._session.execute(delete(PlatformUserRoleEntity).where(PlatformUserRoleEntity.user_id == user.user_id))
        await self._session.execute(delete(PlatformUserCredentialEntity).where(PlatformUserCredentialEntity.user_id == user.user_id))
        await self._session.delete(user)
        await self._session.flush()

    async def list_permissions(self, *, app_id: str | None = None) -> list[PermissionEntity]:
        statement = select(PermissionEntity)
        if app_id:
            statement = statement.where(PermissionEntity.app_id == app_id)
        rows = await self._session.scalars(statement.order_by(PermissionEntity.app_id, PermissionEntity.permission_code))
        return list(rows)

    async def list_all_roles(self, *, app_id: str | None = None) -> list[AppRoleEntity]:
        statement = select(AppRoleEntity)
        if app_id:
            statement = statement.where(AppRoleEntity.app_id == app_id)
        rows = await self._session.scalars(statement.order_by(AppRoleEntity.app_id, AppRoleEntity.role_code))
        return list(rows)

    async def list_role_permission_codes(self, *, app_id: str, role_code: str) -> tuple[str, ...]:
        rows = await self._session.scalars(select(AppRolePermissionEntity.permission_code).where(AppRolePermissionEntity.app_id == app_id, AppRolePermissionEntity.role_code == role_code).order_by(AppRolePermissionEntity.permission_code))
        return tuple(rows)

    async def add_role(self, row: AppRoleEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def update_role(self, *, role: AppRoleEntity, display_name: str, status: str) -> None:
        role.display_name = display_name
        role.status = status
        role.row_version = int(role.row_version) + 1
        await self._session.flush()

    async def replace_role_permissions(self, *, app_id: str, role_code: str, permission_codes: tuple[str, ...]) -> None:
        await self._session.execute(delete(AppRolePermissionEntity).where(AppRolePermissionEntity.app_id == app_id, AppRolePermissionEntity.role_code == role_code))
        self._session.add_all(AppRolePermissionEntity(app_id=app_id, role_code=role_code, permission_code=permission_code) for permission_code in permission_codes)
        await self._session.flush()


__all__ = ["AccessControlRepository"]
