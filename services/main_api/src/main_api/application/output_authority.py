"""Main API 基于当前成员与角色核验后台输出，不伪造门户身份。"""

from datetime import datetime, timezone

from platform_core.authorization import BUILT_IN_APP_ROLE_CODES, built_in_role_permissions
from platform_core.contracts.aiops.output_authority import (
    OutputAuthorityRequest, OutputAuthoritySnapshot, output_required_permissions,
)


class OutputAuthorityDenied(PermissionError):
    """后台发起人的当前账号、空间或权限不再允许完成该输出。"""


class OutputAuthorityService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def resolve(self, request: OutputAuthorityRequest) -> OutputAuthoritySnapshot:
        try:
            domain_id = int(request.domain_id)
        except ValueError as exc:
            raise OutputAuthorityDenied() from exc
        if domain_id <= 0 or str(domain_id) != request.domain_id or request.tenant_id is not None:
            raise OutputAuthorityDenied()
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(request.actor_id)
            if (user is None or user.status != "ACTIVE"
                or not await uow.domains.exists_active(domain_id=domain_id)):
                raise OutputAuthorityDenied()
            security_level = int(user.max_security_level)
            if not 0 <= security_level <= 3:
                raise OutputAuthorityDenied()
            permissions = set(await uow.access.permissions_for(
                app_id="aiops", domain_id=domain_id, user_id=request.actor_id,
            ))
            roles = await uow.access.list_roles(
                app_id="aiops", domain_id=domain_id, user_id=request.actor_id,
            )
            for role in roles:
                if role in BUILT_IN_APP_ROLE_CODES:
                    permissions.update(built_in_role_permissions("aiops", role))
            if not output_required_permissions(request) <= permissions:
                raise OutputAuthorityDenied()
            return OutputAuthoritySnapshot(request=request, permissions=tuple(sorted(permissions)),
                max_security_level=security_level, checked_at=datetime.now(timezone.utc))
