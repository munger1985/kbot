"""Main API 拥有的持久化实体。"""

from .domain import PlatformDomainEntity
from .access_control import (
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
from .notification import (
    BackgroundOperationEntity,
    NotificationInboxEntity,
    NotificationPreferenceEntity,
    OperationWatchEntity,
    WorkItemEntity,
)

__all__ = [
    "AppDomainEntity",
    "AppMemberEntity",
    "AppMemberRoleEntity",
    "AppMemberRoleScopeEntity",
    "AppRoleEntity",
    "AppRolePermissionEntity",
    "BackgroundOperationEntity",
    "NotificationInboxEntity",
    "NotificationPreferenceEntity",
    "OperationWatchEntity",
    "PlatformApplicationEntity",
    "PlatformDomainEntity",
    "PlatformUserEntity",
    "PlatformUserCredentialEntity",
    "PlatformUserRoleEntity",
    "PermissionEntity",
    "WorkItemEntity",
]
