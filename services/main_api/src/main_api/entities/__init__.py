"""Main API 拥有的持久化实体。"""

from .domain import PlatformDomainEntity
from .access_control import (
    AppMemberRoleEntity,
    AppRoleEntity,
    AppRolePermissionEntity,
    PermissionEntity,
    PlatformUserCredentialEntity,
    PlatformUserEntity,
)
from .notification import (
    BackgroundOperationEntity,
    NotificationInboxEntity,
    NotificationPreferenceEntity,
    OperationWatchEntity,
    WorkItemEntity,
)

__all__ = [
    "AppMemberRoleEntity",
    "AppRoleEntity",
    "AppRolePermissionEntity",
    "BackgroundOperationEntity",
    "NotificationInboxEntity",
    "NotificationPreferenceEntity",
    "OperationWatchEntity",
    "PlatformDomainEntity",
    "PlatformUserEntity",
    "PlatformUserCredentialEntity",
    "PermissionEntity",
    "WorkItemEntity",
]
