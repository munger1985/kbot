"""Main API 应用服务。"""

from .domain_management import (
    DomainConflictError,
    DomainManagementService,
)
from .domain_validation import DomainValidationService
from .notification_center import NotificationCenterError, NotificationCenterService
from .notification_projection import (
    NotificationDispatcher,
    NotificationProjectionService,
)
from .access_control import (
    AccessConfigurationError,
    AccessControlService,
    AccessDeniedError,
    AccessSnapshot,
    GLOBAL_ADMIN_USER_ID,
    is_reserved_global_admin,
)
from .access_management import (
    AccessManagementError,
    AccessManagementService,
)
from .user_auth import (
    KM_PORTAL_DOMAIN_NAME,
    UserAuthenticationError,
    UserAuthService,
    UserTokenClaims,
    UserTokenCodec,
    create_user_token_codec,
)

__all__ = [
    "AccessControlService",
    "AccessConfigurationError",
    "AccessDeniedError",
    "AccessSnapshot",
    "AccessManagementError",
    "AccessManagementService",
    "GLOBAL_ADMIN_USER_ID",
    "is_reserved_global_admin",
    "DomainConflictError",
    "DomainManagementService",
    "DomainValidationService",
    "KM_PORTAL_DOMAIN_NAME",
    "NotificationCenterError",
    "NotificationCenterService",
    "NotificationDispatcher",
    "NotificationProjectionService",
    "UserAuthenticationError",
    "UserAuthService",
    "UserTokenClaims",
    "UserTokenCodec",
    "create_user_token_codec",
]
