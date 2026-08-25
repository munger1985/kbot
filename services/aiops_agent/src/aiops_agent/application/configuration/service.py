"""AIOps 配置资源 Application Service 门面。"""

from .base import ConfigurationServiceBase
from .common import ConfigurationScope
from .inspection_service import InspectionConfigurationMixin
from .notification_service import NotificationConfigurationMixin
from .diagnostic_source_service import DiagnosticSourceConfigurationMixin
from .policy_service import PolicyConfigurationMixin
from .target_service import TargetConfigurationMixin


class AIOpsConfigurationService(
    TargetConfigurationMixin,
    DiagnosticSourceConfigurationMixin,
    PolicyConfigurationMixin,
    InspectionConfigurationMixin,
    NotificationConfigurationMixin,
    ConfigurationServiceBase,
):
    """组合配置能力，供 API 只注入一个稳定门面。"""


__all__ = ["AIOpsConfigurationService", "ConfigurationScope"]
