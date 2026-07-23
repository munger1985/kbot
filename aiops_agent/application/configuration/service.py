"""AIOps 配置资源 Application Service 门面。"""

from .base import ConfigurationServiceBase
from .common import ConfigurationScope
from .inspection_service import InspectionConfigurationMixin
from .monitor_service import MonitorConfigurationMixin
from .policy_service import PolicyConfigurationMixin
from .target_service import TargetConfigurationMixin


class AIOpsConfigurationService(
    TargetConfigurationMixin,
    MonitorConfigurationMixin,
    PolicyConfigurationMixin,
    InspectionConfigurationMixin,
    ConfigurationServiceBase,
):
    """组合六类配置能力，供 API 只注入一个稳定门面。"""


__all__ = ["AIOpsConfigurationService", "ConfigurationScope"]
