"""Main API 拥有的持久化实体。"""

from .domain import PlatformDomainEntity
from .slack import SlackDeliveryEntity, SlackInboxEntity, SlackThreadEntity

__all__ = [
    "PlatformDomainEntity",
    "SlackDeliveryEntity",
    "SlackInboxEntity",
    "SlackThreadEntity",
]
