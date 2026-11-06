"""Main API 管理的外部渠道表；跨服务只共享事件合同。"""

from sqlalchemy import Numeric
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type
from platform_core.persistence.notification_delivery import define_external_tables

external_tables = define_external_tables(
    metadata=BaseEntity.metadata, prefix="KBOT_NOTIFY_", id_type=UUIDv7Type(),
    domain_type=Numeric(38, 0), json_type=OracleNativeJSON(),
    timestamp_type=UniversalTimestamp(timezone=True),
)
