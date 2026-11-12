"""Exadata 专用指标只读上下文，不包含数据库 Target 或数据库凭据。"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExadataMetricRead:
    source_locator_key: str
    max_response_bytes: int = 2 * 1024 * 1024
    window_start: datetime | None = None
    window_end: datetime | None = None
    pagination: None = None
