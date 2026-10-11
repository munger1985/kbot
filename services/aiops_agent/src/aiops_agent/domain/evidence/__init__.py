"""AIOps 证据与事件关联领域模型。"""

from .correlation import (
    SituationCorrelationDecision,
    correlate_signal_event,
    validate_event_class_map,
)

from .events import (
    SignalEventStatus,
    SignalSeverity,
)
from .metrics import (
    DEFAULT_BASELINE_METRICS,
    METRIC_ROLE_AVAILABILITY,
    METRIC_ROLE_STORAGE_MAX,
    METRIC_ROLE_STORAGE_USED,
    METRIC_ROLE_STORAGE_UTILIZATION,
)
from .observations import (
    summarize_points,
)
from .trends import extract_metric_trend_rows, summarize_numeric_trend

__all__ = [
    "DEFAULT_BASELINE_METRICS",
    "METRIC_ROLE_AVAILABILITY",
    "METRIC_ROLE_STORAGE_MAX",
    "METRIC_ROLE_STORAGE_USED",
    "METRIC_ROLE_STORAGE_UTILIZATION",
    "SignalEventStatus",
    "SignalSeverity",
    "SituationCorrelationDecision",
    "correlate_signal_event",
    "summarize_points",
    "extract_metric_trend_rows",
    "summarize_numeric_trend",
    "validate_event_class_map",
]
