"""巡检与对比报告的不可变 Artifact 契约。"""

from .comparison import ComparisonPlan, ComparisonResult
from .models import ReportContent

__all__ = ["ComparisonPlan", "ComparisonResult", "ReportContent"]
