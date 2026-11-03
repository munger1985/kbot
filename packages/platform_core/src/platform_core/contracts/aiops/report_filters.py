"""报告中心的显式业务分类；不根据标题、模板名或来源文字猜测。"""

from __future__ import annotations

from enum import StrEnum

from platform_core.contracts.aiops.types import ReportType


class ReportCategory(StrEnum):
    INSPECTION = "INSPECTION"
    DIAGNOSIS = "DIAGNOSIS"
    SPECIAL_ANALYSIS = "SPECIAL_ANALYSIS"


REPORT_CATEGORY_TYPES: dict[ReportCategory, tuple[ReportType, ...]] = {
    ReportCategory.INSPECTION: (
        ReportType.INSPECTION_DAILY,
        ReportType.INSPECTION_WEEKLY,
        ReportType.INSPECTION_MONTHLY,
        ReportType.INSPECTION_QUARTERLY,
        ReportType.INSPECTION_ANNUAL,
        ReportType.INSPECTION_CUSTOM,
    ),
    ReportCategory.DIAGNOSIS: (
        ReportType.INCIDENT,
        ReportType.PERFORMANCE,
    ),
    ReportCategory.SPECIAL_ANALYSIS: (
        ReportType.COMPARISON,
        ReportType.ORACLE_WORKLOAD,
        ReportType.ORACLE_WORKLOAD_DIFF,
        ReportType.MYSQL_WORKLOAD,
        ReportType.MYSQL_WORKLOAD_DIFF,
        ReportType.MYSQL_ACTIVITY,
        ReportType.MYSQL_SQL_HEALTHCHECK,
        ReportType.POSTGRESQL_WORKLOAD,
        ReportType.POSTGRESQL_WORKLOAD_DIFF,
        ReportType.POSTGRESQL_ACTIVITY,
        ReportType.POSTGRESQL_SQL_HEALTHCHECK,
        ReportType.POSTGRESQL_PGBADGER,
    ),
}


def category_report_types(category: ReportCategory | str | None) -> tuple[str, ...] | None:
    """空分类表示全部；新增报告类型必须显式归类并通过穷尽性测试。"""
    if category is None:
        return None
    return tuple(item.value for item in REPORT_CATEGORY_TYPES[ReportCategory(category)])
