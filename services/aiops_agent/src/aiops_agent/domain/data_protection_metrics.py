"""根据演练验证事实计算 RTO/RPO，与报告生成耗时无关。"""

from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from typing import Literal


@dataclass(frozen=True)
class ProtectionMetric:
    target_seconds: int | None
    measured_min_seconds: int | None
    measured_max_seconds: int | None
    gap_min_seconds: int | None
    gap_max_seconds: int | None
    verdict: Literal["PASS", "OVER_TARGET", "INCONCLUSIVE", "NOT_MEASURED", "NOT_COMPARABLE", "NO_TARGET"]


def _metric(target: int | None, lower: int | None = None, upper: int | None = None, *, comparable: bool = True):
    if not comparable:
        return ProtectionMetric(target, lower, upper, None, None, "NOT_COMPARABLE")
    if lower is None or upper is None:
        return ProtectionMetric(target, None, None, None, None, "NOT_MEASURED")
    if target is None:
        return ProtectionMetric(target, lower, upper, None, None, "NO_TARGET")
    verdict = "PASS" if upper <= target else "OVER_TARGET" if lower > target else "INCONCLUSIVE"
    return ProtectionMetric(target, lower, upper, lower - target, upper - target, verdict)


def _seconds(later: datetime | None, earlier: datetime | None) -> int | None:
    if later is None or earlier is None or later.utcoffset() is None or earlier.utcoffset() is None:
        return None
    elapsed = (later.astimezone(UTC) - earlier.astimezone(UTC)).total_seconds()
    return ceil(elapsed) if elapsed >= 0 else None


def evaluate_protection_metrics(
    *, simulated_failure_at: datetime, verified_at: datetime | None,
    recovery_coordinate: str | None, recovered_at_earliest: datetime | None,
    recovered_at_latest: datetime | None, target_rto: int | None, target_rpo: int | None,
    database_verified: bool, latest_recovery_scenario: bool,
) -> tuple[ProtectionMetric, ProtectionMetric]:
    """两个恢复时刻相等表示精确时间；未验证的时间或坐标不能当成实测。"""
    rto = _seconds(verified_at, simulated_failure_at) if database_verified else None
    lower = _seconds(simulated_failure_at, recovered_at_latest)
    upper = _seconds(simulated_failure_at, recovered_at_earliest)
    if rto is None or not recovery_coordinate or lower is None or upper is None or lower > upper:
        lower = upper = None
    return (
        _metric(target_rto, rto, rto),
        _metric(target_rpo, lower, upper, comparable=latest_recovery_scenario),
    )
