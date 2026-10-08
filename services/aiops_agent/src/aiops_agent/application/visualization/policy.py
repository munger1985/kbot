"""依据版本化场景档案选择监控图表，不从图形推导诊断结论。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from aiops_agent.contracts.evidence import MetricObservation, ObservationSet
from platform_core.contracts.visualization import ChartSpec
from platform_core.visualization import ChartSkill


_CATALOG_PATH = Path(__file__).with_name("chart_profiles.json")
_LOW_COVERAGE_RATIO = 0.8


@lru_cache(maxsize=1)
def _profiles() -> dict[str, dict[str, Any]]:
    payload = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "AIOPS_CHART_PROFILE_CATALOG.v1":
        raise ValueError("图表档案目录版本无效")
    profiles: dict[str, dict[str, Any]] = {}
    for raw_profile in payload.get("profiles") or ():
        profile = dict(raw_profile)
        profile_id = str(profile.get("profile_id") or "")
        if not profile_id or profile_id in profiles:
            raise ValueError("图表档案ID缺失或重复")
        slots = tuple(dict(item) for item in profile.get("slots") or ())
        if not slots:
            raise ValueError(f"图表档案缺少指标槽位：{profile_id}")
        profile["slots"] = slots
        profiles[profile_id] = profile
    return profiles


def chart_profile_window_seconds(profile_id: str | None) -> int | None:
    if not profile_id:
        return None
    profile = _profiles().get(profile_id)
    if profile is None:
        raise ValueError(f"未知图表档案：{profile_id}")
    return int(profile["default_window_seconds"])


def chart_profile_metric_codes(profile_id: str | None) -> tuple[str, ...]:
    if not profile_id:
        return ()
    profile = _profiles().get(profile_id)
    if profile is None:
        raise ValueError(f"未知图表档案：{profile_id}")
    return tuple(
        dict.fromkeys(
            str(metric_code)
            for slot in profile["slots"]
            for metric_code in slot.get("candidate_metric_codes") or ()
        )
    )


class AIOpsChartSelectionPolicy:
    """按场景、指标槽位和质量规则确定性生成有序图表。"""

    @classmethod
    def compile(
        cls,
        *,
        profile_id: str | None,
        artifact_id: str,
        result: ObservationSet,
    ) -> tuple[ChartSpec, ...]:
        if not profile_id:
            return ()
        profile = _profiles().get(profile_id)
        if profile is None:
            raise ValueError(f"未知图表档案：{profile_id}")
        observations = {
            item.metric_code: item for item in result.observations
        }
        charts: list[ChartSpec] = []
        for slot in profile["slots"]:
            observation = cls._first_usable_observation(
                observations,
                tuple(slot.get("candidate_metric_codes") or ()),
                suppress_when_flat=bool(slot.get("suppress_when_flat")),
            )
            if observation is None:
                continue
            chart = cls._compile_observation(
                profile_id=profile_id,
                slot=slot,
                artifact_id=artifact_id,
                observation=observation,
            )
            if chart is not None:
                charts.append(chart)
            if len(charts) >= int(profile["max_charts"]):
                break
        return tuple(charts)

    @classmethod
    def _first_usable_observation(
        cls,
        observations: dict[str, MetricObservation],
        candidates: tuple[str, ...],
        *,
        suppress_when_flat: bool,
    ) -> MetricObservation | None:
        for metric_code in candidates:
            observation = observations.get(metric_code)
            if observation is None or not cls._has_chartable_series(observation):
                continue
            if suppress_when_flat and cls._is_flat(observation):
                continue
            return observation
        return None

    @staticmethod
    def _has_chartable_series(observation: MetricObservation) -> bool:
        return any(
            len(
                [
                    point
                    for point in series.points
                    if point.quality == "GOOD"
                    and isinstance(point.value, (int, float))
                    and not isinstance(point.value, bool)
                ]
            )
            >= 2
            for series in observation.series
        )

    @staticmethod
    def _is_flat(observation: MetricObservation) -> bool:
        values = [
            float(point.value)
            for series in observation.series
            for point in series.points
            if point.quality == "GOOD"
            and isinstance(point.value, (int, float))
            and not isinstance(point.value, bool)
        ]
        return bool(values) and max(values) == min(values)

    @staticmethod
    def _compile_observation(
        *,
        profile_id: str,
        slot: dict[str, Any],
        artifact_id: str,
        observation: MetricObservation,
    ) -> ChartSpec | None:
        series_items = []
        for index, series in enumerate(observation.series):
            points = [
                {"x": point.observed_at.isoformat(), "y": point.value}
                for point in series.points
                if point.quality == "GOOD"
                and isinstance(point.value, (int, float))
                and not isinstance(point.value, bool)
            ]
            if len(points) < 2:
                continue
            dimensions = {
                key: value
                for key, value in series.dimensions.items()
                if key not in {"target_key", "instance"}
            }
            name = str(
                dimensions.get("tablespace")
                or dimensions.get("tablespace_name")
                or ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(dimensions.items())
                )
                or f"序列 {index + 1}"
            )
            series_items.append(
                {
                    "series_key": ",".join(
                        f"{key}={value}"
                        for key, value in sorted(series.dimensions.items())
                    )
                    or f"series-{index + 1}",
                    "name": name,
                    "points": points,
                }
            )
        if not series_items:
            return None
        low_coverage = observation.coverage_ratio < _LOW_COVERAGE_RATIO
        return ChartSkill.time_series(
            title=str(slot["title"]),
            unit=observation.unit,
            series=series_items,
            source_ids=(f"artifact:{artifact_id}#prometheus",),
            metadata={
                "visualization_profile_id": profile_id,
                "slot_id": str(slot["slot_id"]),
                "metric_code": observation.metric_code,
                "window_start": observation.window_start.isoformat(),
                "window_end": observation.window_end.isoformat(),
                "coverage_ratio": round(observation.coverage_ratio, 4),
                "coverage_status": "LOW" if low_coverage else "SUFFICIENT",
                "coverage_warning": (
                    "监控采样覆盖率低于80%，图表仅供观察"
                    if low_coverage
                    else None
                ),
                "sample_count": observation.actual_points,
            },
        )
