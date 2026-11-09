"""AIOps 场景化图表选择策略；共享 ChartSkill 只负责画图。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from platform_core.contracts.visualization import ChartSpec


@dataclass(frozen=True)
class MetricCandidate:
    metric_code: str
    title: str


@dataclass(frozen=True)
class MetricSlot:
    slot_id: str
    label: str
    candidates: tuple[MetricCandidate, ...]
    required: bool
    finding_types: frozenset[str]
    check_ids: frozenset[str]
    display_condition: str


@dataclass(frozen=True)
class VisualizationProfile:
    profile_id: str
    default_window_seconds: int | None
    eager_monitoring: bool
    max_charts: int
    slots: tuple[MetricSlot, ...]

    def active_slots(
        self, selected_check_ids: Iterable[str] = ()
    ) -> tuple[MetricSlot, ...]:
        selected = frozenset(selected_check_ids)
        return tuple(
            slot
            for slot in self.slots
            if not slot.check_ids or bool(slot.check_ids & selected)
        )


class VisualizationProfileCatalog:
    def __init__(self, profiles: tuple[VisualizationProfile, ...]) -> None:
        self._profiles = {item.profile_id: item for item in profiles}

    def get(self, profile_id: str | None) -> VisualizationProfile | None:
        if not profile_id:
            return None
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise ValueError(f"未知 AIOps 图表 Profile：{profile_id}") from exc


@lru_cache(maxsize=1)
def load_visualization_profiles() -> VisualizationProfileCatalog:
    path = Path(__file__).with_name("common_profiles.v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "AIOPS_VISUALIZATION_PROFILES.v1":
        raise ValueError("AIOps 图表 Profile 版本无效")
    profiles = []
    for raw_profile in payload.get("profiles", ()):
        slots = []
        for raw_slot in raw_profile.get("slots", ()):
            slots.append(
                MetricSlot(
                    slot_id=str(raw_slot["slot_id"]),
                    label=str(raw_slot["label"]),
                    candidates=tuple(
                        MetricCandidate(
                            metric_code=str(item["metric_code"]),
                            title=str(item["title"]),
                        )
                        for item in raw_slot.get("candidates", ())
                    ),
                    required=bool(raw_slot.get("required", False)),
                    finding_types=frozenset(raw_slot.get("finding_types", ())),
                    check_ids=frozenset(raw_slot.get("check_ids", ())),
                    display_condition=str(
                        raw_slot.get("display_condition", "ALWAYS")
                    ),
                )
            )
        profiles.append(
            VisualizationProfile(
                profile_id=str(raw_profile["profile_id"]),
                default_window_seconds=(
                    int(raw_profile["default_window_seconds"])
                    if raw_profile.get("default_window_seconds") is not None
                    else None
                ),
                eager_monitoring=bool(raw_profile.get("eager_monitoring")),
                max_charts=int(raw_profile.get("max_charts", 4)),
                slots=tuple(slots),
            )
        )
    return VisualizationProfileCatalog(tuple(profiles))


class ChartSelectionPolicy:
    """按诊断 Profile、Finding 和展示条件选择监控趋势图。"""

    _minimum_coverage = 0.8

    def __init__(
        self, catalog: VisualizationProfileCatalog | None = None
    ) -> None:
        self._catalog = catalog or load_visualization_profiles()

    def select(
        self,
        *,
        profile_id: str | None,
        charts: Iterable[ChartSpec],
        finding_types: Iterable[str] = (),
        selected_check_ids: Iterable[str] = (),
    ) -> tuple[ChartSpec, ...]:
        profile = self._catalog.get(profile_id)
        if profile is None:
            return ()
        by_metric: dict[str, ChartSpec] = {}
        for chart in charts:
            metric_code = str(chart.metadata.get("metric_code") or "")
            if not metric_code:
                continue
            current = by_metric.get(metric_code)
            if current is None or float(
                chart.metadata.get("coverage_ratio") or 0
            ) > float(current.metadata.get("coverage_ratio") or 0):
                by_metric[metric_code] = chart
        findings = frozenset(str(value) for value in finding_types)
        chosen: list[tuple[bool, int, ChartSpec]] = []
        for priority, slot in enumerate(
            profile.active_slots(selected_check_ids)
        ):
            candidate = next(
                (
                    (item, by_metric[item.metric_code])
                    for item in slot.candidates
                    if item.metric_code in by_metric
                ),
                None,
            )
            if candidate is None:
                continue
            metric, chart = candidate
            if (
                slot.display_condition == "CHANGED_OR_ALERT"
                and not self._has_state_change(chart)
                and not bool(chart.metadata.get("has_related_alert"))
            ):
                continue
            metadata = {**chart.metadata, "visualization_profile_id": profile.profile_id, "slot_id": slot.slot_id}
            coverage = metadata.get("coverage_ratio")
            if isinstance(coverage, (int, float)) and coverage < self._minimum_coverage:
                metadata.update(
                    {
                        "coverage_warning": (
                            "监控采样覆盖率低于 80%，该图仅供参考，"
                            "不能据此给出强趋势结论。"
                        ),
                        "trend_conclusion_allowed": False,
                    }
                )
            else:
                metadata["trend_conclusion_allowed"] = True
            selected_chart = chart.model_copy(
                update={"title": metric.title, "metadata": metadata}
            )
            chosen.append(
                (bool(slot.finding_types & findings), priority, selected_chart)
            )
        chosen.sort(key=lambda item: (not item[0], item[1]))
        return tuple(item[2] for item in chosen[: profile.max_charts])

    @staticmethod
    def _has_state_change(chart: ChartSpec) -> bool:
        values = {
            point.y
            for series in chart.series
            for point in series.points
        }
        return len(values) > 1


def profile_metric_slots(
    profile_id: str | None,
    selected_check_ids: Iterable[str] = (),
) -> tuple[dict, ...]:
    profile = load_visualization_profiles().get(profile_id)
    if profile is None:
        return ()
    return tuple(
        {
            "slot_id": slot.slot_id,
            "label": slot.label,
            "required": slot.required,
            "metric_codes": tuple(
                item.metric_code for item in slot.candidates
            ),
        }
        for slot in profile.active_slots(selected_check_ids)
    )
