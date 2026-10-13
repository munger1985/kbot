"""实时监控 Readiness、查询预算、缓存键与视图投影。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

from aiops_agent.contracts.evidence import MetricObservation
from platform_core.contracts.aiops.monitoring import (
    MonitoringGap,
    MonitoringInstanceSummary,
    MonitoringPanel,
    MonitoringPoint,
    MonitoringSeries,
    MonitoringSourceSummary,
    MonitoringView,
    MonitoringWindow,
    MonitoringWindowName,
)
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_EVENT_QUERY,
    CAPABILITY_METRIC_QUERY_RANGE,
)


_WINDOW_SECONDS: dict[MonitoringWindowName, int] = {
    "15m": 900,
    "1h": 3600,
    "6h": 21_600,
    "24h": 86_400,
}


class MonitoringQueryError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def resolve_monitoring_window(
    window: MonitoringWindowName, *, now: datetime
) -> MonitoringWindow:
    end = now.astimezone(UTC)
    return MonitoringWindow(
        name=window,
        start=end - timedelta(seconds=_WINDOW_SECONDS[window]),
        end=end,
    )


def project_source_readiness(
    *,
    source_id: UUID,
    display_name: str,
    source_type: str,
    status: str,
    connectivity_status: str,
    capabilities: set[str],
    active_binding_count: int,
    metrics_aligned: bool,
    alert_ingress_ready: bool,
) -> MonitoringSourceSummary:
    if source_type not in {"PROMETHEUS", "ZABBIX"}:
        raise MonitoringQueryError(
            "MONITORING_SOURCE_TYPE_UNSUPPORTED",
            "一期实时监控只支持 Prometheus 和 Zabbix",
        )
    gaps: list[MonitoringGap] = []
    diagnostic_gaps: list[MonitoringGap] = []
    if status != "ENABLED":
        readiness = "DISABLED"
        gaps.append(_source_gap("SOURCE_DISABLED", "监控源未启用"))
    elif connectivity_status != "CONNECTED":
        readiness = "DISCONNECTED"
        gaps.append(_source_gap("SOURCE_DISCONNECTED", "监控源连接不可用", True))
    elif CAPABILITY_METRIC_QUERY_RANGE not in capabilities or not metrics_aligned:
        readiness = "CAPABILITY_MISSING"
        gaps.append(
            _source_gap(
                "SOURCE_METRIC_CAPABILITY_MISSING",
                "监控源缺少对齐后的时序指标能力",
            )
        )
    elif active_binding_count < 1:
        readiness = "NO_MAPPED_INSTANCE"
        gaps.append(_source_gap("SOURCE_NO_MAPPED_INSTANCE", "监控源尚未映射实例"))
    else:
        readiness = "READY"

    diagnostic_requirements = {
        CAPABILITY_METRIC_QUERY_RANGE,
        CAPABILITY_EVENT_QUERY,
    }
    if readiness != "READY":
        diagnostic = "UNAVAILABLE"
    elif (
        diagnostic_requirements.issubset(capabilities)
        and alert_ingress_ready
    ):
        diagnostic = "READY"
    else:
        diagnostic = "PARTIAL"
        if CAPABILITY_EVENT_QUERY not in capabilities:
            diagnostic_gaps.append(
                _source_gap(
                    "SOURCE_EVENT_QUERY_MISSING", "监控源缺少活动事件查询能力"
                )
            )
        if not alert_ingress_ready:
            diagnostic_gaps.append(
                _source_gap(
                    "SOURCE_ALERT_INGRESS_MISSING", "告警入站与恢复链路尚未就绪"
                )
            )
    return MonitoringSourceSummary(
        source_id=source_id,
        display_name=display_name,
        source_type=source_type,
        monitoring_readiness=readiness,
        diagnostic_readiness=diagnostic,
        capability_gaps=tuple(gaps),
        diagnostic_gaps=tuple(diagnostic_gaps),
    )


def build_monitoring_cache_key(
    *,
    domain_id: str,
    source_id: UUID,
    instance_ids: tuple[UUID, ...],
    profile_id: str,
    profile_version: str,
    profile_variant_ids: tuple[str, ...],
    window: str,
    end_time: datetime,
    binding_versions: tuple[int, ...],
    source_config_version: int,
    compare_source_id: UUID | None = None,
) -> str:
    payload = {
        "domain_id": domain_id,
        "source_id": str(source_id),
        "instance_ids": sorted(str(item) for item in instance_ids),
        "profile_id": profile_id,
        "profile_version": profile_version,
        "profile_variant_ids": sorted(profile_variant_ids),
        "compare_source_id": str(compare_source_id) if compare_source_id else None,
        "window": window,
        "end_time_bucket": int(end_time.timestamp()) // 15,
        "binding_versions": sorted(binding_versions),
        "source_config_version": source_config_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class MonitoringViewBuilder:
    def __init__(self, *, metric_catalog):
        self._metric_catalog = metric_catalog

    def build(
        self,
        *,
        source: MonitoringSourceSummary,
        profile,
        window: MonitoringWindow,
        instances: tuple[MonitoringInstanceSummary, ...],
        observations: tuple[MetricObservation, ...],
        gaps: tuple[MonitoringGap, ...] = (),
        generated_at: datetime | None = None,
    ) -> MonitoringView:
        self._validate_request(source, profile, instances)
        instance_by_id = {str(item.instance_id): item for item in instances}
        selected_ids = set(instance_by_id)
        selected_observations = tuple(
            item
            for item in observations
            if str(item.target_id) in selected_ids
            and item.metric_code in profile.metric_codes
        )
        panels = tuple(
            self._panel(
                source=source,
                metric_code=metric_code,
                instances=instance_by_id,
                observations=selected_observations,
                gaps=gaps,
            )
            for metric_code in profile.metric_codes
        )
        total_series = sum(len(item.series) for item in panels)
        if total_series > 96:
            raise MonitoringQueryError(
                "MONITORING_QUERY_BUDGET_EXCEEDED",
                "标准化 Series 数量超过单次请求预算",
            )
        current_instances = tuple(
            self._current_instance_summary(
                instance=instance,
                metric_codes=profile.metric_codes,
                observations=selected_observations,
                gaps=gaps,
            )
            for instance in instances
        )
        partial = bool(gaps) or any(
            item.quality != "GOOD" for item in panels
        )
        return MonitoringView(
            generated_at=generated_at or datetime.now(UTC),
            source=source,
            profile=profile.summary(),
            window=window,
            instances=current_instances,
            panels=panels,
            gaps=gaps,
            partial=partial,
        )

    @staticmethod
    def _current_instance_summary(
        *, instance, metric_codes, observations, gaps
    ):
        """使用本次查询证据覆盖历史 Binding 健康状态。"""

        instance_id = str(instance.instance_id)
        current_observations = tuple(
            item
            for item in observations
            if str(item.target_id) == instance_id
        )
        observed_metric_codes = {
            item.metric_code
            for item in current_observations
            if any(series.points for series in item.series)
        }
        if not observed_metric_codes:
            status = "UNAVAILABLE"
        else:
            has_gap = any(
                item.instance_id is None
                or str(item.instance_id) == instance_id
                for item in gaps
            )
            incomplete = observed_metric_codes != set(metric_codes)
            degraded_observation = any(
                item.coverage_ratio < 0.8 or item.truncated
                for item in current_observations
            )
            status = (
                "PARTIAL"
                if has_gap or incomplete or degraded_observation
                else "AVAILABLE"
            )
        return instance.model_copy(update={"status": status})

    @staticmethod
    def _validate_request(source, profile, instances) -> None:
        if source.monitoring_readiness != "READY":
            raise MonitoringQueryError(
                "MONITORING_SOURCE_NOT_READY", "监控源尚未满足实时查询门禁"
            )
        if not 1 <= len(instances) <= 12 or len(instances) != len(
            {item.instance_id for item in instances}
        ):
            raise MonitoringQueryError(
                "MONITORING_QUERY_BUDGET_EXCEEDED", "实例数量无效或超过 12 个"
            )
        supported = set(profile.supported_db_types)
        if any(item.db_type not in supported for item in instances):
            raise MonitoringQueryError(
                "MONITORING_PROFILE_INCOMPATIBLE",
                "Profile 与选定实例的数据库类型不兼容",
            )

    def _panel(self, *, source, metric_code, instances, observations, gaps):
        definition = self._metric_catalog.get(metric_code)
        metric_observations = tuple(
            item for item in observations if item.metric_code == metric_code
        )
        series: list[MonitoringSeries] = []
        summaries = []
        for observation in sorted(
            metric_observations, key=lambda item: str(item.target_id)
        ):
            instance = instances[str(observation.target_id)]
            summaries.append(observation.summary)
            for item in observation.series:
                identity = json.dumps(
                    {
                        "instance_id": str(observation.target_id),
                        "source_id": str(source.source_id),
                        "metric_code": metric_code,
                        "dimensions": item.dimensions,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                series.append(
                    MonitoringSeries(
                        source_id=source.source_id,
                        source_display_name=source.display_name,
                        source_type=source.source_type,
                        instance_id=instance.instance_id,
                        instance_display_name=instance.display_name,
                        series_key=hashlib.sha256(identity.encode()).hexdigest(),
                        dimensions=item.dimensions,
                        points=tuple(
                            MonitoringPoint.model_validate(
                                point.model_dump(mode="python")
                            )
                            for point in item.points[:240]
                        ),
                        coverage_ratio=observation.coverage_ratio,
                    )
                )
        metric_gaps = tuple(
            item for item in gaps if item.metric_code == metric_code
        )
        if not series:
            quality = "NO_DATA"
        elif metric_gaps or any(
            item.coverage_ratio < 0.8 or item.truncated
            for item in metric_observations
        ):
            quality = "PARTIAL"
        else:
            quality = "GOOD"
        return MonitoringPanel(
            panel_id=metric_code,
            title=definition.name,
            description=definition.description,
            visualization=(
                "STATE_TIMELINE"
                if definition.value_kind == "STATE"
                else "TIME_SERIES"
            ),
            metric_code=metric_code,
            semantic_role=definition.semantic_role,
            unit=definition.unit,
            value_kind=definition.value_kind,
            series=tuple(series),
            summary=_merge_summaries(summaries),
            quality=quality,
        )


def _source_gap(code: str, detail: str, retryable: bool = False) -> MonitoringGap:
    return MonitoringGap(
        scope="SOURCE", code=code, detail=detail, retryable=retryable
    )


def _merge_summaries(values: list[dict]) -> dict[str, float | int | str | None]:
    if not values:
        return {}
    numeric_last = [
        item.get("last")
        for item in values
        if isinstance(item.get("last"), (int, float))
    ]
    return {
        "instance_count": len(values),
        "last_min": min(numeric_last) if numeric_last else None,
        "last_max": max(numeric_last) if numeric_last else None,
    }
