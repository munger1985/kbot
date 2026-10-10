"""实时监控来源、实例、Profile 与 View 应用用例。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from aiohttp import ClientError

from aiops_agent.adapters.diagnostic_sources.base import (
    DiagnosticSourceAdapterError,
)
from aiops_agent.application.errors import (
    AIOpsApplicationError,
    resource_not_found,
)
from aiops_agent.application.monitoring_snapshot import MonitoringSnapshotBuilder
from platform_core.contracts.aiops.monitoring import (
    MonitoringGap,
    MonitoringInstanceSummary,
    MonitoringSourceSummary,
    MonitoringView,
    MonitoringViewRequest,
    MonitoringWindow,
)
from aiops_agent.contracts.evidence import ObservationGap
from aiops_agent.monitoring import (
    MonitoringQueryError,
    MonitoringViewBuilder,
    build_monitoring_cache_key,
    project_source_readiness,
    resolve_grafana_integration,
    resolve_metric_definitions,
    resolve_monitoring_window,
)
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_EVENT_RECEIVE,
    DiagnosticSourceContext,
    MetricsEvidenceResult,
    MetricsEvidenceRequest,
)


class MonitoringApplicationService:
    """只通过 UoW、冻结快照和 Adapter 生成脱敏实时视图。"""

    def __init__(
        self,
        *,
        uow_factory,
        metric_catalog,
        profile_catalog,
        diagnostic_source_registry,
        secret_store,
        max_response_bytes: int,
    ):
        self._uow_factory = uow_factory
        self._metric_catalog = metric_catalog
        self._profile_catalog = profile_catalog
        self._diagnostic_source_registry = diagnostic_source_registry
        self._secret_store = secret_store
        self._max_response_bytes = max_response_bytes
        self._snapshot_builder = MonitoringSnapshotBuilder(
            metric_catalog=metric_catalog,
            default_window_seconds=3600,
            max_response_bytes=max_response_bytes,
        )
        self._view_builder = MonitoringViewBuilder(
            metric_catalog=metric_catalog
        )
        self._cache: dict[str, tuple[datetime, MonitoringView]] = {}

    async def list_sources(self, *, scope) -> tuple[MonitoringSourceSummary, ...]:
        async with self._uow_factory() as uow:
            sources = await uow.diagnostic_sources.page_scoped(
                domain_id=scope.domain_id,
                statuses=None,
                before_created_at=None,
                before_id=None,
                limit=1000,
            )
            summaries = []
            for source in sources:
                if source.source_type not in {"PROMETHEUS", "ZABBIX"}:
                    continue
                bindings = await uow.targets.list_source_bindings_by_source(
                    diagnostic_source_id=source.diagnostic_source_id,
                    domain_id=scope.domain_id,
                )
                active = [item for item in bindings if item.status == "ACTIVE"]
                enabled_count = 0
                for binding in active:
                    target = await uow.targets.get_scoped(
                        target_id=binding.target_id,
                        domain_id=scope.domain_id,
                    )
                    if target is not None and target.status == "ENABLED":
                        enabled_count += 1
                summaries.append(self._source_summary(source, enabled_count))
            return tuple(
                sorted(
                    summaries,
                    key=lambda item: (item.display_name, str(item.source_id)),
                )
            )

    async def list_instances(
        self, *, scope, source_id: UUID
    ) -> tuple[MonitoringInstanceSummary, ...]:
        async with self._uow_factory() as uow:
            source = await self._source(uow, scope.domain_id, source_id)
            bindings = await uow.targets.list_source_bindings_by_source(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
            )
            active = [item for item in bindings if item.status == "ACTIVE"]
            mapped_targets = []
            for binding in active:
                target = await uow.targets.get_scoped(
                    target_id=binding.target_id,
                    domain_id=scope.domain_id,
                )
                if target is None or target.status != "ENABLED":
                    continue
                mapped_targets.append((target, binding))
            source_summary = self._source_summary(source, len(mapped_targets))
            instances = [
                self._instance_summary(
                    target=target,
                    binding=binding,
                    source=source_summary,
                )
                for target, binding in mapped_targets
            ]
            return tuple(
                sorted(
                    instances,
                    key=lambda item: (item.display_name, str(item.instance_id)),
                )
            )

    async def list_profiles(self, *, scope, source_id: UUID):
        instances = await self.list_instances(scope=scope, source_id=source_id)
        db_types = tuple(sorted({item.db_type for item in instances}))
        if not db_types:
            return ()
        return self._profile_catalog.list_for_db_types(db_types)

    async def list_target_profiles(self, *, scope, target_id: UUID):
        async with self._uow_factory() as uow:
            target = await uow.targets.get_scoped(
                target_id=target_id, domain_id=scope.domain_id
            )
            if target is None or target.status != "ENABLED":
                raise resource_not_found("Monitoring Instance")
            return self._profile_catalog.list_for_db_types((target.db_type,))

    async def get_target_view(
        self,
        *,
        scope,
        target_id: UUID,
        source_id: UUID,
        profile_id: str,
        window: str,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        compare_source_id: UUID | None = None,
    ) -> MonitoringView:
        generated_at = datetime.now(UTC)
        explicit_window = self._explicit_window(
            window_start=window_start,
            window_end=window_end,
            now=generated_at,
        )
        try:
            request = MonitoringViewRequest(
                profile_id=profile_id,
                instance_ids=(target_id,),
                window=window,
            )
        except ValueError as exc:
            raise AIOpsApplicationError(
                code="MONITORING_REQUEST_INVALID",
                message="实时监控请求参数无效",
                status_code=422,
            ) from exc
        if compare_source_id == source_id:
            raise AIOpsApplicationError(
                code="MONITORING_COMPARE_SOURCE_INVALID",
                message="对比来源必须与主来源不同",
                status_code=422,
            )
        primary = await self.get_view(
            scope=scope,
            source_id=source_id,
            request=request,
            now=generated_at,
            window_override=explicit_window,
        )
        if compare_source_id is None:
            return primary
        comparison = await self.get_view(
            scope=scope,
            source_id=compare_source_id,
            request=request,
            now=generated_at,
            window_override=explicit_window,
        )
        return self._merge_comparison(primary, comparison)

    async def get_view(
        self,
        *,
        scope,
        source_id: UUID,
        request: MonitoringViewRequest,
        now: datetime | None = None,
        window_override: MonitoringWindow | None = None,
    ) -> MonitoringView:
        generated_at = (now or datetime.now(UTC)).astimezone(UTC)
        try:
            profile = self._profile_catalog.get(request.profile_id)
        except KeyError as exc:
            raise AIOpsApplicationError(
                code="MONITORING_PROFILE_NOT_FOUND",
                message="Monitoring Profile 不存在",
                status_code=404,
            ) from exc
        requested = set(request.instance_ids)
        window = window_override or resolve_monitoring_window(
            request.window, now=generated_at
        )
        async with self._uow_factory() as uow:
            source = await self._source(uow, scope.domain_id, source_id)
            bindings = [
                item
                for item in await uow.targets.list_source_bindings_by_source(
                    diagnostic_source_id=source_id,
                    domain_id=scope.domain_id,
                )
                if item.status == "ACTIVE" and item.target_id in requested
            ]
            if {item.target_id for item in bindings} != requested:
                raise resource_not_found("Monitoring Instance")
            targets = []
            for binding in bindings:
                target = await uow.targets.get_scoped(
                    target_id=binding.target_id,
                    domain_id=scope.domain_id,
                )
                if target is None or target.status != "ENABLED":
                    raise resource_not_found("Monitoring Instance")
                targets.append((target, binding))
            source_summary = self._source_summary(source, len(bindings))
            self._validate_query_gate(source_summary, profile, targets)
            cache_key = build_monitoring_cache_key(
                domain_id=str(scope.domain_id),
                source_id=source_id,
                instance_ids=request.instance_ids,
                profile_id=profile.profile_id,
                window=(
                    request.window
                    if window.name is not None
                    else f"{window.start.isoformat()}/{window.end.isoformat()}"
                ),
                end_time=generated_at,
                binding_versions=tuple(
                    int(item.row_version) for _, item in targets
                ),
                source_config_version=int(source.row_version),
            )
            cached = self._cache.get(cache_key)
            if cached is not None and cached[0] > generated_at:
                return cached[1]
            instances = tuple(
                self._instance_summary(
                    target=target, binding=binding, source=source_summary
                )
                for target, binding in targets
            )
            observations = []
            gaps = []
            for target, binding in targets:
                try:
                    snapshot = await self._snapshot_builder.build(
                        uow=uow,
                        domain_id=scope.domain_id,
                        target=target,
                        now=generated_at,
                        allowed_source_ids=(source_id,),
                        window_start=window.start,
                        window_end=window.end,
                        requested_metric_codes=profile.metric_codes,
                    )
                except (AIOpsApplicationError, KeyError, ValueError):
                    gaps.append(
                        MonitoringGap(
                            scope="INSTANCE",
                            source_id=source_id,
                            instance_id=target.target_id,
                            code="SOURCE_CONFIGURATION_INVALID",
                            detail="监控源或实例映射配置无效",
                        )
                    )
                    continue
                gaps.extend(
                    self._snapshot_gap(target.target_id, source_id, item)
                    for item in snapshot["initial_gaps"]
                )
                if not snapshot["bindings"]:
                    continue
                frozen = snapshot["bindings"][0]
                result = await self._query_binding(
                    target_id=target.target_id,
                    frozen=frozen,
                    window=window,
                    trace_id=scope.trace_id,
                    max_response_bytes=max(
                        1024, self._max_response_bytes // len(targets)
                    ),
                )
                observations.extend(result.observations)
                gaps.extend(
                    MonitoringGap(
                        scope="METRIC" if item.metric_code else "INSTANCE",
                        source_id=source_id,
                        instance_id=target.target_id,
                        metric_code=item.metric_code,
                        code=item.code,
                        detail=item.detail,
                        retryable=item.retryable,
                    )
                    for item in result.gaps
                )
        try:
            view = self._view_builder.build(
                source=source_summary,
                profile=profile,
                window=window,
                instances=instances,
                observations=tuple(observations),
                gaps=tuple(gaps),
                generated_at=generated_at,
            )
            view = view.model_copy(
                update={
                    "dashboard": resolve_grafana_integration(
                        source_type=source_summary.source_type,
                        profile_dashboard_uid=profile.grafana_dashboard_uid,
                        instance_count=len(instances),
                    )
                }
            )
        except MonitoringQueryError as exc:
            raise AIOpsApplicationError(
                code=exc.code,
                message=str(exc),
                status_code=(
                    409 if exc.code == "MONITORING_SOURCE_NOT_READY" else 422
                ),
            ) from exc
        if len(view.model_dump_json().encode("utf-8")) > self._max_response_bytes:
            raise AIOpsApplicationError(
                code="MONITORING_QUERY_BUDGET_EXCEEDED",
                message="标准化监控视图超过单次响应预算",
                status_code=422,
            )
        self._cache[cache_key] = (generated_at + timedelta(seconds=15), view)
        return view

    @staticmethod
    def _merge_comparison(
        primary: MonitoringView,
        comparison: MonitoringView,
    ) -> MonitoringView:
        if comparison.source.source_type == primary.source.source_type:
            raise AIOpsApplicationError(
                code="MONITORING_COMPARE_SOURCE_TYPE_DUPLICATE",
                message="单实例来源对比仅支持 Prometheus 与 Zabbix 互比",
                status_code=422,
            )
        comparison_panels = {
            panel.metric_code: panel for panel in comparison.panels
        }
        panels = []
        for panel in primary.panels:
            other = comparison_panels.get(panel.metric_code)
            if other is None:
                panels.append(panel)
                continue
            combined_series = panel.series + other.series
            if not combined_series:
                quality = "NO_DATA"
            elif panel.quality == "GOOD" and other.quality == "GOOD":
                quality = "GOOD"
            else:
                quality = "PARTIAL"
            panels.append(
                panel.model_copy(
                    update={
                        "series": combined_series,
                        "summary": {
                            "source_count": 2,
                            "series_count": len(combined_series),
                        },
                        "quality": quality,
                    }
                )
            )
        return primary.model_copy(
            update={
                "compare_source": comparison.source,
                "panels": tuple(panels),
                "gaps": primary.gaps + comparison.gaps,
                "partial": primary.partial or comparison.partial,
            }
        )

    async def _query_binding(
        self, *, target_id, frozen, window, trace_id, max_response_bytes
    ):
        source = frozen["source"]
        try:
            credentials = {}
            if source.get("secret_ref"):
                secret = await self._secret_store.resolve(source["secret_ref"])
                credentials = dict(secret.values)
                if "value" in credentials:
                    credentials["token"] = credentials["value"]
            context = DiagnosticSourceContext(
                source_id=str(source["source_id"]),
                source_type=source["source_type"],
                adapter_id=source["adapter_id"],
                adapter_version=source["adapter_version"],
                config_version=source["config_version"],
                endpoint=source["endpoint"],
                credentials=credentials,
                declared_capabilities=source["declared_capabilities"],
                config=source["config"],
            )
            adapter = self._diagnostic_source_registry.create(
                context, capability="metric.query_range"
            )
            seconds = int((window.end - window.start).total_seconds())
            return await adapter.query_metrics(
                MetricsEvidenceRequest(
                    target_id=str(target_id),
                    binding_id=str(frozen["binding_id"]),
                    source_locator_key=frozen["source_locator_key"],
                    source_locator=dict(frozen["source_locator"]),
                    metric_definitions=resolve_metric_definitions(frozen),
                    window_start=window.start,
                    window_end=window.end,
                    requested_step_seconds=max(60, seconds // 240),
                    max_response_bytes=max_response_bytes,
                    trace_id=trace_id,
                )
            )
        except AIOpsApplicationError as exc:
            return self._query_failure(
                frozen,
                code="SOURCE_CREDENTIAL_UNAVAILABLE",
                detail="监控源凭据本次不可用",
                retryable=exc.retryable,
            )
        except DiagnosticSourceAdapterError as exc:
            return self._query_failure(
                frozen,
                code=exc.code,
                detail="监控源查询失败",
                retryable=exc.retryable,
            )
        except (ClientError, TimeoutError):
            return self._query_failure(
                frozen,
                code="SOURCE_UNREACHABLE",
                detail="监控源暂时不可达",
                retryable=True,
            )
        except (LookupError, TypeError, ValueError):
            return self._query_failure(
                frozen,
                code="SOURCE_CONFIGURATION_INVALID",
                detail="监控源或实例映射配置无效",
            )

    @staticmethod
    def _query_failure(frozen, *, code, detail, retryable=False):
        return MetricsEvidenceResult(
            gaps=(
                ObservationGap(
                    source_id=frozen["source"]["source_id"],
                    binding_id=frozen["binding_id"],
                    code=code,
                    detail=detail,
                    retryable=retryable,
                ),
            )
        )

    @staticmethod
    def _explicit_window(*, window_start, window_end, now):
        if window_start is None and window_end is None:
            return None
        if window_start is None or window_end is None:
            raise AIOpsApplicationError(
                code="MONITORING_REQUEST_INVALID",
                message="精确时间窗起止时间必须同时提供",
                status_code=422,
            )
        try:
            window = MonitoringWindow(start=window_start, end=window_end)
        except ValueError as exc:
            raise AIOpsApplicationError(
                code="MONITORING_REQUEST_INVALID",
                message="精确时间窗无效",
                status_code=422,
            ) from exc
        if window.end > now + timedelta(seconds=5):
            raise AIOpsApplicationError(
                code="MONITORING_REQUEST_INVALID",
                message="精确时间窗结束时间不能位于未来",
                status_code=422,
            )
        return window

    async def _source(self, uow, domain_id, source_id):
        source = await uow.diagnostic_sources.get_scoped(
            diagnostic_source_id=source_id, domain_id=domain_id
        )
        if source is None or source.source_type not in {"PROMETHEUS", "ZABBIX"}:
            raise resource_not_found("Monitoring Source")
        return source

    @staticmethod
    def _validate_query_gate(source, profile, targets):
        if source.monitoring_readiness != "READY":
            raise AIOpsApplicationError(
                code="MONITORING_SOURCE_NOT_READY",
                message="监控源尚未满足实时查询门禁",
                status_code=409,
            )
        supported = set(profile.supported_db_types)
        if any(target.db_type not in supported for target, _ in targets):
            raise AIOpsApplicationError(
                code="MONITORING_PROFILE_INCOMPATIBLE",
                message="Profile 与选定实例的数据库类型不兼容",
                status_code=422,
            )

    @staticmethod
    def _source_summary(source, active_binding_count):
        capabilities = set((source.declared_capabilities_json or {}).keys())
        config = dict(source.config_json or {})
        alert_ready = (
            CAPABILITY_EVENT_RECEIVE in capabilities
            if source.source_type == "ZABBIX"
            else bool(config.get("alert_ingress_ready"))
        )
        return project_source_readiness(
            source_id=source.diagnostic_source_id,
            display_name=source.display_name,
            source_type=source.source_type,
            status=source.status,
            connectivity_status=source.connectivity_status,
            capabilities=capabilities,
            active_binding_count=active_binding_count,
            metrics_aligned=True,
            alert_ingress_ready=alert_ready,
        )

    @staticmethod
    def _instance_summary(*, target, binding, source):
        degraded = binding.health_status not in {"READY", "HEALTHY", "UNKNOWN"}
        return MonitoringInstanceSummary(
            instance_id=target.target_id,
            display_name=target.display_name,
            db_type=target.db_type,
            status="PARTIAL" if degraded else "AVAILABLE",
            monitoring_readiness=source.monitoring_readiness,
            diagnostic_readiness=source.diagnostic_readiness,
            capability_gaps=source.capability_gaps,
        )

    @staticmethod
    def _snapshot_gap(instance_id, source_id, item):
        return MonitoringGap(
            scope="METRIC" if item.get("metric_code") else "INSTANCE",
            source_id=source_id,
            instance_id=instance_id,
            metric_code=item.get("metric_code"),
            code=str(item.get("code") or "SOURCE_QUERY_UNSUPPORTED"),
            detail=str(item.get("detail") or "监控证据不可用"),
        )
