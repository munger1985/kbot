"""OEM 原生监控：采集前后核验授权资源与绑定，目录和采样独立限额。"""

from datetime import UTC, datetime, timedelta
import asyncio
from importlib.resources import files
import json
from urllib.parse import unquote

from aiohttp import ClientError

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.errors import AIOpsApplicationError, resource_not_found, validation_failed
from aiops_agent.application.monitoring_snapshot import freeze_source_config
from aiops_agent.application.monitoring_oem_overview import baseline_series, overview_panels, freshness_seconds
from aiops_agent.monitoring import resolve_monitoring_baseline_window, resolve_monitoring_window
from aiops_agent.ports.diagnostic_source import (
    DiagnosticSourceContext, MetricsDiscoveryRequest, NativeMetricQueryRequest,
)
from platform_core.contracts.aiops.monitoring_native import (
    MonitoringNativeBaseline, MonitoringNativeCatalog, MonitoringNativeMetric, MonitoringNativePoint, MonitoringNativeSamples,
)
from platform_core.contracts.aiops.oracle_services import ORACLE_SERVICE_TYPES


def public_metric(entry):
    return MonitoringNativeMetric(
        name=entry.name, display_name=entry.help_text or entry.name,
        group=unquote(entry.name.split("/", 1)[0]),
        native_value_type=entry.native_value_type or "UNKNOWN", unit=entry.unit,
        dimensions=entry.label_names, query_supported=entry.query_supported,
        latest_supported=entry.native_value_type in {"NUMBER", "STRING"},
    )


def public_points(points, dimensions, *, captured_at, max_age, truncated, latest):
    """历史有效值可画趋势，但只有新鲜且完整的末次值能作当前比较。"""
    result = []
    for (when, value), labels in zip(points, dimensions, strict=True):
        empty = value is None or not value.strip()
        fresh = max_age is not None and timedelta(0) <= captured_at - when <= timedelta(seconds=max_age)
        quality = "TRUNCATED" if truncated else "EMPTY" if empty else "STALE" if latest and not fresh else "OBSERVED"
        result.append(MonitoringNativePoint(observed_at=when, value=value, dimensions=labels,
            quality=quality, current_eligible=bool(fresh and not empty and not truncated)))
    return tuple(result)


class NativeMonitoringService:
    """复用监控应用依赖；不写业务配置，也不向浏览器开放 Provider 查询。"""

    def __init__(self, monitoring):
        self._monitoring = monitoring
        self._catalog_cache = {}
        self._baseline_cache = {}
        from .oracle_service_monitoring import OracleServiceMonitoring
        self.oracle = OracleServiceMonitoring(self)
        self._defaults = json.loads(files("aiops_agent.resources").joinpath("monitoring/oem_native.v1.json").read_text(encoding="utf-8"))["defaults"]

    async def _binding(self, scope, source_id, instance_id):
        async with self._monitoring._uow_factory() as uow:
            rows = await uow.business_systems.list_monitoring_catalog(
                tenant_id=getattr(scope, "tenant_id", None), domain_id=scope.domain_id)
            for _, resource, target, binding in rows:
                if resource is None or binding is None or binding.diagnostic_source_id != source_id:
                    continue
                if (target.target_id if target is not None else resource.resource_id) != instance_id:
                    continue
                if target is not None and target.status != "ENABLED":
                    continue
                service_resource = resource.resource_type_key in ORACLE_SERVICE_TYPES
                if not service_resource and resource.resource_type_key not in {"DATABASE", "PHYSICAL_SERVER", "VIRTUAL_MACHINE", "CONTAINER"}:
                    continue
                source = await self._monitoring._source(uow, scope.domain_id, source_id)
                if source.source_type not in ({"OEM", "PROMETHEUS", "ZABBIX"} if service_resource else {"OEM"}) or getattr(source, "tenant_id", None) != getattr(scope, "tenant_id", None):
                    raise resource_not_found("OEM Monitoring Resource")
                if source.status != "ENABLED" or source.connectivity_status != "CONNECTED":
                    raise AIOpsApplicationError(code="MONITORING_SOURCE_NOT_READY", message="OEM 监控源未启用或尚未连接", status_code=409)
                frozen_source = freeze_source_config(source)
                capability_scope = dict(binding.capability_scope_json or {})
                allowed = capability_scope.get("capabilities")
                if allowed is not None:
                    if not isinstance(allowed, (list, tuple)) or not allowed or not all(isinstance(item, str) for item in allowed):
                        raise validation_failed("监控绑定 capabilities 格式无效")
                    frozen_source["declared_capabilities"] = {name: value for name, value in frozen_source["declared_capabilities"].items() if name in allowed}
                budget = dict(binding.query_budget_json or {})
                max_bytes = budget.get("max_response_bytes", self._monitoring._max_response_bytes)
                if type(max_bytes) is not int or max_bytes < 1024:
                    raise validation_failed("监控绑定响应字节预算无效")
                return {
                    "source": frozen_source, "locator": binding.source_locator_key,
                    "binding_id": binding.resource_source_binding_id, "version": binding.row_version,
                    "resource_id": resource.resource_id,
                    "target_type": (binding.source_locator_json or {}).get("target_type"),
                    "capability_scope": capability_scope, "query_budget": budget,
                    "max_response_bytes": min(1048576, self._monitoring._max_response_bytes, max_bytes),
                    "scope": resource.resource_type_key if service_resource else "DATABASE" if resource.resource_type_key == "DATABASE" else "HOST",
                    **({"source_locator": dict(binding.source_locator_json or {}),
                        "mapping_overrides": dict(binding.mapping_overrides_json or {})} if service_resource else {}),
                }
        raise resource_not_found("OEM Monitoring Resource")

    async def _adapter(self, frozen, capability):
        source = frozen["source"]
        if capability not in source["declared_capabilities"]:
            raise DiagnosticSourceAdapterError("SOURCE_CAPABILITY_FORBIDDEN", "监控绑定未授权此指标读取能力")
        credentials = {}
        if source["secret_ref"]:
            credentials = dict((await self._monitoring._secret_store.resolve(source["secret_ref"])).values)
            if "value" in credentials:
                credentials["token"] = credentials["value"]
        return self._monitoring._diagnostic_source_registry.create(DiagnosticSourceContext(
            source_id=source["source_id"], source_type=source["source_type"], adapter_id=source["adapter_id"],
            adapter_version=source["adapter_version"], config_version=source["config_version"],
            endpoint=source["endpoint"], credentials=credentials,
            declared_capabilities=source["declared_capabilities"], config=source["config"],
        ), capability=capability)

    async def _unchanged(self, scope, source_id, instance_id, frozen):
        if await self._binding(scope, source_id, instance_id) != frozen:
            raise AIOpsApplicationError(code="MONITORING_SOURCE_CHANGED", message="资源或监控绑定已变化，请重新查询", status_code=409)

    async def _catalog(self, scope, source_id, instance_id, frozen):
        now = datetime.now(UTC)
        key = (str(scope.domain_id), str(getattr(scope, "tenant_id", None)), source_id, instance_id)
        cached = self._catalog_cache.get(key)
        if cached and cached[0] > now and cached[1] == frozen:
            return cached[2]
        adapter = await self._adapter(frozen, "metric.discover")
        async with asyncio.timeout(adapter._timeout.total):
            result = await adapter.discover_monitoring_metrics(MetricsDiscoveryRequest(
                action_id="a0", target_id=str(instance_id), binding_id=str(frozen["binding_id"]),
                source_locator_key=frozen["locator"], scope=frozen["scope"],
                window_start=now-timedelta(hours=1), window_end=now, trace_id=scope.trace_id,
                max_response_bytes=frozen["max_response_bytes"],
            ))
        if len(self._catalog_cache) >= 32:
            self._catalog_cache.clear()
        self._catalog_cache[key] = (now+timedelta(seconds=30), frozen, result)
        return result

    async def catalog(self, *, scope, source_id, instance_id, search="", offset=0, limit=100):
        frozen = await self._binding(scope, source_id, instance_id)
        if frozen['scope'] in ORACLE_SERVICE_TYPES:
            try:
                return await self.oracle.catalog(scope=scope, source_id=source_id, instance_id=instance_id,
                    frozen=frozen, search=search, offset=offset, limit=limit)
            except (DiagnosticSourceAdapterError, ClientError, TimeoutError, LookupError) as exc:
                self._raise_provider(exc)
        try:
            entries, truncated, revision = await self._catalog(scope, source_id, instance_id, frozen)
        except (DiagnosticSourceAdapterError, ClientError, TimeoutError, LookupError) as exc:
            self._raise_provider(exc)
        await self._unchanged(scope, source_id, instance_id, frozen)
        metrics = sorted((self._metric(entry, frozen) for entry in entries), key=lambda entry: entry.name)
        query = search.strip().casefold()
        filtered = [entry for entry in metrics if not query or query in f"{entry.name} {entry.display_name} {entry.group}".casefold()]
        default_metric = next((entry for preferred in self._defaults[frozen["scope"]] for entry in metrics
            if entry.name == preferred["name"] and entry.unit == preferred["unit"]
            and entry.native_value_type == preferred["native_value_type"] and entry.query_supported), None)
        panels = overview_panels(frozen["scope"], metrics, target_type=frozen["target_type"])
        if frozen["target_type"] == "oracle_pdb":
            default_metric = next((panel.metric for panel in panels if panel.metric and panel.metric.query_supported), None)
        return self._response(MonitoringNativeCatalog(
            source_id=source_id, instance_id=instance_id, entries=tuple(filtered[offset:offset+limit]),
            default_metric=default_metric,
            discovered_count=len(filtered), next_offset=offset+limit if offset+limit < len(filtered) else None,
            truncated=truncated, revision=revision,
            warnings=("目录来自该资源实际安装的 OEM 插件；目录可见不代表窗口内已有采样。",
                      "数值指标支持历史时序；数值与文本指标另可读取最近采集快照，快照不代表历史覆盖。"),
            overview_panels=panels,
        ), frozen)

    async def samples(self, *, scope, source_id, instance_id, query):
        frozen = await self._binding(scope, source_id, instance_id)
        if frozen['scope'] in ORACLE_SERVICE_TYPES:
            try:
                return await self.oracle.samples(scope=scope, source_id=source_id, instance_id=instance_id, frozen=frozen, query=query)
            except (DiagnosticSourceAdapterError, ClientError, TimeoutError, LookupError) as exc:
                self._raise_provider(exc)
        window = resolve_monitoring_window(query.window, now=datetime.now(UTC))
        try:
            entries, _, _ = await self._catalog(scope, source_id, instance_id, frozen)
            entry = next((entry for entry in entries if entry.name == query.metric_name), None)
            if entry is None:
                raise resource_not_found("OEM Metric")
            if query.mode == "HISTORY" and not entry.query_supported:
                raise AIOpsApplicationError(code="SOURCE_METRIC_TYPE_MISMATCH", message="该 OEM 指标不是数值时序；已保留原生目录类型", status_code=422)
            adapter = await self._adapter(frozen, "metric.query_range")
            deadline = asyncio.get_running_loop().time() + adapter._timeout.total
            if query.baseline and frozen["max_response_bytes"] < 2048:
                raise validation_failed("同时读取当前值与历史基线至少需要 2048 字节预算")
            native_request = NativeMetricQueryRequest(
                action_id="a0", target_id=str(instance_id), binding_id=str(frozen["binding_id"]),
                source_locator_key=frozen["locator"], window_start=window.start, window_end=window.end,
                trace_id=scope.trace_id, max_response_bytes=frozen["max_response_bytes"])
            if query.mode == "LATEST":
                async with asyncio.timeout(adapter._timeout.total):
                    latest = await adapter.query_latest_metric(native_request, metric_name=query.metric_name,
                        limit=1000, _expected_target_type=frozen["target_type"])
                if latest["unit"] != entry.unit or latest["native_value_type"] != entry.native_value_type:
                    raise DiagnosticSourceAdapterError("SOURCE_METADATA_CHANGED", "OEM 指标元数据已变化")
                if any(set(labels) != set(entry.label_names) for labels in latest["dimensions"]):
                    raise DiagnosticSourceAdapterError("SOURCE_DIMENSIONS_MISMATCH", "OEM 指标维度已变化")
                await self._unchanged(scope, source_id, instance_id, frozen)
                baseline = await self._baseline(adapter, native_request, frozen, entry, query.baseline, deadline)
                await self._unchanged(scope, source_id, instance_id, frozen)
                metric = self._metric(entry, frozen)
                return self._response(MonitoringNativeSamples(
                    source_id=source_id, instance_id=instance_id, metric=metric,
                    window_start=None, window_end=None, captured_at=latest["captured_at"],
                    points=public_points(latest["points"], latest["dimensions"], captured_at=latest["captured_at"],
                        max_age=metric.current_max_age_seconds, truncated=latest["truncated"], latest=True),
                    truncated=latest["truncated"], response_sha256=latest["response_sha256"],
                    query_mode="LATEST", history_resolution="LATEST_SNAPSHOT",
                    baseline=baseline,
                    warnings=("这是 OEM 最近一次采集快照，不受所选历史时间窗约束；请核对采集时间，不视为实时或完整历史。",
                        "原始文本、空值及单位不转换为健康结论，不以快照替代历史查询。",
                        "当前比较采用已审核总览的保守时效策略，不代表 OEM 在线采集调度；未审核时效的指标仅保留原始采样。")), frozen)
            async with asyncio.timeout(adapter._timeout.total):
                samples = await adapter.query_native_metric(native_request,
                    metric_name=query.metric_name, limit=1000, _paginate=True,
                    _expected_target_type=frozen["target_type"])
            if samples.unit != entry.unit:
                raise DiagnosticSourceAdapterError("SOURCE_UNIT_MISMATCH", "OEM 指标单位在目录与采样间发生变化")
            if any(set(labels) != set(entry.label_names) for labels in samples.sample_dimensions):
                raise DiagnosticSourceAdapterError("SOURCE_DIMENSIONS_MISMATCH", "OEM 指标维度在目录与采样间发生变化")
            baseline = await self._baseline(adapter, native_request, frozen, entry, query.baseline, deadline)
        except (DiagnosticSourceAdapterError, ClientError, TimeoutError, LookupError) as exc:
            self._raise_provider(exc)
        await self._unchanged(scope, source_id, instance_id, frozen)
        metric = self._metric(entry, frozen)
        return self._response(MonitoringNativeSamples(
            source_id=source_id, instance_id=instance_id, metric=metric,
            window_start=window.start, window_end=window.end, captured_at=samples.captured_at,
            points=public_points(samples.points, samples.sample_dimensions, captured_at=samples.captured_at,
                max_age=metric.current_max_age_seconds, truncated=samples.truncated, latest=False),
            truncated=samples.truncated, response_sha256=samples.response_sha256,
            baseline=baseline,
            warnings=("OEM 自动选择原始、小时或日聚合；采样覆盖率未知，不推断 counter/gauge 或自动换算速率。",
                      "窗口无采样表示暂无数据，不代表资源宕机；缺失值不补零。",
                      "历史趋势保留原采样；仅完整且符合审核时效的末次值参与当前比较。"),
        ), frozen)

    @staticmethod
    def _metric(entry, frozen):
        metric = public_metric(entry)
        return metric.model_copy(update={"current_max_age_seconds": freshness_seconds(
            frozen["scope"], metric, target_type=frozen["target_type"])})

    async def _baseline(self, adapter, request, frozen, entry, period, deadline):
        if period is None or entry.native_value_type != "NUMBER":
            return None
        now = datetime.now(UTC)
        window = resolve_monitoring_baseline_window(period, now=now)
        key = (str(request.target_id), str(frozen["binding_id"]), entry.name, period)
        cached = self._baseline_cache.get(key)
        if cached and cached[0] > now and cached[1] == frozen:
            return cached[2]
        remaining = adapter._remaining_response_bytes
        if remaining < 1024:
            return MonitoringNativeBaseline(period=period, start=window.start, end=window.end,
                unavailable_reason="当前采样已使用本次读取预算，剩余字节不足以读取完整基线；当前趋势保留。")
        try:
            async with asyncio.timeout(max(0, deadline - asyncio.get_running_loop().time())):
                summary = await adapter.query_native_metric(
                    request.model_copy(update={"window_start": window.start, "window_end": window.end,
                        "max_response_bytes": min(request.max_response_bytes, remaining)}),
                    metric_name=entry.name, limit=1000, _paginate=True, _summary_only=True,
                    _expected_target_type=frozen["target_type"])
            if summary["unit"] != entry.unit or any(set(series["dimensions"]) != set(entry.label_names)
                    for series in summary["series"]):
                raise DiagnosticSourceAdapterError("SOURCE_METADATA_CHANGED", "OEM 历史基线元数据已变化")
            result = MonitoringNativeBaseline(period=period, start=window.start, end=window.end,
                response_sha256=summary.get("response_sha256"),
                series=baseline_series(summary), truncated=summary["truncated"],
                unavailable_reason="历史结果已截断，不生成不完整基线。" if summary["truncated"] else
                    None if summary["series"] else "基线窗口内暂无有效采样。")
        except (DiagnosticSourceAdapterError, ClientError, TimeoutError, LookupError) as exc:
            return MonitoringNativeBaseline(period=period, start=window.start, end=window.end,
                unavailable_reason={
                    "SOURCE_RESULT_TRUNCATED": "OEM 完整基线超过本次剩余读取预算；不计算部分均值，当前趋势保留。",
                    "SOURCE_HISTORY_WINDOW_UNSUPPORTED": "原厂 EMCLI 历史最多支持最近7天；不能用短窗口冒充所选基线。",
                }.get(getattr(exc, "code", ""), "OEM 历史基线读取失败；当前采样不受影响，可重试。"))
        if len(self._baseline_cache) >= 64:
            self._baseline_cache.clear()
        self._baseline_cache[key] = (now + timedelta(minutes=5), frozen, result)
        return result

    @staticmethod
    def _response(result, frozen):
        if len(result.model_dump_json().encode("utf-8")) > frozen["max_response_bytes"]:
            raise AIOpsApplicationError(code="MONITORING_QUERY_BUDGET_EXCEEDED", message="OEM 标准化结果超出单次响应字节预算", status_code=422)
        return result

    @staticmethod
    def _raise_provider(exc):
        code = getattr(exc, "code", "SOURCE_UNREACHABLE" if isinstance(exc, (ClientError, TimeoutError)) else "SOURCE_CAPABILITY_FORBIDDEN")
        message = {
            "SOURCE_AUTH_FAILED": "OEM 拒绝指标读取，请检查来源账号权限；这不代表被监控资源宕机",
            "SOURCE_UNREACHABLE": "OEM 连接失败或超时，无法取得本次采样；请检查服务连接",
            "SOURCE_PROVIDER_ERROR": "OEM 服务端查询失败（HTTP 5xx）；不是网络不可达或无数据，请检查 OEM 服务日志",
            "SOURCE_RESULT_TRUNCATED": "OEM 响应超出读取字节预算，无法将本次结果视为完整采样",
            "SOURCE_RATE_LIMITED": "OEM 指标读取触发限流，请稍后重试",
            "SOURCE_COLLECTION_CHANGED": "OEM 最近采集快照在分页期间发生变化，请重新读取",
        }.get(code, "OEM 原生指标读取失败，请检查来源权限、插件、采样窗口或响应预算")
        raise AIOpsApplicationError(code=code, message=message, status_code=502,
            retryable=getattr(exc, "retryable", isinstance(exc, (ClientError, TimeoutError)))) from exc
