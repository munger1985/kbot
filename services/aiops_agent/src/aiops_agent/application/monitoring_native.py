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
from aiops_agent.application.monitoring_oem_overview import baseline_series, overview_panels
from aiops_agent.monitoring import resolve_monitoring_baseline_window, resolve_monitoring_window
from aiops_agent.ports.diagnostic_source import (
    DiagnosticSourceContext, MetricsDiscoveryRequest, NativeMetricQueryRequest,
)
from platform_core.contracts.aiops.monitoring_native import (
    MonitoringNativeBaseline, MonitoringNativeCatalog, MonitoringNativeMetric, MonitoringNativePoint, MonitoringNativeSamples,
)


def public_metric(entry):
    return MonitoringNativeMetric(
        name=entry.name, display_name=entry.help_text or entry.name,
        group=unquote(entry.name.split("/", 1)[0]),
        native_value_type=entry.native_value_type or "UNKNOWN", unit=entry.unit,
        dimensions=entry.label_names, query_supported=entry.query_supported,
        latest_supported=entry.native_value_type in {"NUMBER", "STRING"},
    )


class NativeMonitoringService:
    """复用监控应用依赖；不写业务配置，也不向浏览器开放 Provider 查询。"""

    def __init__(self, monitoring):
        self._monitoring = monitoring
        self._catalog_cache = {}
        self._baseline_cache = {}
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
                if resource.resource_type_key not in {"DATABASE", "PHYSICAL_SERVER", "VIRTUAL_MACHINE", "CONTAINER"}:
                    continue
                source = await self._monitoring._source(uow, scope.domain_id, source_id)
                if source.source_type != "OEM" or getattr(source, "tenant_id", None) != getattr(scope, "tenant_id", None):
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
                    "scope": "DATABASE" if resource.resource_type_key == "DATABASE" else "HOST",
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
            source_id=source["source_id"], source_type="OEM", adapter_id=source["adapter_id"],
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
        try:
            entries, truncated, revision = await self._catalog(scope, source_id, instance_id, frozen)
        except (DiagnosticSourceAdapterError, ClientError, TimeoutError, LookupError) as exc:
            self._raise_provider(exc)
        await self._unchanged(scope, source_id, instance_id, frozen)
        metrics = sorted((public_metric(entry) for entry in entries), key=lambda entry: entry.name)
        query = search.strip().casefold()
        filtered = [entry for entry in metrics if not query or query in f"{entry.name} {entry.display_name} {entry.group}".casefold()]
        default_metric = next((entry for preferred in self._defaults[frozen["scope"]] for entry in metrics
            if entry.name == preferred["name"] and entry.unit == preferred["unit"]
            and entry.native_value_type == preferred["native_value_type"] and entry.query_supported), None)
        return self._response(MonitoringNativeCatalog(
            source_id=source_id, instance_id=instance_id, entries=tuple(filtered[offset:offset+limit]),
            default_metric=default_metric,
            discovered_count=len(filtered), next_offset=offset+limit if offset+limit < len(filtered) else None,
            truncated=truncated, revision=revision,
            warnings=("目录来自该资源实际安装的 OEM 插件；目录可见不代表窗口内已有采样。",
                      "数值指标支持历史时序；数值与文本指标另可读取最近采集快照，快照不代表历史覆盖。"),
            overview_panels=overview_panels(frozen["scope"], metrics),
        ), frozen)

    async def samples(self, *, scope, source_id, instance_id, query):
        frozen = await self._binding(scope, source_id, instance_id)
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
                trace_id=scope.trace_id, max_response_bytes=frozen["max_response_bytes"] // (2 if query.baseline else 1))
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
                return self._response(MonitoringNativeSamples(
                    source_id=source_id, instance_id=instance_id, metric=public_metric(entry),
                    window_start=None, window_end=None, captured_at=latest["captured_at"],
                    points=tuple(MonitoringNativePoint(observed_at=when, value=value, dimensions=labels)
                        for (when, value), labels in zip(latest["points"], latest["dimensions"], strict=True)),
                    truncated=latest["truncated"], response_sha256=latest["response_sha256"],
                    query_mode="LATEST", history_resolution="LATEST_SNAPSHOT",
                    baseline=baseline,
                    warnings=("这是 OEM 最近一次采集快照，不受所选历史时间窗约束；请核对采集时间，不视为实时或完整历史。",
                        "原始文本、空值及单位不转换为健康结论，不以快照替代历史查询。")), frozen)
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
        return self._response(MonitoringNativeSamples(
            source_id=source_id, instance_id=instance_id, metric=public_metric(entry),
            window_start=window.start, window_end=window.end, captured_at=samples.captured_at,
            points=tuple(MonitoringNativePoint(observed_at=when, value=value, dimensions=dimensions)
                for (when, value), dimensions in zip(samples.points, samples.sample_dimensions, strict=True)),
            truncated=samples.truncated, response_sha256=samples.response_sha256,
            baseline=baseline,
            warnings=("OEM 自动选择原始、小时或日聚合；采样覆盖率未知，不推断 counter/gauge 或自动换算速率。",
                      "窗口无采样表示暂无数据，不代表资源宕机；缺失值不补零。"),
        ), frozen)

    async def _baseline(self, adapter, request, frozen, entry, period, deadline):
        if period is None or entry.native_value_type != "NUMBER":
            return None
        now = datetime.now(UTC)
        window = resolve_monitoring_baseline_window(period, now=now)
        key = (str(request.target_id), str(frozen["binding_id"]), entry.name, period)
        cached = self._baseline_cache.get(key)
        if cached and cached[0] > now and cached[1] == frozen:
            return cached[2]
        try:
            async with asyncio.timeout(max(0, deadline - asyncio.get_running_loop().time())):
                summary = await adapter.query_native_metric(
                    request.model_copy(update={"window_start": window.start, "window_end": window.end}),
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
        except (DiagnosticSourceAdapterError, ClientError, TimeoutError, LookupError):
            return MonitoringNativeBaseline(period=period, start=window.start, end=window.end,
                unavailable_reason="OEM 历史基线读取失败；当前采样不受影响，可重试。")
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
            "SOURCE_UNREACHABLE": "OEM 服务返回错误或暂时不可达，无法取得本次采样；可重试或检查 OEM 服务日志",
            "SOURCE_RESULT_TRUNCATED": "OEM 响应超出读取字节预算，无法将本次结果视为完整采样",
            "SOURCE_RATE_LIMITED": "OEM 指标读取触发限流，请稍后重试",
            "SOURCE_COLLECTION_CHANGED": "OEM 最近采集快照在分页期间发生变化，请重新读取",
        }.get(code, "OEM 原生指标读取失败，请检查来源权限、插件、采样窗口或响应预算")
        raise AIOpsApplicationError(code=code, message=message, status_code=502,
            retryable=getattr(exc, "retryable", isinstance(exc, (ClientError, TimeoutError)))) from exc
