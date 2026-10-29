"""OEM 13.5 官方指标目录与历史接口；只读取已冻结 Target GUID。

目录与数值类型不证明 counter/gauge；metricTimeSeries 的留存层自动聚合
必须沿证据链保留。只做有界读取，不跟随响应 URL，不伪造分页。
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import re
from urllib.parse import quote, unquote

import aiohttp

from aiops_agent.contracts.evidence import DiscoveredMetric, MetricDiscoveryResult, NativeMetricSamples
from aiops_agent.ports.diagnostic_source import CAPABILITY_METRIC_DISCOVERY, CAPABILITY_METRIC_QUERY_RANGE
from .base import DiagnosticSourceAdapterError


def _fail(code="SOURCE_RESPONSE_INVALID"):
    raise DiagnosticSourceAdapterError(code, "OEM 原生指标的绑定、元数据或采样无法核验")


def _name(value):
    if (not isinstance(value, str) or not value or len(value) > 64
        or value in {".", ".."} or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        _fail()
    return value


def metric_selector(group, metric):
    return quote(_name(group), safe="") + "/" + quote(_name(metric), safe="")


def parse_metric_selector(selector):
    if not isinstance(selector, str) or len(selector) > 256:
        _fail("SOURCE_QUERY_INVALID")
    parts = selector.split("/")
    if len(parts) != 2:
        _fail("SOURCE_QUERY_INVALID")
    group, metric = (_name(unquote(part)) for part in parts)
    if metric_selector(group, metric) != selector:
        _fail("SOURCE_QUERY_INVALID")
    return group, metric


def _hash(*digests):
    return hashlib.sha256(":".join(digests).encode()).hexdigest()


def _collection(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        _fail()
    items = payload["items"]
    count = payload.get("count")
    if (isinstance(count, bool) or not isinstance(count, (str, int))
        or not str(count).isdecimal() or int(count) != len(items)
        or any(not isinstance(item, dict) for item in items)):
        _fail()
    links = payload.get("links", {})
    if not isinstance(links, dict):
        _fail()
    # 只识别是否还存在后页；绝不执行来自响应的地址。
    more = links.get("next") is not None
    return items, more


class OEMNativeMetrics:
    """复用 Adapter 网络及总字节预算，模型不能控制地址、Target 或认证。"""

    async def health_check(self, request):
        self._remaining_response_bytes = 65536
        try:
            payload, _ = await self._native_get("/targets", params={"limit": 1}, max_bytes=65536)
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                _fail()
            return self._health_result(healthy=True)
        except (DiagnosticSourceAdapterError, aiohttp.ClientError, TimeoutError) as exc:
            return self._health_result(healthy=False, error_code=getattr(exc, "code", "SOURCE_UNREACHABLE"))

    def _native_api(self):
        endpoint = self._endpoint().rstrip("/")
        if endpoint.endswith("/em/api"):
            return endpoint
        if endpoint.endswith("/em"):
            return endpoint + "/api"
        _fail("SOURCE_ENDPOINT_INVALID")

    async def _native_get(self, suffix, *, params=None, max_bytes):
        credentials = self.context.credentials
        username, password = credentials.get("username"), credentials.get("password")
        if bool(username) != bool(password):
            _fail("SOURCE_AUTH_FAILED")
        auth = aiohttp.BasicAuth(username, password) if username and password else None
        headers = {} if auth is not None else self._headers()
        async with self._session.get(
            self._native_api() + suffix, headers=headers, auth=auth, params=params,
            timeout=self._timeout, allow_redirects=False,
        ) as response:
            if response.status not in {200, 401, 403, 429} and response.status < 500:
                _fail("SOURCE_QUERY_UNSUPPORTED" if response.status == 404 else "SOURCE_RESPONSE_INVALID")
            return await self._response_json(response, max_bytes=max_bytes)

    async def _native_target(self, request):
        locator = request.source_locator_key
        if not isinstance(locator, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", locator):
            _fail("SOURCE_TARGET_LOCATOR_INVALID")
        target, digest = await self._native_get("/targets/" + locator, max_bytes=request.max_response_bytes // 4)
        if (not isinstance(target, dict) or not isinstance(target.get("targetName"), str)
            or not 0 < len(target["targetName"]) <= 256
            or not isinstance(target.get("targetType"), str) or not 0 < len(target["targetType"]) <= 64
            or ("targetId" in target and str(target["targetId"]).casefold() != locator.casefold())):
            _fail("SOURCE_TARGET_MISMATCH")
        return target, digest

    async def discover_metrics(self, request):
        if CAPABILITY_METRIC_DISCOVERY not in self.context.declared_capabilities:
            _fail("SOURCE_CAPABILITY_FORBIDDEN")
        if request.scope != "DATABASE":
            _fail("SOURCE_QUERY_UNSUPPORTED")
        self._remaining_response_bytes = request.max_response_bytes
        _, target_hash = await self._native_target(request)
        payload, digest = await self._native_get(
            "/targets/" + request.source_locator_key + "/metricGroups",
            params={"include": "metrics"}, max_bytes=request.max_response_bytes * 3 // 4,
        )
        groups, more = _collection(payload)
        entries, seen = [], set()
        for group in groups:
            group_name = _name(group.get("name"))
            keys = group.get("keys", [])
            metrics = group.get("metrics")
            if (not isinstance(keys, list) or len(keys) > 16 or not isinstance(metrics, list)
                or any(not isinstance(key, dict) for key in keys)):
                _fail()
            key_names = tuple(_name(key.get("name")) for key in keys)
            if len(set(key_names)) != len(key_names):
                _fail()
            for metric in metrics:
                if not isinstance(metric, dict) or not isinstance(metric.get("dataType"), str):
                    _fail()
                selector = metric_selector(group_name, metric.get("name"))
                if selector in seen:
                    _fail("SOURCE_ITEM_AMBIGUOUS")
                seen.add(selector)
                unit = metric.get("unitDisplayName")
                if unit is not None and (not isinstance(unit, str) or len(unit) > 64):
                    _fail()
                if len(entries) < request.limit and len(selector) <= 256:
                    entries.append(DiscoveredMetric(
                        name=selector, label_names=key_names, metric_type="UNKNOWN",
                        native_value_type=metric["dataType"][:64], unit=unit,
                        help_text=str(metric.get("displayName") or selector)[:2000],
                        metadata_state="VERIFIED", query_supported=metric["dataType"] == "NUMBER",
                    ))
        return MetricDiscoveryResult(
            action_id=request.action_id, target_id=request.target_id, binding_id=request.binding_id,
            source_id=self.context.source_id, scope=request.scope,
            window_start=request.window_start, window_end=request.window_end, captured_at=datetime.now(UTC),
            entries=tuple(entries), truncated=more or len(seen) > len(entries),
            response_sha256=_hash(target_hash, digest),
            warnings=("OEM 原生组/指标目录不证明窗口采样覆盖或计数器语义；精确名称为编码后的组/指标。",
                "OEM 按实际留存自动选择原始、小时或日均值；不能据请求窗口宣称原始执行明细或精确窗口增量。"),
        )

    async def query_native_metric(self, request, *, metric_name, limit, allow_text=False,
                                  _expected_target_type=None, _shared_budget=False):
        if CAPABILITY_METRIC_QUERY_RANGE not in self.context.declared_capabilities:
            _fail("SOURCE_CAPABILITY_FORBIDDEN")
        group, metric = parse_metric_selector(metric_name)
        if type(limit) is not int or not 1 <= limit <= 1000:
            _fail("SOURCE_QUERY_INVALID")
        if request.pagination is not None:
            _fail("SOURCE_QUERY_UNSUPPORTED")
        if not _shared_budget:
            self._remaining_response_bytes = request.max_response_bytes
        target, target_hash = await self._native_target(request)
        if _expected_target_type is not None and target["targetType"] != _expected_target_type:
            _fail("SOURCE_TARGET_MISMATCH")
        metadata, metadata_hash = await self._native_get(
            "/targets/" + request.source_locator_key + "/metricGroups/" + quote(group, safe=""),
            max_bytes=request.max_response_bytes // 4,
        )
        if (not isinstance(metadata, dict) or metadata.get("name") != group
            or not isinstance(metadata.get("metrics"), list) or not isinstance(metadata.get("keys", []), list)):
            _fail("SOURCE_TARGET_MISMATCH")
        matches = [item for item in metadata["metrics"] if isinstance(item, dict) and item.get("name") == metric]
        if len(matches) != 1:
            _fail("SOURCE_ITEM_AMBIGUOUS")
        item = matches[0]
        if item.get("dataType") != "NUMBER":
            _fail("SOURCE_METRIC_TYPE_MISMATCH")
        keys = metadata.get("keys", [])
        if len(keys) > 16 or any(not isinstance(key, dict) for key in keys):
            _fail()
        key_names = tuple(_name(key.get("name")) for key in keys)
        if len(set(key_names)) != len(key_names):
            _fail()
        unit = item.get("unitDisplayName")
        if unit is not None and (not isinstance(unit, str) or len(unit) > 64):
            _fail()
        lower = request.window_start - timedelta(microseconds=request.window_start.microsecond % 1000)
        upper = request.window_end + timedelta(microseconds=(-request.window_end.microsecond) % 1000)
        def stamp(value):
            return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        payload, response_hash = await self._native_get("/metricTimeSeries", params={
            "targetId": request.source_locator_key, "metricGroupName": group, "metricName": metric,
            "timeCollectedGreaterThanOrEqualTo": stamp(lower),
            "timeCollectedLessThan": stamp(upper), "limit": 20,
        }, max_bytes=request.max_response_bytes // 2)
        series, more = _collection(payload)
        if len(series) > 20:
            _fail()
        points, dimensions, identities = [], [], set()
        total = 0
        for native in series:
            if (native.get("targetName") != target["targetName"] or native.get("targetTypeName") != target["targetType"]
                or native.get("metricGroupName") != group or native.get("metricName") != metric
                or ("targetId" in native and str(native["targetId"]).casefold() != request.source_locator_key.casefold())):
                _fail("SOURCE_TARGET_MISMATCH")
            values = native.get("keyValues", [])
            if not isinstance(values, list) or len(values) != len(key_names):
                _fail()
            labels = {}
            for key in values:
                if (not isinstance(key, dict) or key.get("name") not in key_names or key["name"] in labels
                    or not isinstance(key.get("value"), str) or len(key["value"]) > 256):
                    _fail()
                labels[key["name"]] = key["value"]
            identity = tuple(sorted(labels.items()))
            if identity in identities:
                _fail("SOURCE_ITEM_AMBIGUOUS")
            identities.add(identity)
            samples = native.get("datapoints")
            if not isinstance(samples, list):
                _fail()
            previous = None
            for sample in samples:
                try:
                    if not isinstance(sample, list) or len(sample) != 2 or not isinstance(sample[0], str):
                        raise ValueError
                    when = datetime.fromisoformat(sample[0].replace("Z", "+00:00"))
                    value = sample[1]
                    if type(value) is int:
                        value = str(value)
                    if (not isinstance(value, str) or not value or len(value) > 256 or when.tzinfo is None
                        or not lower <= when < upper
                        or (previous is not None and when <= previous)):
                        raise ValueError
                    numeric = Decimal(value)
                    if not numeric.is_finite() or abs(numeric.as_tuple().exponent) > 1024:
                        raise ValueError
                except (ValueError, TypeError, InvalidOperation, OverflowError):
                    _fail()
                previous = when
                # 官方接口以毫秒为界；扩大至相邻毫秒后仍按冻结的精确窗口投影。
                if not request.window_start <= when < request.window_end:
                    continue
                total += 1
                if len(points) < limit:
                    points.append((when.astimezone(UTC), value))
                    dimensions.append(labels.copy())
        return NativeMetricSamples(
            action_id=request.action_id, target_id=request.target_id, binding_id=request.binding_id,
            source_id=self.context.source_id, source_type="OEM", metric_name=metric_name,
            native_value_type="NUMBER", unit=unit, history_resolution="PROVIDER_MANAGED",
            sample_dimensions=tuple(dimensions), source_version=f"{self.context.adapter_version}:{self.context.config_version}",
            window_start=request.window_start, window_end=request.window_end, captured_at=datetime.now(UTC),
            points=tuple(points), truncated=more or total > limit,
            response_sha256=_hash(target_hash, metadata_hash, response_hash),
        )
