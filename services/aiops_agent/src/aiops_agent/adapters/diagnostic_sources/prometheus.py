"""Prometheus 指标和活动规则证据 Adapter。"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

import aiohttp
from loguru import logger

from aiops_agent.ports.diagnostic_source import (
    EventEvidenceResult,
    EventEvidenceRequest,
    HostDiscoveryCandidate,
    InstanceDiscoveryCandidate,
    InstanceDiscoveryRequest,
    InstanceDiscoveryResult,
    MetricsEvidenceRequest,
    MetricsDiscoveryRequest,
    MetricsEvidenceResult,
    SourceHealthRequest,
    SourceHealthResult,
)

from aiops_agent.contracts.evidence import DiscoveredMetric, MetricDiscoveryResult, SourceEventQuery

from .base import BaseDiagnosticSourceAdapter, DiagnosticSourceAdapterError


def _escape_prometheus_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class PrometheusAdapter(BaseDiagnosticSourceAdapter):
    _minimum_complete_coverage_ratio = 0.8
    _DISCOVERY_SERIES = {
        "ORACLE": ("oracledb_up", "oracle", "target_key"),
        "MYSQL": ("mysql_up", "mysql", "target_key"),
        "POSTGRESQL": ("pg_up", "postgres", "target_key"),
    }

    async def discover_metrics(self, request: MetricsDiscoveryRequest) -> MetricDiscoveryResult:
        """只访问绑定目标的系列目录，字节与结果规模均有界。"""
        self._remaining_response_bytes = request.max_response_bytes
        if request.scope == "HOST":
            if not request.host_target_key:
                raise DiagnosticSourceAdapterError("SOURCE_TARGET_MAPPING_MISSING", "未绑定主机，不能发现其他主机指标")
            locators = (("target_key", request.host_target_key),)
        else:
            locators = (("instance", request.source_locator_key), ("target_key", request.source_locator_key))
        selectors = ['{' + label + '="' + _escape_prometheus_label(locator) + '"}' for label, locator in locators]
        async with self._session.get(
            f"{self._endpoint().rstrip('/')}/api/v1/series",
            headers=self._headers(), params={
                "match[]": selectors, "start": request.window_start.timestamp(),
                "end": request.window_end.timestamp(), "limit": 1001,
            }, timeout=self._timeout,
        ) as response:
            payload, digest = await self._response_json(response, max_bytes=request.max_response_bytes // 2)
        if not isinstance(payload, dict) or payload.get("status") != "success" or not isinstance(payload.get("data"), list):
            raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "指标目录响应格式无效")
        names: dict[str, set[str]] = {}
        raw = payload["data"]
        if len(raw) > 1000:
            raw = raw[:1000]
        for series in raw:
            if not isinstance(series, dict) or not any(series.get(label) == locator for label, locator in locators):
                raise DiagnosticSourceAdapterError("SOURCE_TARGET_MISMATCH", "指标目录返回了绑定范围以外的系列")
            name = series.get("__name__")
            if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]{0,255}", name):
                continue
            labels = {key for key in series if key != "__name__" and isinstance(key, str) and len(key) <= 128}
            names.setdefault(name, set()).update(labels)
        truncated = (
            len(payload["data"]) >= 1000 or len(names) > request.limit
            or any(len(labels) > 128 for labels in names.values())
            or bool(payload.get("warnings"))
        )
        metadata, metadata_warnings, metadata_hashes = await self._target_metric_metadata(
            locators=locators, names=set(sorted(names)[:request.limit]),
            max_bytes=request.max_response_bytes // 2,
        )
        return MetricDiscoveryResult(
            action_id=request.action_id, target_id=request.target_id, binding_id=request.binding_id,
            source_id=self.context.source_id, scope=request.scope,
            window_start=request.window_start, window_end=request.window_end,
            captured_at=datetime.now(UTC), response_sha256=hashlib.sha256(
                ":".join((digest, *metadata_hashes)).encode("ascii")
            ).hexdigest(),
            entries=tuple(DiscoveredMetric(name=name, label_names=tuple(sorted(names[name])[:128]), **metadata.get(name, {}))
                          for name in sorted(names)[:request.limit]),
            truncated=truncated,
            warnings=("系列目录只能说明请求窗口曾可见；不能证明当前有采样。元数据不证明数值或历史覆盖。",
                      *metadata_warnings,
                      *(("目录达到上限或源返回警告，不代表完整指标清单。",) if truncated else ())),
        )

    async def _target_metric_metadata(self, *, locators, names, max_bytes):
        """按绑定目标补元数据；不读取全局 metadata，三次请求合计预算不超过目录限额。"""
        metadata = {}
        warnings = []
        response_hashes = []
        for label, locator in locators:
            try:
                async with self._session.get(
                    f"{self._endpoint().rstrip('/')}/api/v1/targets/metadata",
                    headers=self._headers(),
                    params={"match_target": '{' + label + '="' + _escape_prometheus_label(locator) + '"}', "limit": 1001},
                    timeout=self._timeout,
                ) as response:
                    payload, response_hash = await self._response_json(response, max_bytes=max_bytes // len(locators))
                response_hashes.append(response_hash)
                if not isinstance(payload, dict) or payload.get("status") != "success" or not isinstance(payload.get("data"), list):
                    raise ValueError("元数据格式无效")
                if len(payload["data"]) > 1000:
                    warnings.append("目标元数据达到上限，未返回条目仍标未知。")
                rows = payload["data"][:1000]
                if any(
                    not isinstance(item, dict) or not isinstance(item.get("target"), dict)
                    or item["target"].get(label) != locator for item in rows
                ):
                    raise ValueError("元数据超出绑定目标范围")
                for item in rows:
                    name = item.get("metric")
                    if name not in names:
                        continue
                    kind = str(item.get("type") or "UNKNOWN").upper()
                    if kind not in {"COUNTER", "GAUGE", "HISTOGRAM", "SUMMARY", "INFO", "STATESET", "UNTYPED"}:
                        kind = "UNKNOWN"
                    entry = {
                        "metric_type": kind, "help_text": str(item.get("help") or "")[:2000] or None,
                        "unit": str(item.get("unit") or "")[:64] or None,
                        "metadata_state": "VERIFIED" if kind != "UNKNOWN" else "UNKNOWN",
                    }
                    if name in metadata and metadata[name] != entry:
                        metadata[name] = {"metric_type": "UNKNOWN", "metadata_state": "CONFLICT"}
                        warnings.append("同名指标元数据冲突，未选择其中一种类型。")
                    else:
                        metadata[name] = entry
            except (DiagnosticSourceAdapterError, ValueError, TypeError):
                warnings.append("目标元数据未核验；缺失类型、单位和帮助信息保持未知。")
        return metadata, tuple(dict.fromkeys(warnings)), tuple(response_hashes)

    async def discover_instances(
        self, request: InstanceDiscoveryRequest
    ) -> InstanceDiscoveryResult:
        db_types = request.db_types or tuple(self._DISCOVERY_SERIES)
        params: list[tuple[str, str | float]] = []
        for db_type in db_types:
            metric_name, job, _label = self._DISCOVERY_SERIES[db_type]
            params.append(("match[]", f'{metric_name}{{job="{job}"}}'))
        if request.include_hosts or request.host_locator_keys:
            params.append(("match[]", 'up{job="node"}'))
        now = datetime.now(UTC).timestamp()
        params.extend((("start", now - 3600), ("end", now)))
        async with self._session.get(
            f"{self._endpoint().rstrip('/')}/api/v1/series",
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        ) as response:
            payload, _ = await self._response_json(
                response, max_bytes=2 * 1024 * 1024
            )
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID",
                "Prometheus 实例目录响应无效",
            )
        series = payload.get("data")
        if not isinstance(series, list):
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID",
                "Prometheus 实例目录响应无效",
            )
        requested_keys = set(request.locator_keys)
        requested_host_keys = set(request.host_locator_keys)
        candidates: dict[tuple[str, str], InstanceDiscoveryCandidate] = {}
        host_candidates: dict[str, HostDiscoveryCandidate] = {}
        metric_types = {
            config[0]: (db_type, config[2])
            for db_type, config in self._DISCOVERY_SERIES.items()
            if db_type in db_types
        }
        for item in series:
            if not isinstance(item, dict):
                continue
            if (
                str(item.get("__name__") or "") == "up"
                and str(item.get("job") or "") == "node"
            ):
                host_key = str(item.get("target_key") or "").strip()
                if (
                    host_key
                    and len(host_key) <= 512
                    and (
                        not requested_host_keys
                        or host_key in requested_host_keys
                    )
                ):
                    host_candidates[host_key] = HostDiscoveryCandidate(
                        source_locator_key=host_key,
                        display_name="Node Exporter 主机",
                    )
                continue
            metric_type = metric_types.get(str(item.get("__name__") or ""))
            if metric_type is None:
                continue
            db_type, locator_label = metric_type
            locator_key = str(item.get(locator_label) or "").strip()
            if (
                not locator_key
                or len(locator_key) > 512
                or (requested_keys and locator_key not in requested_keys)
                or (
                    request.after_locator_key is not None
                    and locator_key <= request.after_locator_key
                )
            ):
                continue
            candidates[(db_type, locator_key)] = InstanceDiscoveryCandidate(
                source_locator_key=locator_key,
                source_locator={
                    "target_key": locator_key,
                    "instance": locator_key,
                    "database_type": db_type,
                },
                db_type=db_type,
                display_name=f"{db_type.title()} 数据库实例",
            )
        ordered = sorted(
            candidates.values(),
            key=lambda item: (item.source_locator_key, item.db_type),
        )
        return InstanceDiscoveryResult(
            candidates=tuple(ordered[: request.limit]),
            host_candidates=tuple(
                host_candidates[key]
                for key in sorted(host_candidates)[: request.limit]
            ),
        )

    async def health_check(
        self, request: SourceHealthRequest
    ) -> SourceHealthResult:
        """验证 Prometheus 查询 API，避免把 Exporter 误当成 Server。"""
        try:
            async with self._session.get(
                (
                    f"{self._endpoint().rstrip('/')}"
                    "/api/v1/status/buildinfo"
                ),
                headers=self._headers(),
                timeout=self._timeout,
            ) as response:
                if response.status in {401, 403}:
                    return self._health_result(
                        healthy=False, error_code="SOURCE_AUTH_FAILED"
                    )
                if response.status != 200:
                    return self._health_result(
                        healthy=False,
                        error_code="SOURCE_API_UNAVAILABLE",
                    )
                payload = await response.json()
                healthy = (
                    isinstance(payload, dict)
                    and payload.get("status") == "success"
                    and isinstance(payload.get("data"), dict)
                )
                return self._health_result(
                    healthy=healthy,
                    error_code=(
                        None if healthy else "SOURCE_RESPONSE_INVALID"
                    ),
                )
        except (
            aiohttp.ClientError,
            TimeoutError,
            ValueError,
            TypeError,
        ):
            return self._health_result(
                healthy=False, error_code="SOURCE_UNREACHABLE"
            )

    async def query_events(
        self, request: EventEvidenceRequest
    ) -> EventEvidenceResult:
        self._remaining_response_bytes = request.max_response_bytes
        try:
            async with self._session.get(
                f"{self._endpoint().rstrip('/')}/api/v1/alerts",
                headers=self._headers(),
                timeout=self._timeout,
                allow_redirects=False,
            ) as response:
                if response.status != 200 and response.status not in {401, 403, 429} and response.status < 500:
                    raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "活动告警接口未返回成功状态")
                payload, response_hash = await self._response_json(
                    response, max_bytes=request.max_response_bytes
                )
            alerts = []
            if (not isinstance(payload, dict) or payload.get("status") != "success"
                or not isinstance(payload.get("data"), dict)
                or not isinstance(payload["data"].get("alerts"), list)):
                raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "活动告警正文不符合来源合同")
            for item in payload.get("data", {}).get("alerts", []):
                if not isinstance(item, dict) or not isinstance(item.get("labels"), dict):
                    raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "活动告警标签无法核验")
                labels = item.get("labels") or {}
                if labels.get("instance") != request.source_locator_key:
                    continue
                alerts.append(
                    {
                        "source_type": "PROMETHEUS",
                        "name": str(
                            labels.get("alertname", "prometheus.alert")
                        )[:128],
                        "severity": str(
                            labels.get("severity", "warning")
                        ).upper(),
                        "status": str(item.get("state", "active")).upper(),
                        "active_at": item.get("activeAt"),
                    }
                )
                if len(alerts) >= request.max_events:
                    break
            return EventEvidenceResult(events=tuple(alerts), query=SourceEventQuery(
                source_type="PROMETHEUS", source_id=self.context.source_id,
                target_id=request.target_id, binding_id=request.binding_id, query_scope="CURRENT_ALERTS",
                window_start=request.window_start, window_end=request.window_end,
                captured_at=datetime.now(UTC), response_sha256=response_hash,
                source_version=f"{self.context.adapter_id}:{self.context.adapter_version}:{self.context.config_version}",
                truncated=len(alerts) >= request.max_events,
            ))
        except (
            DiagnosticSourceAdapterError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            code = (
                exc.code
                if isinstance(exc, DiagnosticSourceAdapterError)
                else "SOURCE_RESPONSE_INVALID"
            )
            return EventEvidenceResult(
                gaps=(
                    self._gap(
                        request,  # type: ignore[arg-type]
                        metric_code=None,
                        code=code,
                        detail="Prometheus 活动告警读取失败",
                        retryable=(
                            exc.retryable
                            if isinstance(exc, DiagnosticSourceAdapterError)
                            else False
                        ),
                    ),
                )
            )

    async def query_metrics(
        self, request: MetricsEvidenceRequest
    ) -> MetricsEvidenceResult:
        observations = []
        gaps = []
        source_locator_key = request.source_locator_key
        locator_fingerprint = hashlib.sha256(
            source_locator_key.encode("utf-8")
        ).hexdigest()[:16]
        host_target_key = request.source_locator.get("host_target_key") or (
            "__unbound_host_target__"
        )
        for definition in request.metric_definitions:
            provider = definition.providers.get("PROMETHEUS")
            if provider is None or provider.query_template is None:
                gaps.append(
                    self._gap(
                        request,
                        metric_code=definition.metric_code,
                        code="SOURCE_QUERY_UNSUPPORTED",
                        detail="Prometheus 未定义该指标",
                    )
                )
                continue
            duration = int(
                (request.window_end - request.window_start).total_seconds()
            )
            step = max(
                request.requested_step_seconds,
                definition.min_step_seconds,
                max(1, duration // definition.max_points),
            )
            query_templates = (
                provider.query_template,
                *provider.fallback_query_templates,
            )
            best_observation = None
            query_errors: list[DiagnosticSourceAdapterError] = []
            for query_index, query_template in enumerate(query_templates):
                query_variant = (
                    "primary"
                    if query_index == 0
                    else f"fallback:{query_index}"
                )
                query = query_template.replace(
                    "${external_target}",
                    _escape_prometheus_label(source_locator_key),
                ).replace(
                    "${host_target}",
                    _escape_prometheus_label(host_target_key),
                )
                try:
                    raw_series, response_hash = await self._query_range(
                        request=request,
                        query=query,
                        step=step,
                    )
                except DiagnosticSourceAdapterError as exc:
                    query_errors.append(exc)
                    logger.warning(
                        "Prometheus 指标查询失败 | trace_id={} | metric_code={} | "
                        "query_variant={} | locator_fingerprint={} | "
                        "window_start={} | window_end={} | step={} | error_code={}",
                        request.trace_id,
                        definition.metric_code,
                        query_variant,
                        locator_fingerprint,
                        request.window_start.isoformat(),
                        request.window_end.isoformat(),
                        step,
                        exc.code,
                    )
                    continue
                log_result = logger.debug if raw_series else logger.info
                log_result(
                    "Prometheus 指标查询完成 | trace_id={} | metric_code={} | "
                    "query_variant={} | locator_fingerprint={} | "
                    "window_start={} | window_end={} | step={} | "
                    "series_count={} | sample_count={}",
                    request.trace_id,
                    definition.metric_code,
                    query_variant,
                    locator_fingerprint,
                    request.window_start.isoformat(),
                    request.window_end.isoformat(),
                    step,
                    len(raw_series),
                    sum(len(points) for _, points in raw_series),
                )
                if not raw_series:
                    continue
                fallback_used = query_index > 0
                observation = self._observation(
                    request=request,
                    definition=definition,
                    raw_series=raw_series,
                    provider_response_hash=response_hash,
                    effective_step=step,
                    truncated=len(raw_series) > definition.max_series,
                    warnings=(
                        (
                            "记录指标不可用或历史覆盖不足，已采用原始指标回退查询",
                        )
                        if fallback_used
                        else ()
                    ),
                    provenance={
                        "query_variant": query_variant,
                        "fallback_used": fallback_used,
                    },
                )
                if (
                    best_observation is None
                    or observation.coverage_ratio
                    > best_observation.coverage_ratio
                ):
                    best_observation = observation
                if (
                    observation.coverage_ratio
                    >= self._minimum_complete_coverage_ratio
                ):
                    break
            if best_observation is not None:
                observations.append(best_observation)
                continue
            if query_errors:
                exc = query_errors[-1]
                gaps.append(
                    self._gap(
                        request,
                        metric_code=definition.metric_code,
                        code=exc.code,
                        detail=str(exc),
                        retryable=exc.retryable,
                    )
                )
                continue
            gaps.append(
                self._gap(
                    request,
                    metric_code=definition.metric_code,
                    code="SOURCE_NO_DATA",
                    detail=f"指标“{definition.name}”的 Prometheus 查询未返回任何采样",
                )
            )
        return MetricsEvidenceResult(
            observations=tuple(observations), gaps=tuple(gaps)
        )

    async def _query_range(
        self,
        *,
        request: MetricsEvidenceRequest,
        query: str,
        step: int,
    ) -> tuple[
        list[tuple[dict[str, str], list[tuple[datetime, object]]]], str
    ]:
        """执行一次受控范围查询并归一 Prometheus 矩阵结果。"""
        async with self._session.get(
            f"{self._endpoint().rstrip('/')}/api/v1/query_range",
            headers=self._headers(),
            params={
                "query": query,
                "start": request.window_start.timestamp(),
                "end": request.window_end.timestamp(),
                "step": step,
            },
            timeout=self._timeout,
        ) as response:
            payload, response_hash = await self._response_json(
                response, max_bytes=request.max_response_bytes
            )
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID",
                "Prometheus 返回格式无效",
            )
        result = payload.get("data", {}).get("result", [])
        raw_series = []
        for item in result:
            points = [
                (datetime.fromtimestamp(float(ts), tz=UTC), value)
                for ts, value in item.get("values", [])
            ]
            if points:
                raw_series.append((item.get("metric", {}), points))
        return raw_series, response_hash
