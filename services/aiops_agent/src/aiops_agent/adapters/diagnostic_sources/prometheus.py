"""Prometheus 指标和活动规则证据 Adapter。"""

from __future__ import annotations

from datetime import UTC, datetime

import aiohttp

from aiops_agent.ports.diagnostic_source import (
    EventEvidenceResult,
    EventEvidenceRequest,
    MetricsEvidenceRequest,
    MetricsEvidenceResult,
    SourceHealthRequest,
    SourceHealthResult,
)

from .base import BaseDiagnosticSourceAdapter, DiagnosticSourceAdapterError


def _escape_prometheus_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class PrometheusAdapter(BaseDiagnosticSourceAdapter):
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
        try:
            async with self._session.get(
                f"{self._endpoint().rstrip('/')}/api/v1/alerts",
                headers=self._headers(),
                timeout=self._timeout,
            ) as response:
                payload, _ = await self._response_json(
                    response, max_bytes=1024 * 1024
                )
            alerts = []
            for item in payload.get("data", {}).get("alerts", []):
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
            return EventEvidenceResult(events=tuple(alerts))
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
            query = provider.query_template.replace(
                "${external_target}",
                _escape_prometheus_label(source_locator_key),
            )
            try:
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
                if (
                    not isinstance(payload, dict)
                    or payload.get("status") != "success"
                ):
                    raise DiagnosticSourceAdapterError(
                        "SOURCE_RESPONSE_INVALID",
                        "Prometheus 返回格式无效",
                    )
                result = payload.get("data", {}).get("result", [])
                raw_series = []
                for item in result:
                    points = [
                        (
                            datetime.fromtimestamp(float(ts), tz=UTC),
                            value,
                        )
                        for ts, value in item.get("values", [])
                    ]
                    raw_series.append((item.get("metric", {}), points))
                if not raw_series:
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="SOURCE_NO_DATA",
                            detail="Prometheus 未返回采样",
                        )
                    )
                    continue
                observations.append(
                    self._observation(
                        request=request,
                        definition=definition,
                        raw_series=raw_series,
                        provider_response_hash=response_hash,
                        effective_step=step,
                        truncated=(
                            len(raw_series) > definition.max_series
                        ),
                    )
                )
            except DiagnosticSourceAdapterError as exc:
                gaps.append(
                    self._gap(
                        request,
                        metric_code=definition.metric_code,
                        code=exc.code,
                        detail=str(exc),
                        retryable=exc.retryable,
                    )
                )
        return MetricsEvidenceResult(
            observations=tuple(observations), gaps=tuple(gaps)
        )
