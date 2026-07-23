"""Prometheus/Alertmanager Adapter。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from aiops_agent.contracts.monitoring import (
    NormalizedMonitorEvent,
    NormalizedWebhookBatch,
)
from aiops_agent.domain.monitoring import (
    MonitorEventStatus,
    MonitorSeverity,
)
from aiops_agent.ports.monitor import (
    AlertQueryRequest,
    AlertQueryResult,
    MetricQueryRequest,
    MetricQueryResult,
    RawWebhookRequest,
)

from .base import BaseMonitorAdapter, MonitorAdapterError


_SEVERITY = {
    "critical": MonitorSeverity.CRITICAL,
    "high": MonitorSeverity.HIGH,
    "warning": MonitorSeverity.WARNING,
    "warn": MonitorSeverity.WARNING,
    "info": MonitorSeverity.INFO,
}


def _escape_prometheus_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class PrometheusAdapter(BaseMonitorAdapter):
    async def query_alerts(
        self, request: AlertQueryRequest
    ) -> AlertQueryResult:
        external_label = str(
            self.context.capabilities.get(
                "external_target_label", "instance"
            )
        )
        try:
            async with self._session.get(
                f"{self.context.endpoint.rstrip('/')}/api/v1/alerts",
                headers=self._headers(),
                timeout=self._timeout,
            ) as response:
                payload, _ = await self._response_json(
                    response, max_bytes=1024 * 1024
                )
            alerts = []
            for item in payload.get("data", {}).get("alerts", []):
                labels = item.get("labels") or {}
                if labels.get(external_label) != request.external_target_key:
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
                if len(alerts) >= request.max_alerts:
                    break
            return AlertQueryResult(alerts=tuple(alerts))
        except (MonitorAdapterError, KeyError, TypeError, ValueError) as exc:
            code = (
                exc.code
                if isinstance(exc, MonitorAdapterError)
                else "MONITOR_RESPONSE_INVALID"
            )
            return AlertQueryResult(
                gaps=(
                    self._gap(
                        request,  # type: ignore[arg-type]
                        metric_code=None,
                        code=code,
                        detail="Prometheus 活动告警读取失败",
                        retryable=(
                            exc.retryable
                            if isinstance(exc, MonitorAdapterError)
                            else False
                        ),
                    ),
                )
            )

    async def query_metrics(
        self, request: MetricQueryRequest
    ) -> MetricQueryResult:
        observations = []
        gaps = []
        for definition in request.metric_definitions:
            provider = definition.providers.get("PROMETHEUS")
            if provider is None or provider.query_template is None:
                gaps.append(
                    self._gap(
                        request,
                        metric_code=definition.metric_code,
                        code="MONITOR_QUERY_UNSUPPORTED",
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
                _escape_prometheus_label(request.external_target_key),
            )
            try:
                async with self._session.get(
                    f"{self.context.endpoint.rstrip('/')}/api/v1/query_range",
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
                    raise MonitorAdapterError(
                        "MONITOR_RESPONSE_INVALID",
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
                            code="MONITOR_NO_DATA",
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
            except MonitorAdapterError as exc:
                gaps.append(
                    self._gap(
                        request,
                        metric_code=definition.metric_code,
                        code=exc.code,
                        detail=str(exc),
                        retryable=exc.retryable,
                    )
                )
        return MetricQueryResult(
            observations=tuple(observations), gaps=tuple(gaps)
        )

    async def verify_and_parse_webhook(
        self, request: RawWebhookRequest
    ) -> NormalizedWebhookBatch:
        self._verify_hmac(
            headers=request.headers,
            body=request.body,
            received_at=request.received_at,
        )
        payload = self._json(request.body)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("alerts"), list
        ):
            raise MonitorAdapterError(
                "MONITOR_RESPONSE_INVALID",
                "Alertmanager Webhook 格式无效",
            )
        batch_status = str(payload.get("status", "firing")).lower()
        events = []
        for item in payload["alerts"]:
            labels = item.get("labels") or {}
            annotations = item.get("annotations") or {}
            external_label = str(
                self.context.capabilities.get(
                    "external_target_label", "instance"
                )
            )
            external = str(labels.get(external_label, "")).strip()
            if not external:
                raise MonitorAdapterError(
                    "MONITOR_TARGET_NOT_FOUND",
                    "Alertmanager 告警缺少登记的目标标签",
                )
            starts_at = str(item.get("startsAt", ""))
            occurred = datetime.fromisoformat(
                starts_at.replace("Z", "+00:00")
            )
            status_text = str(item.get("status", batch_status)).lower()
            status = (
                MonitorEventStatus.RESOLVED
                if status_text == "resolved"
                else MonitorEventStatus.FIRING
            )
            problem = str(labels.get("alertname", "prometheus.alert"))
            alert_fingerprint = str(
                item.get("fingerprint") or ""
            ).strip()
            transition_at = (
                str(item.get("endsAt", ""))
                if status == MonitorEventStatus.RESOLVED
                else starts_at
            )
            source_key = hashlib.sha256(
                (
                    f"{alert_fingerprint}|{external}|{problem}|"
                    f"{status}|{transition_at}"
                ).encode()
            ).hexdigest()
            events.append(
                NormalizedMonitorEvent(
                    source_event_key=source_key,
                    external_target_key=external,
                    event_type=problem[:64],
                    event_status=status,
                    severity=_SEVERITY.get(
                        str(labels.get("severity", "warning")).lower(),
                        MonitorSeverity.WARNING,
                    ),
                    occurred_at=occurred,
                    fingerprint_basis=problem,
                    summary=str(
                        annotations.get("summary")
                        or annotations.get("description")
                        or problem
                    )[:1000],
                    provider_attributes={
                        "alertname": problem,
                        "status": status_text,
                    },
                    normalizer_version="prometheus-alertmanager.v1",
                )
            )
        return NormalizedWebhookBatch(
            provider_delivery_id=request.headers.get(
                "x-alertmanager-delivery-id"
            ),
            events=tuple(events),
        )
