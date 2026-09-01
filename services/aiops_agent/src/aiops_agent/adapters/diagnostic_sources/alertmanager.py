"""Alertmanager 事件接收 Adapter。"""

from __future__ import annotations

import hashlib
from datetime import datetime

import aiohttp

from aiops_agent.contracts.evidence import (
    NormalizedSignalBatch,
    NormalizedSignalEvent,
)
from aiops_agent.domain.evidence import SignalEventStatus, SignalSeverity
from aiops_agent.ports.diagnostic_source import (
    SignalWebhookRequest,
    SourceHealthRequest,
    SourceHealthResult,
)

from .base import BaseDiagnosticSourceAdapter, DiagnosticSourceAdapterError

_SEVERITY = {
    "critical": SignalSeverity.CRITICAL,
    "high": SignalSeverity.HIGH,
    "warning": SignalSeverity.WARNING,
    "warn": SignalSeverity.WARNING,
    "info": SignalSeverity.INFO,
}
_TARGET_LABEL = "target_key"


class AlertmanagerAdapter(BaseDiagnosticSourceAdapter):
    """接收 Alertmanager Webhook，并保留来源告警语义。"""

    endpoint_required = False

    async def health_check(
        self, request: SourceHealthRequest
    ) -> SourceHealthResult:
        if self.context.endpoint is None:
            healthy = bool(self.context.credentials.get("webhook_secret"))
            return self._health_result(
                healthy=healthy,
                error_code=None if healthy else "SOURCE_AUTH_FAILED",
            )
        try:
            async with self._session.get(
                f"{self.context.endpoint.rstrip('/')}/-/ready",
                headers=self._headers(),
                timeout=self._timeout,
            ) as response:
                if response.status in {401, 403}:
                    return self._health_result(
                        healthy=False, error_code="SOURCE_AUTH_FAILED"
                    )
                return self._health_result(
                    healthy=response.status == 200,
                    error_code=(
                        None
                        if response.status == 200
                        else "SOURCE_API_UNAVAILABLE"
                    ),
                )
        except (aiohttp.ClientError, TimeoutError):
            return self._health_result(
                healthy=False, error_code="SOURCE_UNREACHABLE"
            )

    async def verify_and_normalize_webhook(
        self, request: SignalWebhookRequest
    ) -> NormalizedSignalBatch:
        self._verify_hmac(
            headers=request.headers,
            body=request.body,
            received_at=request.received_at,
        )
        payload = self._json(request.body)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("alerts"), list
        ):
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID",
                "Alertmanager Webhook 格式无效",
            )
        batch_status = str(payload.get("status", "firing")).lower()
        events = []
        for item in payload["alerts"]:
            labels = item.get("labels") or {}
            annotations = item.get("annotations") or {}
            source_locator_key = str(labels.get(_TARGET_LABEL, "")).strip()
            if not source_locator_key:
                raise DiagnosticSourceAdapterError(
                    "SOURCE_TARGET_NOT_FOUND",
                    f"Alertmanager 告警缺少 {_TARGET_LABEL} 标签",
                )
            starts_at = str(item.get("startsAt", ""))
            try:
                occurred_at = datetime.fromisoformat(
                    starts_at.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise DiagnosticSourceAdapterError(
                    "SOURCE_RESPONSE_INVALID",
                    "Alertmanager startsAt 格式无效",
                ) from exc
            status_text = str(item.get("status", batch_status)).lower()
            status = (
                SignalEventStatus.RESOLVED
                if status_text == "resolved"
                else SignalEventStatus.FIRING
            )
            alertname = str(labels.get("alertname", "prometheus.alert"))
            event_class = str(
                labels.get("event_class") or alertname
            )
            transition_at = (
                str(item.get("endsAt", ""))
                if status == SignalEventStatus.RESOLVED
                else starts_at
            )
            source_fingerprint = str(item.get("fingerprint") or "").strip()
            source_event_key = hashlib.sha256(
                (
                    f"{source_fingerprint}|{source_locator_key}|"
                    f"{event_class}|{status}|{transition_at}"
                ).encode("utf-8")
            ).hexdigest()
            events.append(
                NormalizedSignalEvent(
                    source_event_key=source_event_key,
                    source_locator_key=source_locator_key,
                    event_type=event_class[:64],
                    event_status=status,
                    severity=_SEVERITY.get(
                        str(labels.get("severity", "warning")).lower(),
                        SignalSeverity.WARNING,
                    ),
                    occurred_at=occurred_at,
                    fingerprint_basis=(source_fingerprint or event_class),
                    summary=str(
                        annotations.get("summary")
                        or annotations.get("description")
                        or event_class
                    )[:1000],
                    provider_attributes={
                        "alertname": alertname,
                        "event_class": event_class,
                        "status": status_text,
                        "target_label": _TARGET_LABEL,
                    },
                    normalizer_version="alertmanager.v1",
                )
            )
        return NormalizedSignalBatch(
            provider_delivery_id=request.headers.get(
                "x-alertmanager-delivery-id"
            ),
            events=tuple(events),
        )
