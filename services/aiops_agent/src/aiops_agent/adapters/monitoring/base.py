"""监控 Adapter 的安全公共实现。"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import aiohttp

from aiops_agent.contracts.monitoring import (
    MetricObservation,
    MetricPoint,
    MetricSeries,
    ObservationGap,
)
from aiops_agent.domain.monitoring import summarize_points
from aiops_agent.ports.monitor import (
    MetricQueryRequest,
    MetricQueryResult,
    MonitorHealthRequest,
    MonitorHealthResult,
    MonitorProviderContext,
)


class MonitorAdapterError(RuntimeError):
    def __init__(
        self, code: str, message: str, *, retryable: bool = False
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class BaseMonitorAdapter:
    adapter_version = "1.0.0"

    def __init__(
        self,
        *,
        context: MonitorProviderContext,
        session: aiohttp.ClientSession,
        request_timeout_seconds: float,
        webhook_replay_seconds: int,
    ):
        self.context = context
        self._session = session
        self._timeout = aiohttp.ClientTimeout(
            total=request_timeout_seconds
        )
        self._webhook_replay_seconds = webhook_replay_seconds
        self._validate_endpoint(context.endpoint)

    @staticmethod
    def _validate_endpoint(endpoint: str) -> None:
        parsed = urlparse(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise MonitorAdapterError(
                "MONITOR_ENDPOINT_INVALID", "监控源地址格式无效"
            )

    def _headers(self) -> dict[str, str]:
        token = self.context.credentials.get("token")
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def health_check(
        self, request: MonitorHealthRequest
    ) -> MonitorHealthResult:
        try:
            async with self._session.get(
                self.context.endpoint,
                headers=self._headers(),
                timeout=self._timeout,
            ) as response:
                if response.status in {401, 403}:
                    return MonitorHealthResult(
                        healthy=False,
                        error_code="MONITOR_AUTH_FAILED",
                        adapter_version=self.adapter_version,
                    )
                return MonitorHealthResult(
                    healthy=response.status < 500,
                    error_code=(
                        None
                        if response.status < 500
                        else "MONITOR_UNREACHABLE"
                    ),
                    adapter_version=self.adapter_version,
                )
        except (aiohttp.ClientError, TimeoutError):
            return MonitorHealthResult(
                healthy=False,
                error_code="MONITOR_UNREACHABLE",
                adapter_version=self.adapter_version,
            )

    async def query_alerts(self, request):
        from aiops_agent.ports.monitor import AlertQueryResult

        return AlertQueryResult()

    def _verify_hmac(
        self,
        *,
        headers: dict[str, str],
        body: bytes,
        received_at: datetime,
    ) -> None:
        secret = self.context.credentials.get("webhook_secret")
        if not secret:
            raise MonitorAdapterError(
                "MONITOR_AUTH_FAILED", "Webhook 验签凭据不存在"
            )
        timestamp_text = headers.get("x-kbot-timestamp")
        signature = headers.get("x-kbot-signature", "")
        try:
            timestamp = datetime.fromtimestamp(
                int(timestamp_text or ""), tz=UTC
            )
        except (TypeError, ValueError, OSError) as exc:
            raise MonitorAdapterError(
                "MONITOR_AUTH_FAILED", "Webhook 时间戳无效"
            ) from exc
        if (
            abs((received_at.astimezone(UTC) - timestamp).total_seconds())
            > self._webhook_replay_seconds
        ):
            raise MonitorAdapterError(
                "MONITOR_REPLAY_REJECTED", "Webhook 已超出重放窗口"
            )
        signed = timestamp_text.encode("ascii") + b"." + body
        expected = hmac.new(
            secret.encode("utf-8"), signed, hashlib.sha256
        ).hexdigest()
        supplied = signature.removeprefix("sha256=")
        if not hmac.compare_digest(expected, supplied):
            raise MonitorAdapterError(
                "MONITOR_AUTH_FAILED", "Webhook 签名无效"
            )

    @staticmethod
    def _json(body: bytes) -> Any:
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MonitorAdapterError(
                "MONITOR_RESPONSE_INVALID", "监控正文不是有效 JSON"
            ) from exc

    def _observation(
        self,
        *,
        request: MetricQueryRequest,
        definition,
        raw_series: list[tuple[dict[str, str], list[tuple[datetime, Any]]]],
        provider_response_hash: str,
        effective_step: int,
        truncated: bool,
        warnings: tuple[str, ...] = (),
    ) -> MetricObservation:
        allowed_dimensions = frozenset(
            definition.expected_dimensions
        )
        series: list[MetricSeries] = []
        for dimensions, raw_points in raw_series[
            : definition.max_series
        ]:
            safe_dimensions = {
                key: str(value)[:256]
                for key, value in sorted(dimensions.items())
                if key in allowed_dimensions
            }
            points = []
            for observed_at, value in raw_points[
                : definition.max_points
            ]:
                quality = "GOOD"
                parsed: float | str | bool | None = value
                if definition.value_kind == "STATE":
                    try:
                        numeric_state = float(value)
                        parsed = (
                            int(numeric_state)
                            if numeric_state.is_integer()
                            else numeric_state
                        )
                    except (TypeError, ValueError):
                        parsed = value
                else:
                    try:
                        parsed = float(value)
                        if not math.isfinite(parsed):
                            raise ValueError
                    except (TypeError, ValueError):
                        parsed = None
                        quality = "INVALID"
                points.append(
                    MetricPoint(
                        observed_at=observed_at,
                        value=parsed,
                        quality=quality,
                    )
                )
            series.append(
                MetricSeries(
                    dimensions=safe_dimensions,
                    points=tuple(points),
                )
            )
        frozen = tuple(series)
        actual = sum(
            1
            for item in frozen
            for point in item.points
            if point.quality == "GOOD"
        )
        expected = max(
            1,
            int(
                (
                    request.window_end - request.window_start
                ).total_seconds()
                // effective_step
            ),
        )
        provider = definition.providers[self.context.source_type]
        return MetricObservation(
            metric_code=definition.metric_code,
            semantic_version=definition.semantic_version,
            unit=definition.unit,
            value_kind=definition.value_kind,
            window_start=request.window_start,
            window_end=request.window_end,
            requested_step_seconds=request.requested_step_seconds,
            effective_step_seconds=effective_step,
            source_id=self.context.source_id,
            source_type=self.context.source_type,
            source_version=self.context.source_version,
            target_id=request.target_id,
            binding_id=request.binding_id,
            external_target_fingerprint=hashlib.sha256(
                request.external_target_key.encode("utf-8")
            ).hexdigest(),
            series=frozen,
            summary=summarize_points(frozen),
            expected_points=expected,
            actual_points=actual,
            coverage_ratio=min(1.0, actual / expected),
            truncated=truncated,
            warnings=warnings,
            provenance={
                "template_id": provider.template_id,
                "template_version": provider.template_version,
                "provider_response_hash": provider_response_hash,
                "adapter_version": self.adapter_version,
            },
        )

    def _gap(
        self,
        request: MetricQueryRequest,
        *,
        metric_code: str | None,
        code: str,
        detail: str,
        retryable: bool = False,
    ) -> ObservationGap:
        return ObservationGap(
            metric_code=metric_code,
            source_id=self.context.source_id,
            binding_id=request.binding_id,
            code=code,
            detail=detail,
            retryable=retryable,
        )

    async def _response_json(
        self, response: aiohttp.ClientResponse, *, max_bytes: int
    ) -> tuple[Any, str]:
        if response.status in {401, 403}:
            raise MonitorAdapterError(
                "MONITOR_AUTH_FAILED", "监控源拒绝认证"
            )
        if response.status == 429:
            raise MonitorAdapterError(
                "MONITOR_RATE_LIMITED",
                "监控源触发限流",
                retryable=True,
            )
        if response.status >= 500:
            raise MonitorAdapterError(
                "MONITOR_UNREACHABLE",
                "监控源暂时不可用",
                retryable=True,
            )
        raw = await response.read()
        if len(raw) > max_bytes:
            raise MonitorAdapterError(
                "MONITOR_RESULT_TRUNCATED", "监控响应超过字节预算"
            )
        return self._json(raw), hashlib.sha256(raw).hexdigest()

    async def query_metrics(
        self, request: MetricQueryRequest
    ) -> MetricQueryResult:
        raise NotImplementedError
