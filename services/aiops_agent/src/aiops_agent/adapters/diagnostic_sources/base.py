"""诊断源 Adapter 的安全公共实现。"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import aiohttp

from aiops_agent.contracts.evidence import (
    MetricObservation,
    MetricPoint,
    MetricSeries,
    ObservationGap,
)
from aiops_agent.domain.evidence import summarize_points
from aiops_agent.ports.diagnostic_source import (
    EventEvidenceRequest,
    LogEvidenceRequest,
    MetricsEvidenceRequest,
    MetricsEvidenceResult,
    SourceHealthRequest,
    SourceHealthResult,
    DiagnosticSourceContext,
)


class DiagnosticSourceAdapterError(RuntimeError):
    def __init__(
        self, code: str, message: str, *, retryable: bool = False
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class BaseDiagnosticSourceAdapter:
    adapter_version = "1.0.0"
    endpoint_required = True

    def __init__(
        self,
        *,
        context: DiagnosticSourceContext,
        session: aiohttp.ClientSession,
        request_timeout_seconds: float,
        webhook_replay_seconds: int,
        supported_capabilities: frozenset[str] = frozenset(),
    ):
        self.context = context
        self._session = session
        self._timeout = aiohttp.ClientTimeout(
            total=request_timeout_seconds
        )
        self._webhook_replay_seconds = webhook_replay_seconds
        self._supported_capabilities = supported_capabilities
        if context.endpoint is None:
            if self.endpoint_required:
                raise DiagnosticSourceAdapterError(
                    "SOURCE_ENDPOINT_REQUIRED", "诊断源缺少查询地址"
                )
        else:
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
            raise DiagnosticSourceAdapterError(
                "SOURCE_ENDPOINT_INVALID", "诊断源地址格式无效"
            )

    def _endpoint(self) -> str:
        if self.context.endpoint is None:
            raise DiagnosticSourceAdapterError(
                "SOURCE_ENDPOINT_REQUIRED", "诊断源缺少查询地址"
            )
        return self.context.endpoint

    def _health_result(
        self, *, healthy: bool, error_code: str | None = None
    ) -> SourceHealthResult:
        return SourceHealthResult(
            healthy=healthy,
            error_code=error_code,
            adapter_id=self.context.adapter_id,
            adapter_version=self.adapter_version,
            discovered_capabilities=tuple(
                sorted(self._supported_capabilities)
            ),
        )

    def _headers(self) -> dict[str, str]:
        token = self.context.credentials.get("token")
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def health_check(
        self, request: SourceHealthRequest
    ) -> SourceHealthResult:
        try:
            async with self._session.get(
                self._endpoint(),
                headers=self._headers(),
                timeout=self._timeout,
            ) as response:
                if response.status in {401, 403}:
                    return self._health_result(
                        healthy=False, error_code="SOURCE_AUTH_FAILED"
                    )
                return self._health_result(
                    healthy=response.status < 500,
                    error_code=(
                        None
                        if response.status < 500
                        else "SOURCE_UNREACHABLE"
                    ),
                )
        except (aiohttp.ClientError, TimeoutError):
            return self._health_result(
                healthy=False, error_code="SOURCE_UNREACHABLE"
            )

    def _verify_hmac(
        self,
        *,
        headers: dict[str, str],
        body: bytes,
        received_at: datetime,
    ) -> None:
        secret = self.context.credentials.get("webhook_secret")
        if not secret:
            raise DiagnosticSourceAdapterError(
                "SOURCE_AUTH_FAILED", "Webhook 验签凭据不存在"
            )
        timestamp_text = headers.get("x-kbot-timestamp")
        signature = headers.get("x-kbot-signature", "")
        try:
            timestamp = datetime.fromtimestamp(
                int(timestamp_text or ""), tz=UTC
            )
        except (TypeError, ValueError, OSError) as exc:
            raise DiagnosticSourceAdapterError(
                "SOURCE_AUTH_FAILED", "Webhook 时间戳无效"
            ) from exc
        if (
            abs((received_at.astimezone(UTC) - timestamp).total_seconds())
            > self._webhook_replay_seconds
        ):
            raise DiagnosticSourceAdapterError(
                "SOURCE_REPLAY_REJECTED", "Webhook 已超出重放窗口"
            )
        signed = timestamp_text.encode("ascii") + b"." + body
        expected = hmac.new(
            secret.encode("utf-8"), signed, hashlib.sha256
        ).hexdigest()
        supplied = signature.removeprefix("sha256=")
        if not hmac.compare_digest(expected, supplied):
            raise DiagnosticSourceAdapterError(
                "SOURCE_AUTH_FAILED", "Webhook 签名无效"
            )

    @staticmethod
    def _json(body: bytes) -> Any:
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID", "诊断源正文不是有效 JSON"
            ) from exc

    def _observation(
        self,
        *,
        request: MetricsEvidenceRequest,
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
            source_version=self.context.config_version,
            target_id=request.target_id,
            binding_id=request.binding_id,
            external_target_fingerprint=hashlib.sha256(
                request.source_locator_key.encode("utf-8")
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
        request: MetricsEvidenceRequest | EventEvidenceRequest | LogEvidenceRequest,
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
            raise DiagnosticSourceAdapterError(
                "SOURCE_AUTH_FAILED", "诊断源拒绝认证"
            )
        if response.status == 429:
            raise DiagnosticSourceAdapterError(
                "SOURCE_RATE_LIMITED",
                "诊断源触发限流",
                retryable=True,
            )
        if response.status >= 500:
            raise DiagnosticSourceAdapterError(
                "SOURCE_UNREACHABLE",
                "诊断源暂时不可用",
                retryable=True,
            )
        raw = await response.read()
        if len(raw) > max_bytes:
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESULT_TRUNCATED", "诊断源响应超过字节预算"
            )
        return self._json(raw), hashlib.sha256(raw).hexdigest()

    async def query_metrics(
        self, request: MetricsEvidenceRequest
    ) -> MetricsEvidenceResult:
        raise NotImplementedError
