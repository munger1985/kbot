"""Oracle Enterprise Manager 只读指标 Adapter。"""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import quote

from aiops_agent.ports.monitor import (
    AlertQueryRequest,
    AlertQueryResult,
    MetricQueryRequest,
    MetricQueryResult,
)

from .base import BaseMonitorAdapter, MonitorAdapterError


class OEMAdapter(BaseMonitorAdapter):
    async def query_alerts(
        self, request: AlertQueryRequest
    ) -> AlertQueryResult:
        path = (
            f"{self.context.endpoint.rstrip('/')}/targets/"
            f"{quote(request.external_target_key, safe='')}/incidents"
        )
        try:
            async with self._session.get(
                path,
                headers=self._headers(),
                params={
                    "startTime": request.window_start.isoformat(),
                    "endTime": request.window_end.isoformat(),
                    "limit": request.max_alerts,
                },
                timeout=self._timeout,
            ) as response:
                payload, _ = await self._response_json(
                    response, max_bytes=1024 * 1024
                )
            items = (
                payload.get("items", [])
                if isinstance(payload, dict)
                else []
            )
            return AlertQueryResult(
                alerts=tuple(
                    {
                        "source_type": "OEM",
                        "incident_id": str(
                            item.get("incidentId", "")
                        ),
                        "name": str(
                            item.get("summary", "oem.incident")
                        )[:128],
                        "severity": str(
                            item.get("severity", "UNKNOWN")
                        ).upper(),
                        "status": str(
                            item.get("status", "OPEN")
                        ).upper(),
                        "active_at": item.get("createdTime"),
                    }
                    for item in items[: request.max_alerts]
                )
            )
        except (MonitorAdapterError, TypeError, ValueError) as exc:
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
                        detail="OEM Incident 读取失败",
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
            provider = definition.providers.get("OEM")
            if (
                provider is None
                or provider.target_type is None
                or provider.metric_name is None
            ):
                gaps.append(
                    self._gap(
                        request,
                        metric_code=definition.metric_code,
                        code="MONITOR_QUERY_UNSUPPORTED",
                        detail="OEM 未定义该指标",
                    )
                )
                continue
            path = (
                f"{self.context.endpoint.rstrip('/')}/targets/"
                f"{quote(request.external_target_key, safe='')}/"
                f"types/{quote(provider.target_type, safe='')}/metrics/"
                f"{quote(provider.metric_name, safe='')}"
            )
            try:
                async with self._session.get(
                    path,
                    headers=self._headers(),
                    params={
                        "startTime": request.window_start.isoformat(),
                        "endTime": request.window_end.isoformat(),
                    },
                    timeout=self._timeout,
                ) as response:
                    payload, response_hash = await self._response_json(
                        response, max_bytes=request.max_response_bytes
                    )
                rows = (
                    payload.get("items", [])
                    if isinstance(payload, dict)
                    else []
                )
                raw_points = []
                for item in rows:
                    timestamp = str(
                        item.get("timestamp") or item.get("collectionTime")
                    )
                    raw_points.append(
                        (
                            datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            ).astimezone(UTC),
                            item.get("value"),
                        )
                    )
                if not raw_points:
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="MONITOR_NO_DATA",
                            detail="OEM 未返回采样",
                        )
                    )
                    continue
                observations.append(
                    self._observation(
                        request=request,
                        definition=definition,
                        raw_series=[({}, raw_points)],
                        provider_response_hash=response_hash,
                        effective_step=max(
                            request.requested_step_seconds,
                            definition.min_step_seconds,
                        ),
                        truncated=len(raw_points) > definition.max_points,
                    )
                )
            except (
                MonitorAdapterError,
                TypeError,
                ValueError,
            ) as exc:
                code = (
                    exc.code
                    if isinstance(exc, MonitorAdapterError)
                    else "MONITOR_RESPONSE_INVALID"
                )
                gaps.append(
                    self._gap(
                        request,
                        metric_code=definition.metric_code,
                        code=code,
                        detail="OEM 指标响应无法解析",
                        retryable=(
                            exc.retryable
                            if isinstance(exc, MonitorAdapterError)
                            else False
                        ),
                    )
                )
        return MetricQueryResult(
            observations=tuple(observations), gaps=tuple(gaps)
        )

    async def verify_and_parse_webhook(self, request):
        raise MonitorAdapterError(
            "MONITOR_QUERY_UNSUPPORTED",
            "OEM 步骤 5 不直接接收 Incident Webhook",
        )
