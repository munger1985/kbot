"""Zabbix JSON-RPC 与 Webhook Adapter。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

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
    "0": MonitorSeverity.INFO,
    "1": MonitorSeverity.INFO,
    "2": MonitorSeverity.WARNING,
    "3": MonitorSeverity.WARNING,
    "4": MonitorSeverity.HIGH,
    "5": MonitorSeverity.CRITICAL,
}


class ZabbixAdapter(BaseMonitorAdapter):
    async def _rpc(self, method: str, params: dict, *, max_bytes: int):
        body = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        }
        token = self.context.credentials.get("token")
        if token:
            body["auth"] = token
        async with self._session.post(
            self.context.endpoint,
            json=body,
            timeout=self._timeout,
        ) as response:
            payload, response_hash = await self._response_json(
                response, max_bytes=max_bytes
            )
        if not isinstance(payload, dict) or payload.get("error"):
            raise MonitorAdapterError(
                "MONITOR_RESPONSE_INVALID", "Zabbix JSON-RPC 返回错误"
            )
        return payload.get("result", []), response_hash

    async def query_alerts(
        self, request: AlertQueryRequest
    ) -> AlertQueryResult:
        try:
            hosts, _ = await self._rpc(
                "host.get",
                {
                    "output": ["hostid", "host"],
                    "filter": {"host": [request.external_target_key]},
                },
                max_bytes=1024 * 1024,
            )
            exact = [
                item
                for item in hosts
                if item.get("host") == request.external_target_key
            ]
            if len(exact) != 1:
                raise MonitorAdapterError(
                    "MONITOR_TARGET_NOT_FOUND",
                    "Zabbix 精确 Host 不存在",
                )
            problems, _ = await self._rpc(
                "problem.get",
                {
                    "output": ["eventid", "name", "severity", "clock"],
                    "hostids": [exact[0]["hostid"]],
                    "recent": True,
                    "sortfield": ["clock"],
                    "sortorder": "DESC",
                    "limit": request.max_alerts,
                },
                max_bytes=1024 * 1024,
            )
            return AlertQueryResult(
                alerts=tuple(
                    {
                        "source_type": "ZABBIX",
                        "event_id": str(item.get("eventid", "")),
                        "name": str(item.get("name", "zabbix.problem"))[
                            :128
                        ],
                        "severity": str(item.get("severity", "")),
                        "status": "FIRING",
                        "active_at": datetime.fromtimestamp(
                            int(item["clock"]), tz=UTC
                        ).isoformat(),
                    }
                    for item in problems
                )
            )
        except (MonitorAdapterError, TypeError, ValueError, KeyError) as exc:
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
                        detail="Zabbix 活动告警读取失败",
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
        try:
            hosts, _ = await self._rpc(
                "host.get",
                {
                    "output": ["hostid", "host"],
                    "filter": {"host": [request.external_target_key]},
                },
                max_bytes=request.max_response_bytes,
            )
            exact = [
                item
                for item in hosts
                if item.get("host") == request.external_target_key
            ]
            if len(exact) != 1:
                raise MonitorAdapterError(
                    "MONITOR_TARGET_NOT_FOUND",
                    "Zabbix 精确 Host 不存在",
                )
            host_id = exact[0]["hostid"]
            for definition in request.metric_definitions:
                provider = definition.providers.get("ZABBIX")
                if provider is None or provider.exact_item_key is None:
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="MONITOR_QUERY_UNSUPPORTED",
                            detail="Zabbix 未定义该指标",
                        )
                    )
                    continue
                items, _ = await self._rpc(
                    "item.get",
                    {
                        "output": ["itemid", "key_"],
                        "hostids": [host_id],
                        "filter": {"key_": [provider.exact_item_key]},
                    },
                    max_bytes=request.max_response_bytes,
                )
                exact_items = [
                    item
                    for item in items
                    if item.get("key_") == provider.exact_item_key
                ]
                if len(exact_items) != 1:
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="MONITOR_NO_DATA",
                            detail="Zabbix 精确 Item 不存在",
                        )
                    )
                    continue
                history, response_hash = await self._rpc(
                    "history.get",
                    {
                        "output": "extend",
                        "history": 0,
                        "itemids": [exact_items[0]["itemid"]],
                        "time_from": int(request.window_start.timestamp()),
                        "time_till": int(request.window_end.timestamp()),
                        "sortfield": "clock",
                        "sortorder": "ASC",
                        "limit": definition.max_points + 1,
                    },
                    max_bytes=request.max_response_bytes,
                )
                if not history:
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="MONITOR_NO_DATA",
                            detail="Zabbix 未返回历史采样",
                        )
                    )
                    continue
                observations.append(
                    self._observation(
                        request=request,
                        definition=definition,
                        raw_series=[
                            (
                                {},
                                [
                                    (
                                        datetime.fromtimestamp(
                                            int(item["clock"]), tz=UTC
                                        ),
                                        item.get("value"),
                                    )
                                    for item in history
                                ],
                            )
                        ],
                        provider_response_hash=response_hash,
                        effective_step=max(
                            request.requested_step_seconds,
                            definition.min_step_seconds,
                        ),
                        truncated=len(history) > definition.max_points,
                    )
                )
        except MonitorAdapterError as exc:
            for definition in request.metric_definitions:
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
        if not isinstance(payload, dict):
            raise MonitorAdapterError(
                "MONITOR_RESPONSE_INVALID", "Zabbix Webhook 格式无效"
            )
        event_id = str(payload.get("eventid", "")).strip()
        host = str(
            payload.get("host") or payload.get("host_name") or ""
        ).strip()
        if not event_id or not host:
            raise MonitorAdapterError(
                "MONITOR_RESPONSE_INVALID",
                "Zabbix Webhook 缺少 eventid 或 host",
            )
        clock = payload.get("clock")
        occurred = (
            datetime.fromtimestamp(int(clock), tz=UTC)
            if clock is not None
            else request.received_at
        )
        problem = str(
            payload.get("problem") or payload.get("name") or "zabbix.event"
        )
        status = (
            MonitorEventStatus.RESOLVED
            if str(payload.get("status", "")).upper()
            in {"RESOLVED", "OK", "0"}
            else MonitorEventStatus.FIRING
        )
        event = NormalizedMonitorEvent(
            source_event_key=event_id,
            external_target_key=host,
            event_type="zabbix.problem",
            event_status=status,
            severity=_SEVERITY.get(
                str(payload.get("severity", "2")),
                MonitorSeverity.WARNING,
            ),
            occurred_at=occurred,
            fingerprint_basis=hashlib.sha256(
                f"{host}|{problem}".encode()
            ).hexdigest(),
            summary=problem[:1000],
            provider_attributes={
                "eventid": event_id,
                "status": str(payload.get("status", "")),
            },
            normalizer_version="zabbix-webhook.v1",
        )
        return NormalizedWebhookBatch(
            provider_delivery_id=request.headers.get("x-zabbix-delivery-id"),
            events=(event,),
        )
