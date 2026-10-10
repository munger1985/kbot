"""Zabbix JSON-RPC 与 Webhook Adapter。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import aiohttp

from aiops_agent.contracts.evidence import (
    NormalizedSignalEvent,
    NormalizedSignalBatch,
)
from aiops_agent.domain.evidence import (
    SignalEventStatus,
    SignalSeverity,
)
from aiops_agent.ports.diagnostic_source import (
    EventEvidenceRequest,
    EventEvidenceResult,
    MetricsEvidenceRequest,
    MetricsEvidenceResult,
    SignalWebhookRequest,
    SourceHealthRequest,
    SourceHealthResult,
)

from .base import BaseDiagnosticSourceAdapter, DiagnosticSourceAdapterError


_SEVERITY = {
    "0": SignalSeverity.INFO,
    "1": SignalSeverity.INFO,
    "2": SignalSeverity.WARNING,
    "3": SignalSeverity.WARNING,
    "4": SignalSeverity.HIGH,
    "5": SignalSeverity.CRITICAL,
}


class ZabbixAdapter(BaseDiagnosticSourceAdapter):
    _HISTORY_TYPES = {
        "FLOAT": 0,
        "CHARACTER": 1,
        "LOG": 2,
        "UNSIGNED": 3,
        "TEXT": 4,
    }

    async def _rpc(
        self,
        method: str,
        params: dict,
        *,
        max_bytes: int,
        authenticated: bool = True,
    ):
        body = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        }
        if authenticated:
            body["auth"] = await self._authentication_token(max_bytes=max_bytes)
        async with self._session.post(
            self.context.endpoint,
            json=body,
            timeout=self._timeout,
        ) as response:
            payload, response_hash = await self._response_json(
                response, max_bytes=max_bytes
            )
        if not isinstance(payload, dict):
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID", "Zabbix JSON-RPC 返回错误"
            )
        if payload.get("error"):
            error = payload["error"] if isinstance(payload["error"], dict) else {}
            message = " ".join(
                str(error.get(key) or "") for key in ("message", "data")
            ).lower()
            code = (
                "SOURCE_AUTH_FAILED"
                if "author" in message or "session" in message
                else "SOURCE_RESPONSE_INVALID"
            )
            raise DiagnosticSourceAdapterError(code, "Zabbix JSON-RPC 返回错误")
        return payload.get("result", []), response_hash

    async def _authentication_token(self, *, max_bytes: int) -> str:
        mode = str(self.context.config.get("auth_mode") or "").upper()
        api_token = self.context.credentials.get("token")
        username = self.context.credentials.get("username")
        password = self.context.credentials.get("password")
        if mode in {"", "API_TOKEN"} and api_token:
            return api_token
        if mode in {"", "USERNAME_PASSWORD"} and username and password:
            cached = getattr(self, "_login_token", None)
            if cached:
                return cached
            token, _ = await self._rpc(
                "user.login",
                {"username": username, "password": password},
                max_bytes=max_bytes,
                authenticated=False,
            )
            if not isinstance(token, str) or not token:
                raise DiagnosticSourceAdapterError(
                    "SOURCE_AUTH_FAILED", "Zabbix 登录未返回会话"
                )
            self._login_token = token
            return token
        raise DiagnosticSourceAdapterError(
            "SOURCE_AUTH_FAILED", "Zabbix 认证模式或凭据无效"
        )

    async def health_check(
        self, request: SourceHealthRequest
    ) -> SourceHealthResult:
        del request
        try:
            version, _ = await self._rpc(
                "apiinfo.version", {}, max_bytes=64 * 1024, authenticated=False
            )
            if not isinstance(version, str) or not version:
                raise DiagnosticSourceAdapterError(
                    "SOURCE_RESPONSE_INVALID", "Zabbix API 版本响应无效"
                )
            await self._rpc(
                "host.get",
                {"output": ["hostid"], "limit": 1},
                max_bytes=256 * 1024,
            )
            return self._health_result(healthy=True)
        except DiagnosticSourceAdapterError as exc:
            return self._health_result(healthy=False, error_code=exc.code)
        except (aiohttp.ClientError, TimeoutError):
            return self._health_result(
                healthy=False, error_code="SOURCE_UNREACHABLE"
            )

    async def query_events(
        self, request: EventEvidenceRequest
    ) -> EventEvidenceResult:
        try:
            hosts, _ = await self._rpc(
                "host.get",
                {
                    "output": ["hostid", "host"],
                    "filter": {"host": [request.source_locator_key]},
                },
                max_bytes=1024 * 1024,
            )
            exact = [
                item
                for item in hosts
                if item.get("host") == request.source_locator_key
            ]
            if len(exact) != 1:
                raise DiagnosticSourceAdapterError(
                    (
                        "SOURCE_TARGET_NOT_FOUND"
                        if not exact
                        else "SOURCE_TARGET_AMBIGUOUS"
                    ),
                    "Zabbix 精确 Host 不存在或不唯一",
                )
            problems, _ = await self._rpc(
                "problem.get",
                {
                    "output": ["eventid", "name", "severity", "clock"],
                    "hostids": [exact[0]["hostid"]],
                    "recent": True,
                    "sortfield": ["clock"],
                    "sortorder": "DESC",
                    "limit": request.max_events,
                },
                max_bytes=1024 * 1024,
            )
            return EventEvidenceResult(
                events=tuple(
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
        except (DiagnosticSourceAdapterError, TypeError, ValueError, KeyError) as exc:
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
                        detail="Zabbix 活动告警读取失败",
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
        try:
            hosts, _ = await self._rpc(
                "host.get",
                {
                    "output": ["hostid", "host"],
                    "filter": {"host": [request.source_locator_key]},
                },
                max_bytes=request.max_response_bytes,
            )
            exact = [
                item
                for item in hosts
                if item.get("host") == request.source_locator_key
            ]
            if len(exact) != 1:
                raise DiagnosticSourceAdapterError(
                    (
                        "SOURCE_TARGET_NOT_FOUND"
                        if not exact
                        else "SOURCE_TARGET_AMBIGUOUS"
                    ),
                    "Zabbix 精确 Host 不存在或不唯一",
                )
            host_id = exact[0]["hostid"]
            for definition in request.metric_definitions:
                provider = definition.providers.get("ZABBIX")
                if provider is None or provider.exact_item_key is None:
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="SOURCE_QUERY_UNSUPPORTED",
                            detail="Zabbix 未定义该指标",
                        )
                    )
                    continue
                items, _ = await self._rpc(
                    "item.get",
                    {
                        "output": ["itemid", "key_", "value_type", "units"],
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
                if not exact_items:
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="SOURCE_NO_DATA",
                            detail="Zabbix 精确 Item 不存在",
                        )
                    )
                    continue
                if len(exact_items) > 1:
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="SOURCE_ITEM_AMBIGUOUS",
                            detail="Zabbix 精确 Item 不唯一",
                        )
                    )
                    continue
                item = exact_items[0]
                expected_type = provider.value_type
                actual_type = str(item.get("value_type", ""))
                history_type = self._HISTORY_TYPES.get(str(expected_type))
                if history_type is None or actual_type != str(history_type):
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="SOURCE_METRIC_TYPE_MISMATCH",
                            detail="Zabbix Item Value Type 与标准指标不一致",
                        )
                    )
                    continue
                if str(item.get("units") or "") != str(provider.unit or ""):
                    gaps.append(
                        self._gap(
                            request,
                            metric_code=definition.metric_code,
                            code="SOURCE_METRIC_UNIT_MISMATCH",
                            detail="Zabbix Item 单位与标准指标不一致",
                        )
                    )
                    continue
                history, response_hash = await self._rpc(
                    "history.get",
                    {
                        "output": "extend",
                        "history": history_type,
                        "itemids": [item["itemid"]],
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
                            code="SOURCE_NO_DATA",
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
        except (DiagnosticSourceAdapterError, TypeError, ValueError, KeyError) as exc:
            code = (
                exc.code
                if isinstance(exc, DiagnosticSourceAdapterError)
                else "SOURCE_RESPONSE_INVALID"
            )
            for definition in request.metric_definitions:
                gaps.append(
                    self._gap(
                        request,
                        metric_code=definition.metric_code,
                        code=code,
                        detail="Zabbix 指标读取失败",
                        retryable=(
                            exc.retryable
                            if isinstance(exc, DiagnosticSourceAdapterError)
                            else False
                        ),
                    )
                )
        return MetricsEvidenceResult(
            observations=tuple(observations), gaps=tuple(gaps)
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
        if not isinstance(payload, dict):
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID", "Zabbix Webhook 格式无效"
            )
        event_id = str(payload.get("eventid", "")).strip()
        host = str(
            payload.get("host") or payload.get("host_name") or ""
        ).strip()
        if not event_id or not host:
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID",
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
            SignalEventStatus.RESOLVED
            if str(payload.get("status", "")).upper()
            in {"RESOLVED", "OK", "0"}
            else SignalEventStatus.FIRING
        )
        event = NormalizedSignalEvent(
            source_event_key=event_id,
            source_locator_key=host,
            event_type="zabbix.problem",
            event_status=status,
            severity=_SEVERITY.get(
                str(payload.get("severity", "2")),
                SignalSeverity.WARNING,
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
        return NormalizedSignalBatch(
            provider_delivery_id=request.headers.get("x-zabbix-delivery-id"),
            events=(event,),
        )
