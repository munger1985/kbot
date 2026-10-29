"""Zabbix JSON-RPC 与 Webhook Adapter。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

import aiohttp

from aiops_agent.contracts.evidence import (
    DiscoveredMetric, MetricDiscoveryResult, NativeMetricSamples, NativeHistoryMetadata,
    NormalizedSignalEvent,
    NormalizedSignalBatch,
    SourceEventQuery,
)
from aiops_agent.domain.evidence import (
    SignalEventStatus,
    SignalSeverity,
)
from aiops_agent.ports.diagnostic_source import (
    MetricsDiscoveryRequest,
    NativeMetricQueryRequest,
    CAPABILITY_ITEM_HISTORY,
    EventEvidenceRequest,
    EventEvidenceResult,
    InstanceDiscoveryCandidate,
    InstanceDiscoveryRequest,
    InstanceDiscoveryResult,
    MetricsEvidenceRequest,
    MetricsEvidenceResult,
    SignalWebhookRequest,
    SourceHealthRequest,
    SourceHealthResult,
)

from .base import BaseDiagnosticSourceAdapter, DiagnosticSourceAdapterError
from .pagination import SourcePageRow, read_source_pages


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

    @staticmethod
    def _database_type(host: dict) -> str | None:
        aliases = {
            "ORACLE": "ORACLE",
            "MYSQL": "MYSQL",
            "POSTGRES": "POSTGRESQL",
            "POSTGRESQL": "POSTGRESQL",
        }
        tagged: set[str] = set()
        for tag in host.get("tags") or []:
            if not isinstance(tag, dict):
                continue
            name = str(tag.get("tag") or "").strip().lower()
            if name not in {
                "database_type",
                "db_type",
                "ammolite_database_type",
            }:
                continue
            value = str(tag.get("value") or "").strip().upper()
            if value in aliases:
                tagged.add(aliases[value])
        if len(tagged) == 1:
            return next(iter(tagged))
        if tagged:
            return None
        template_types: set[str] = set()
        for template in host.get("parentTemplates") or []:
            if not isinstance(template, dict):
                continue
            name = str(template.get("name") or "").upper()
            for marker, db_type in aliases.items():
                if marker in name:
                    template_types.add(db_type)
        return next(iter(template_types)) if len(template_types) == 1 else None

    async def _metric_host(self, request, *, max_bytes):
        hosts, response_hash = await self._rpc(
            "host.get", {"output": ["hostid", "host"], "filter": {"host": [request.source_locator_key]}, "limit": 2},
            max_bytes=max_bytes,
        )
        if not isinstance(hosts, list) or any(
            not isinstance(item, dict) or item.get("host") != request.source_locator_key for item in hosts
        ):
            raise DiagnosticSourceAdapterError("SOURCE_TARGET_MISMATCH", "来源返回非绑定 Host")
        if len(hosts) != 1:
            raise DiagnosticSourceAdapterError("SOURCE_TARGET_NOT_FOUND", "绑定 Host 不存在或不唯一")
        return str(hosts[0]["hostid"]), response_hash

    async def discover_metrics(self, request: MetricsDiscoveryRequest) -> MetricDiscoveryResult:
        """使用真实 JSON-RPC Item 目录，不以当前值伪装请求窗口历史覆盖。"""
        self._remaining_response_bytes = request.max_response_bytes
        if request.scope != "DATABASE":
            raise DiagnosticSourceAdapterError("SOURCE_QUERY_UNSUPPORTED", "Zabbix 仅发现当前绑定 Host，不推定关联主机权限")
        host_id, host_hash = await self._metric_host(request, max_bytes=request.max_response_bytes // 4)
        items, digest = await self._rpc(
            "item.get", {
                "output": ["itemid", "hostid", "key_", "name", "description", "units", "value_type", "status", "state"],
                "hostids": [host_id], "sortfield": "key_", "limit": request.limit + 1,
            }, max_bytes=request.max_response_bytes * 3 // 4,
        )
        if not isinstance(items, list) or any(
            not isinstance(item, dict) or str(item.get("hostid")) != host_id for item in items
        ):
            raise DiagnosticSourceAdapterError("SOURCE_TARGET_MISMATCH", "Item 目录包含绑定范围外对象")
        entries = []
        for item in items[:request.limit]:
            name = str(item.get("key_") or "")
            if not name or len(name) > 256 or any(ord(char) < 32 for char in name):
                continue
            entries.append(DiscoveredMetric(
                name=name, metric_type="UNKNOWN", native_value_type=str(item.get("value_type", "")),
                unit=str(item.get("units") or "")[:64] or None,
                help_text=str(item.get("description") or item.get("name") or "")[:2000] or None,
                metadata_state="VERIFIED",
                query_supported=((str(item.get("value_type")) in {"0", "3"}
                    or (str(item.get("value_type")) in {"1", "2", "4"}
                        and CAPABILITY_ITEM_HISTORY in self.context.declared_capabilities))
                    and str(item.get("status")) == "0" and str(item.get("state")) == "0"),
            ))
        return MetricDiscoveryResult(
            action_id=request.action_id, target_id=request.target_id, binding_id=request.binding_id,
            source_id=self.context.source_id, scope=request.scope,
            window_start=request.window_start, window_end=request.window_end,
            captured_at=datetime.now(UTC), entries=tuple(entries),
            response_sha256=hashlib.sha256(f"{host_hash}:{digest}".encode()).hexdigest(),
            truncated=len(items) > request.limit or len(entries) < min(len(items), request.limit),
            warnings=("Item 元数据已读取；数值存储类型不证明 counter/gauge 语义，目录不证明请求窗口有采样。",),
        )

    async def query_native_metric(self, request: NativeMetricQueryRequest, *, metric_name, limit, allow_text=False):
        """精确重验 Host/Item；文本日志需额外授权，不猜单位或自动改用趋势聚合。"""
        self._remaining_response_bytes = request.max_response_bytes
        if not isinstance(metric_name, str) or not 0 < len(metric_name) <= 256 or any(ord(char) < 32 for char in metric_name):
            raise DiagnosticSourceAdapterError("SOURCE_QUERY_INVALID", "精确 Item key 无效")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise DiagnosticSourceAdapterError("SOURCE_QUERY_INVALID", "历史结果上限无效")
        host_id, host_hash = await self._metric_host(request, max_bytes=request.max_response_bytes // 4)
        items, item_hash = await self._rpc(
            "item.get", {
                "output": ["itemid", "hostid", "key_", "value_type", "units", "status", "state"],
                "hostids": [host_id], "filter": {"key_": [metric_name]}, "limit": 2,
            }, max_bytes=request.max_response_bytes // 4,
        )
        if not isinstance(items, list) or any(
            not isinstance(item, dict) or str(item.get("hostid")) != host_id or item.get("key_") != metric_name for item in items
        ):
            raise DiagnosticSourceAdapterError("SOURCE_TARGET_MISMATCH", "Item 不属于精确绑定对象")
        if len(items) != 1:
            raise DiagnosticSourceAdapterError("SOURCE_ITEM_AMBIGUOUS", "精确 Item 不存在或不唯一")
        item = items[0]
        value_type = str(item.get("value_type"))
        numeric = value_type in {"0", "3"}
        if value_type not in {"0", "1", "2", "3", "4"} or str(item.get("status")) != "0" or str(item.get("state")) != "0":
            raise DiagnosticSourceAdapterError("SOURCE_METRIC_TYPE_MISMATCH", "Item 类型不受支持或未启用")
        if not numeric and (not allow_text or CAPABILITY_ITEM_HISTORY not in self.context.declared_capabilities):
            raise DiagnosticSourceAdapterError("SOURCE_CAPABILITY_FORBIDDEN", "文本/日志历史需要来源与绑定显式授权 item.history")
        pagination = request.pagination
        pages = pagination.max_pages if pagination is not None else 1
        def epoch_ns(stamp):
            delta = stamp - datetime(1970, 1, 1, tzinfo=UTC)
            return ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds) * 1000
        start_ns, end_ns = epoch_ns(request.window_start), epoch_ns(request.window_end)
        async def fetch_history(frontier, fetch_limit):
            history, digest = await self._rpc("history.get", {
                "output": ["itemid", "clock", "ns", "value", *(["timestamp", "source", "severity", "logeventid"] if value_type == "2" else [])], "history": int(value_type),
                "itemids": [item["itemid"]], "time_from": frontier,
                "time_till": int(request.window_end.timestamp()), "sortfield": ["clock", "ns"],
                "sortorder": "ASC", "limit": fetch_limit,
            }, max_bytes=request.max_response_bytes // 2 // pages)
            if not isinstance(history, list) or len(history) > fetch_limit:
                raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "历史响应格式无效")
            rows = []
            for row in history:
                if not isinstance(row, dict) or str(row.get("itemid")) != str(item["itemid"]):
                    raise DiagnosticSourceAdapterError("SOURCE_TARGET_MISMATCH", "历史采样来自其他 Item")
                try:
                    clock = int(row["clock"])
                    nanoseconds = int(row.get("ns", 0))
                    if not 0 <= nanoseconds <= 999999999:
                        raise ValueError("纳秒偏移无效")
                    stamp = datetime.fromtimestamp(clock, tz=UTC) + timedelta(microseconds=nanoseconds // 1000)
                    value = row["value"]
                    if (not isinstance(value, str) or len(value) > 8192
                        or not int(request.window_start.timestamp()) <= clock <= int(request.window_end.timestamp())):
                        raise ValueError("采样时间或值无效")
                    if numeric:
                        number = Decimal(value)
                        if len(value) > 256 or not number.is_finite() or abs(number.as_tuple().exponent) > 1024:
                            raise ValueError("采样数值无效")
                        if value_type == "3" and (not value.isdecimal() or not 0 <= number <= 18446744073709551615):
                            raise ValueError("无符号计数值无效")
                    metadata = NativeHistoryMetadata(
                        nanoseconds=nanoseconds,
                        log_timestamp_seconds=int(row["timestamp"]) if value_type == "2" and row.get("timestamp") is not None else None,
                        log_source=row.get("source") if value_type == "2" else None,
                        log_severity=str(row["severity"]) if value_type == "2" and row.get("severity") is not None else None,
                        log_event_id=str(row["logeventid"]) if value_type == "2" and row.get("logeventid") is not None else None,
                    )
                except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError, OSError):
                    raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "原生历史采样的时间、值或元数据无效") from None
                rows.append(SourcePageRow(position=clock, order=nanoseconds, identity=f"{clock}:{nanoseconds}",
                    content_hash=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                    value=(stamp, value, metadata), include=start_ns <= clock * 1000000000 + nanoseconds <= end_ns))
            return rows, digest
        summary = None
        if pagination is not None:
            values, summary = await read_source_pages(fetch_page=fetch_history, pagination=pagination,
                result_limit=limit, start=int(request.window_start.timestamp()))
            digests = summary.response_hashes
            truncated = summary.stop_reason != "EXHAUSTED"
        else:
            rows, digest = await fetch_history(int(request.window_start.timestamp()), limit + 1)
            values = tuple(row.value for row in rows[:limit] if row.include)
            digests = (digest,)
            truncated = len(rows) > limit
        return NativeMetricSamples(
            action_id=request.action_id, target_id=request.target_id, binding_id=request.binding_id,
            source_id=self.context.source_id, metric_name=metric_name,
            native_value_type=str(item["value_type"]), unit=(str(item.get("units") or "")[:64] or None) if numeric else None,
            sample_kind="NUMERIC" if numeric else "LOG" if value_type == "2" else "TEXT",
            sample_metadata=tuple(row[2] for row in values),
            source_version=f"{self.context.adapter_version}:{self.context.config_version}",
            window_start=request.window_start, window_end=request.window_end, captured_at=datetime.now(UTC),
            points=tuple((row[0], row[1]) for row in values), truncated=truncated, pagination=summary,
            response_sha256=hashlib.sha256(":".join((host_hash, item_hash, *digests)).encode()).hexdigest(),
        )

    async def discover_instances(
        self, request: InstanceDiscoveryRequest
    ) -> InstanceDiscoveryResult:
        params = {
            "output": ["hostid", "host", "name"],
            "selectTags": ["tag", "value"],
            "selectParentTemplates": ["templateid", "name"],
            "sortfield": "host",
            "sortorder": "ASC",
            "limit": 1001,
        }
        if request.locator_keys:
            params["filter"] = {"host": list(request.locator_keys)}
        hosts, _ = await self._rpc(
            "host.get", params, max_bytes=2 * 1024 * 1024
        )
        if not isinstance(hosts, list):
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID", "Zabbix Host 目录响应无效"
            )
        accepted_types = set(request.db_types)
        requested_keys = set(request.locator_keys)
        candidates = []
        for host in hosts:
            if not isinstance(host, dict):
                continue
            locator_key = str(host.get("host") or "").strip()
            db_type = self._database_type(host)
            if (
                not locator_key
                or len(locator_key) > 512
                or db_type is None
                or (accepted_types and db_type not in accepted_types)
                or (requested_keys and locator_key not in requested_keys)
                or (
                    request.after_locator_key is not None
                    and locator_key <= request.after_locator_key
                )
            ):
                continue
            display_name = str(host.get("name") or locator_key).strip()
            candidates.append(
                InstanceDiscoveryCandidate(
                    source_locator_key=locator_key,
                    source_locator={
                        "host": locator_key,
                        "database_type": db_type,
                    },
                    db_type=db_type,
                    display_name=display_name[:256],
                )
            )
        candidates.sort(key=lambda item: item.source_locator_key)
        return InstanceDiscoveryResult(
            candidates=tuple(candidates[: request.limit])
        )

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
        self._remaining_response_bytes = request.max_response_bytes
        try:
            hosts, host_hash = await self._rpc(
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
            problems, response_hash = await self._rpc(
                "problem.get",
                {
                    "output": ["eventid", "name", "severity", "clock", "r_eventid"],
                    "hostids": [exact[0]["hostid"]],
                    "recent": True,
                    "sortfield": ["clock"],
                    "sortorder": "DESC",
                    "limit": request.max_events,
                },
                max_bytes=request.max_response_bytes,
            )
            if not isinstance(problems, list) or len(problems) > request.max_events:
                raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "问题列表超出来源合同")
            for problem in problems:
                if not isinstance(problem, dict):
                    raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "问题记录格式无效")
                recovery = problem.get("r_eventid")
                if recovery is not None and (type(recovery) not in {str, int} or not str(recovery).isdecimal()):
                    raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "恢复事件标识无法核验")
            return EventEvidenceResult(
                query=SourceEventQuery(
                    source_type="ZABBIX", source_id=self.context.source_id,
                    target_id=request.target_id, binding_id=request.binding_id, query_scope="RECENT_PROBLEMS",
                    window_start=request.window_start, window_end=request.window_end,
                    captured_at=datetime.now(UTC),
                    response_sha256=hashlib.sha256(f"{host_hash}:{response_hash}".encode()).hexdigest(),
                    source_version=f"{self.context.adapter_id}:{self.context.adapter_version}:{self.context.config_version}",
                    truncated=len(problems) >= request.max_events,
                ),
                events=tuple(
                    {
                        "source_type": "ZABBIX",
                        "event_id": str(item.get("eventid", "")),
                        "name": str(item.get("name", "zabbix.problem"))[
                            :128
                        ],
                        "severity": str(item.get("severity", "")),
                        "status": (
                            "UNKNOWN" if item.get("r_eventid") is None
                            else "FIRING" if str(item["r_eventid"]) == "0" else "RESOLVED"
                        ),
                        "is_open": (
                            None if item.get("r_eventid") is None else str(item["r_eventid"]) == "0"
                        ),
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
