"""Loki 日志证据查询 Adapter。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

import aiohttp

from aiops_agent.contracts.evidence import (
    LogEvidenceEntry,
    LogEvidenceSet,
)
from aiops_agent.ports.diagnostic_source import (
    LogEvidenceRequest,
    SourceHealthRequest,
    SourceHealthResult,
)

from .base import BaseDiagnosticSourceAdapter, DiagnosticSourceAdapterError


_CREDENTIAL_VALUE = re.compile(
    r"(?i)\b(password|passwd|pwd|token|secret|api[_-]?key)"
    r"(\s*[=:]\s*)([^\s,;]+)"
)


def _escape_logql_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _selector(labels: dict[str, str]) -> str:
    parts = [
        f'{name}="{_escape_logql_label(value)}"'
        for name, value in sorted(labels.items())
    ]
    return "{" + ",".join(parts) + "}"


class LokiAdapter(BaseDiagnosticSourceAdapter):
    """只允许精确标签选择器，不接收任意 LogQL。"""

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        tenant_id = str(self.context.config.get("tenant_id", "")).strip()
        if tenant_id:
            if len(tenant_id) > 256 or any(
                character in tenant_id for character in "\r\n"
            ):
                raise DiagnosticSourceAdapterError(
                    "SOURCE_CONFIGURATION_INVALID",
                    "Loki tenant_id 格式无效",
                )
            headers["X-Scope-OrgID"] = tenant_id
        return headers

    async def health_check(
        self, request: SourceHealthRequest
    ) -> SourceHealthResult:
        try:
            async with self._session.get(
                f"{self._endpoint().rstrip('/')}/ready",
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

    async def query_logs(
        self, request: LogEvidenceRequest
    ) -> LogEvidenceSet:
        selector = _selector(request.selector_labels)
        query = selector + "".join(
            f" {operator} {json.dumps(value, ensure_ascii=False)}"
            for operator, value in request.line_filters
        )
        query_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "selector": request.selector_labels,
                    "line_filters": request.line_filters,
                    "window_start": request.window_start.isoformat(),
                    "window_end": request.window_end.isoformat(),
                    "max_entries": request.max_entries,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        gaps = ()
        entries: tuple[LogEvidenceEntry, ...] = ()
        truncated = False
        try:
            async with self._session.get(
                f"{self._endpoint().rstrip('/')}/loki/api/v1/query_range",
                headers=self._headers(),
                params={
                    "query": query,
                    "start": int(request.window_start.timestamp() * 1_000_000_000),
                    "end": int(request.window_end.timestamp() * 1_000_000_000),
                    "limit": request.max_entries,
                    "direction": "backward",
                },
                timeout=self._timeout,
            ) as response:
                if 400 <= response.status < 500 and response.status not in {
                    401,
                    403,
                    429,
                }:
                    raise DiagnosticSourceAdapterError(
                        "SOURCE_QUERY_INVALID", "Loki 拒绝受控日志查询"
                    )
                payload, response_hash = await self._response_json(
                    response, max_bytes=request.max_response_bytes
                )
            entries = self._parse_entries(
                payload=payload,
                request=request,
            )
            truncated = len(entries) >= request.max_entries
        except (
            DiagnosticSourceAdapterError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            aiohttp.ClientError,
            TimeoutError,
        ) as exc:
            if isinstance(exc, DiagnosticSourceAdapterError):
                code = exc.code
                retryable = exc.retryable
            elif isinstance(exc, (aiohttp.ClientError, TimeoutError)):
                code = "SOURCE_UNREACHABLE"
                retryable = True
            else:
                code = "SOURCE_RESPONSE_INVALID"
                retryable = False
            gaps = (
                self._gap(
                    request,
                    metric_code=None,
                    code=code,
                    detail="Loki 日志证据读取失败",
                    retryable=retryable,
                ),
            )
            response_hash = ""
        return LogEvidenceSet(
            target_id=request.target_id,
            binding_id=request.binding_id,
            source_id=self.context.source_id,
            window_start=request.window_start,
            window_end=request.window_end,
            entries=entries,
            gaps=gaps,
            collected_at=datetime.now(UTC),
            truncated=truncated,
            query_fingerprint=query_fingerprint,
            provenance={
                "adapter_id": self.context.adapter_id,
                "adapter_version": self.adapter_version,
                "provider_response_hash": response_hash,
            },
        )

    def _parse_entries(
        self,
        *,
        payload: Any,
        request: LogEvidenceRequest,
    ) -> tuple[LogEvidenceEntry, ...]:
        if (
            not isinstance(payload, dict)
            or payload.get("status") != "success"
            or payload.get("data", {}).get("resultType") != "streams"
        ):
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID", "Loki 返回格式无效"
            )
        parsed: list[LogEvidenceEntry] = []
        for stream in payload.get("data", {}).get("result", []):
            labels = self._safe_mapping(stream.get("stream", {}), limit=32)
            for raw_entry in stream.get("values", []):
                if len(parsed) >= request.max_entries:
                    break
                timestamp_ns = int(raw_entry[0])
                line = self._redact_line(str(raw_entry[1]).strip())[:4000]
                if not line:
                    continue
                structured = self._safe_mapping(
                    raw_entry[2] if len(raw_entry) > 2 else {},
                    limit=32,
                )
                observed_at = datetime.fromtimestamp(
                    timestamp_ns / 1_000_000_000, tz=UTC
                )
                fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "timestamp_ns": timestamp_ns,
                            "labels": labels,
                            "line": line,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                parsed.append(
                    LogEvidenceEntry(
                        observed_at=observed_at,
                        line=line,
                        labels=labels,
                        structured_fields=structured,
                        entry_fingerprint=fingerprint,
                    )
                )
        parsed.sort(key=lambda item: (item.observed_at, item.entry_fingerprint))
        return tuple(parsed)

    @staticmethod
    def _safe_mapping(value: Any, *, limit: int) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key)[:128]: (
                "[已脱敏]"
                if re.fullmatch(
                    r"(?i)(password|passwd|pwd|token|secret|api[_-]?key)",
                    str(key),
                )
                else LokiAdapter._redact_line(str(item))[:512]
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[
                :limit
            ]
        }

    @staticmethod
    def _redact_line(line: str) -> str:
        return _CREDENTIAL_VALUE.sub(
            lambda match: (
                f"{match.group(1)}{match.group(2)}[已脱敏]"
            ),
            line,
        )
