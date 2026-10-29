"""OEM官方Incident只读接口；当前未关闭事件不能冒充完整历史告警。"""

from datetime import UTC, datetime
import math

import aiohttp

from aiops_agent.contracts.evidence import SourceEventQuery
from aiops_agent.ports.diagnostic_source import CAPABILITY_EVENT_QUERY, EventEvidenceResult
from .base import DiagnosticSourceAdapterError
from .oem_native import _collection, _fail, _hash


def _text(value, maximum):
    if not isinstance(value, str) or not value or len(value) > maximum:
        _fail()
    return value


def _time(value):
    try:
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError
        return result.astimezone(UTC)
    except (ValueError, TypeError, OverflowError):
        _fail()


class OEMEvents:
    """精确Target筛选、总字节预算和不跟随来源链接的单页读取。"""

    async def query_events(self, request):
        try:
            if CAPABILITY_EVENT_QUERY not in self.context.declared_capabilities:
                _fail("SOURCE_CAPABILITY_FORBIDDEN")
            if not request.window_start < request.window_end:
                _fail("SOURCE_QUERY_INVALID")
            self._remaining_response_bytes = request.max_response_bytes
            target, target_hash = await self._native_target(request)
            captured_at = datetime.now(UTC)
            age = max(1, math.ceil((captured_at - request.window_start).total_seconds() / 3600))
            if age > 2147483647:
                _fail("SOURCE_QUERY_INVALID")
            page_limit = min(request.max_events, 100)
            payload, response_hash = await self._native_get("/incidents", params={
                "targetName": target["targetName"], "targetTypeName": target["targetType"],
                "includeMemberTargets": "false", "ageInHoursLessThanOrEqualTo": age,
                "sort": "timeCreated:DESC", "limit": page_limit,
            }, max_bytes=request.max_response_bytes * 3 // 4)
            items, more = _collection(payload)
            if len(items) > page_limit:
                _fail()
            events, seen, previous = [], set(), None
            for item in items:
                incident_id = _text(item.get("id"), 32)
                if len(incident_id) != 32 or any(char not in "0123456789abcdefABCDEF" for char in incident_id):
                    _fail()
                if incident_id.casefold() in seen:
                    _fail("SOURCE_ITEM_AMBIGUOUS")
                seen.add(incident_id.casefold())
                message = _text(item.get("message"), 4000)
                severity = item.get("severity")
                if not isinstance(severity, str) or severity not in {"FATAL", "CRITICAL", "WARNING", "MINOR_WARNING", "CLEAR"}:
                    _fail()
                status = _text(item.get("status"), 64)
                is_open = item.get("isOpen")
                if type(is_open) is not bool:
                    _fail()
                created, updated = _time(item.get("timeCreated")), _time(item.get("timeUpdated"))
                if updated < created or (previous is not None and created > previous):
                    _fail()
                previous = created
                targets = item.get("targets")
                if not isinstance(targets, list) or not 1 <= len(targets) <= 64:
                    _fail("SOURCE_TARGET_MISMATCH")
                target_matches = 0
                for native in targets:
                    if not isinstance(native, dict):
                        _fail()
                    name = _text(native.get("targetName"), 256)
                    kind = _text(native.get("targetType"), 64)
                    if name == target["targetName"] and kind == target["targetType"]:
                        target_matches += 1
                if target_matches != 1:
                    _fail("SOURCE_TARGET_MISMATCH")
                if not is_open or not request.window_start <= created < request.window_end:
                    continue
                events.append({
                    "source_type": "OEM", "incident_id": incident_id, "name": message,
                    "severity": severity, "status": status.upper(), "is_open": True,
                    "active_at": created.isoformat(), "updated_at": updated.isoformat(),
                })
            query = SourceEventQuery(
                source_type="OEM", source_id=self.context.source_id, target_id=request.target_id,
                binding_id=request.binding_id, query_scope="CURRENT_OPEN_INCIDENTS",
                window_start=request.window_start, window_end=request.window_end,
                captured_at=datetime.now(UTC), source_version=f"{self.context.adapter_id}:{self.context.adapter_version}:{self.context.config_version}",
                response_sha256=_hash(target_hash, response_hash), truncated=more or len(items) == page_limit,
            )
            return EventEvidenceResult(events=tuple(events), query=query)
        except (DiagnosticSourceAdapterError, aiohttp.ClientError, TimeoutError) as exc:
            return EventEvidenceResult(gaps=(self._gap(
                request, metric_code=None, code=getattr(exc, "code", "SOURCE_UNREACHABLE"),
                detail="OEM当前未关闭事件读取未获得可核验结果；不能推断无告警。",
                retryable=getattr(exc, "retryable", isinstance(exc, (aiohttp.ClientError, TimeoutError))),
            ),))
