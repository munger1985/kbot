"""受控、可分页且强制脱敏的本地运行日志查询。"""

from __future__ import annotations

import base64
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal


LogStream = Literal["RUNTIME", "ACCESS"]
_INCLUDED_SERVICES = frozenset(
    {"main_api", "agent_runtime", "knowledge_core", "model_serving", "data_query"}
)
_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_RUNTIME_HEADER = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
    r"\s+\|\s+(?P<level>[A-Z]+)\s+\|\s+"
    r"\[(?P<process>[^\]]+)\]\s+"
    r"(?P<location>.*?)\s+-\s+(?P<message>.*)$"
)
_ACCESS_HEADER = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
    r"\s+\|\s+(?P<level>[A-Z]+)\s+\|\s+"
    r"\[(?P<process>[^\]]+)\]\s+(?P<message>.*)$"
)
_IDENTIFIERS = {
    "error_id": re.compile(r"\berror_id[=:]\s*([^\s|,]+)", re.I),
    "request_id": re.compile(r"\brequest_id[=:]\s*([^\s|,]+)", re.I),
    "trace_id": re.compile(r"\btrace_id[=:]\s*([^\s|,]+)", re.I),
    "run_id": re.compile(r"\brun_id[=:]\s*([^\s|,]+)", re.I),
    "job_id": re.compile(r"\b(?:job_id|kc_job_id)[=:]\s*([^\s|,]+)", re.I),
    "turn_id": re.compile(r"\bturn_id[=:]\s*([^\s|,]+)", re.I),
    "task_id": re.compile(r"\btask_id[=:]\s*([^\s|,]+)", re.I),
    "model_call_id": re.compile(r"\bmodel_call_id[=:]\s*([^\s|,]+)", re.I),
    "model_id": re.compile(r"\bmodel_id[=:]\s*([^\s|,]+)", re.I),
}
_EVENT_NAME = re.compile(r"\bevent[=:]\s*([a-zA-Z0-9._-]+)")
_HTTP_STATUS = re.compile(r"(?:status|Status Code|状态码)[=:]\s*(\d{3})")
_DURATION = re.compile(
    r"(?:duration_ms|Processing Time|处理耗时|耗时)[=:]\s*"
    r"([\d.]+)(?:\s*ms)?"
)
_SENSITIVE_KEY = re.compile(
    r"(?i)(authorization|cookie|api[_-]?key|password|passwd|secret|credential|"
    r"private[_-]?key|access[_-]?token|refresh[_-]?token|database[_-]?username|"
    r"db[_-]?username|username)"
)
_LARGE_KEY = re.compile(
    r"(?i)(raw_result|query_result|result_rows|artifact_payload|prompt|messages|"
    r"document_content|image_base64|response_body|request_body|original_input)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_KEY_VALUE_SECRET = re.compile(
    r"(?i)[\"']?\b(authorization|cookie|api[_-]?key|password|passwd|secret|credential|"
    r"private[_-]?key|access[_-]?token|refresh[_-]?token|database[_-]?username|"
    r"db[_-]?username|username)[\"']?\s*[=:]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s|,;]+)"
)
_URL_CREDENTIAL = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://)([^\s:/@]+):([^\s@]+)@"
)
_ORACLE_CREDENTIAL = re.compile(
    r"(?i)\b([a-z0-9_.-]+)/([^\s@]+)@([a-z0-9_.:-]+(?:/[^\s]+)?)"
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
    re.S,
)


class LogQueryError(ValueError):
    """日志查询参数或受控目录不合法。"""


@dataclass(frozen=True, slots=True)
class LogFileRef:
    service_name: str
    stream: LogStream
    path: Path
    relative_path: str
    size: int
    modified_at: float
    device: int
    inode: int


def _topology_services(path: Path) -> frozenset[str]:
    """从内部拓扑构造允许访问的日志目录别名。"""
    try:
        with path.open("rb") as stream:
            processes = tomllib.load(stream).get("processes", [])
    except (OSError, tomllib.TOMLDecodeError):
        processes = []
    configs = {
        str(item.get("service_config") or "")
        for item in processes
        if str(item.get("service_config") or "") in _INCLUDED_SERVICES
    }
    aliases = set(configs)
    aliases.update(f"kbot_{name}" for name in configs)
    service_names = {
        str(item.get("service_name") or "")
        for item in processes
        if str(item.get("service_config") or "") in _INCLUDED_SERVICES
    }
    aliases.update(service_names)
    aliases.update(item.replace("-", "_") for item in service_names)
    return frozenset(item for item in aliases if item)


class LogFileCatalog:
    """只发现拓扑允许目录下的 runtime/access 日志及轮转文件。"""

    def __init__(
        self, *, root: Path, topology_path: Path, max_files_per_stream: int,
    ) -> None:
        self.root = root.resolve()
        self.allowed_services = _topology_services(topology_path.resolve())
        self.max_files_per_stream = max_files_per_stream

    def files(self) -> list[LogFileRef]:
        if not self.root.is_dir():
            return []
        rows: list[LogFileRef] = []
        for service_name in sorted(self.allowed_services):
            service_dir = self.root / service_name
            if service_dir.is_symlink() or not service_dir.is_dir():
                continue
            resolved_dir = service_dir.resolve()
            if not resolved_dir.is_relative_to(self.root):
                continue
            for stream, base_name in (
                ("RUNTIME", "runtime.log"), ("ACCESS", "access.log"),
            ):
                candidates = []
                for path in service_dir.glob(f"{base_name}*"):
                    if path.is_symlink() or not path.is_file():
                        continue
                    if path.name != base_name and not path.name.startswith(base_name + "."):
                        continue
                    resolved = path.resolve()
                    if not resolved.is_relative_to(resolved_dir):
                        continue
                    stat = resolved.stat()
                    candidates.append((resolved, stat))
                candidates.sort(key=lambda item: item[1].st_mtime, reverse=True)
                for resolved, stat in candidates[: self.max_files_per_stream]:
                    rows.append(LogFileRef(
                        service_name=service_name,
                        stream=stream,
                        path=resolved,
                        relative_path=str(resolved.relative_to(self.root)),
                        size=stat.st_size,
                        modified_at=stat.st_mtime,
                        device=stat.st_dev,
                        inode=stat.st_ino,
                    ))
        return rows


class LocalLogSearchService:
    """在全局扫描预算内查询、分页并重定位日志事件。"""

    def __init__(
        self, *, log_root: Path, topology_path: Path,
        max_files_per_stream: int = 8,
        max_bytes_per_file: int = 2 * 1024 * 1024,
        max_total_scan_bytes: int = 16 * 1024 * 1024,
        max_window_hours: int = 24 * 31,
        max_page_size: int = 500,
        max_export_events: int = 2000,
        max_detail_chars: int = 65536,
        max_field_chars: int = 4096,
    ) -> None:
        self._catalog = LogFileCatalog(
            root=log_root,
            topology_path=topology_path,
            max_files_per_stream=max_files_per_stream,
        )
        self._max_bytes_per_file = max_bytes_per_file
        self._max_total_scan_bytes = max_total_scan_bytes
        self._max_window = timedelta(hours=max_window_hours)
        self.max_page_size = max_page_size
        self.max_export_events = max_export_events
        self._max_detail_chars = max_detail_chars
        self._max_field_chars = max_field_chars

    def services(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for ref in self._catalog.files():
            item = grouped.setdefault(ref.service_name, {
                "service_name": ref.service_name,
                "runtime": None,
                "access": None,
            })
            key = ref.stream.lower()
            summary = item[key] or {
                "stream": ref.stream, "files": 0, "size": 0,
                "modified_at": None,
            }
            summary["files"] += 1
            summary["size"] += ref.size
            modified = datetime.fromtimestamp(
                ref.modified_at, tz=timezone.utc
            ).astimezone().isoformat()
            if summary["modified_at"] is None or modified > summary["modified_at"]:
                summary["modified_at"] = modified
            item[key] = summary
        return [grouped[key] for key in sorted(grouped)]

    def search(
        self, *, service_name: str | None = None,
        streams: set[str] | None = None, levels: set[str] | None = None,
        filter_by_level: bool = False, keyword: str | None = None,
        request_id: str | None = None, trace_id: str | None = None,
        error_id: str | None = None, run_id: str | None = None,
        job_id: str | None = None, http_status: int | None = None,
        started_at: datetime | None = None, ended_at: datetime | None = None,
        cursor: str | None = None, limit: int = 200,
        _limit_ceiling: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, int]:
        limit_ceiling = _limit_ceiling or self.max_page_size
        if limit < 1 or limit > limit_ceiling:
            raise LogQueryError(f"limit 必须介于 1 和 {limit_ceiling} 之间")
        selected_streams = {item.upper() for item in (streams or {"RUNTIME", "ACCESS"})}
        if not selected_streams or not selected_streams <= {"RUNTIME", "ACCESS"}:
            raise LogQueryError("stream 只能包含 RUNTIME 或 ACCESS")
        if service_name and service_name not in self._catalog.allowed_services:
            raise LogQueryError("service_name 不属于受控服务目录")
        requested_levels = {item.upper() for item in (levels or set())}
        if requested_levels - _LEVELS:
            raise LogQueryError("level 包含不支持的日志级别")
        selected_levels = requested_levels if filter_by_level else _LEVELS
        fingerprint = self._fingerprint({
            "service": service_name, "streams": sorted(selected_streams),
            "levels": sorted(selected_levels), "filter_by_level": filter_by_level,
            "keyword": keyword, "request_id": request_id, "trace_id": trace_id,
            "error_id": error_id, "run_id": run_id, "job_id": job_id,
            "http_status": http_status,
            "start": _aware(started_at).isoformat() if started_at else None,
            "end": _aware(ended_at).isoformat() if ended_at else None,
        })
        if cursor:
            after_key, start, end = self._decode_cursor(cursor, fingerprint)
        else:
            after_key = None
            start, end = self._window(started_at, ended_at)
        events = self._scan(
            service_name=service_name, streams=selected_streams,
            levels=selected_levels, keyword=keyword,
            identifiers={
                "request_id": request_id, "trace_id": trace_id,
                "error_id": error_id, "run_id": run_id, "job_id": job_id,
            },
            http_status=http_status, started_at=start, ended_at=end,
        )
        total = len(events)
        if after_key:
            events = [event for event in events if self._sort_key(event) < after_key]
        page = events[:limit]
        next_cursor = None
        if len(events) > limit and page:
            next_cursor = self._encode_cursor(
                self._sort_key(page[-1]), fingerprint, start, end,
            )
        return [self._list_projection(event) for event in page], next_cursor, total

    def export(self, **filters) -> list[dict[str, Any]]:
        """返回受上限保护的列表投影，不导出 Raw 或 Traceback。"""
        filters = dict(filters)
        filters["limit"] = min(
            int(filters.get("limit") or self.max_export_events),
            self.max_export_events,
        )
        events, _, _ = self.search(
            **filters,
            _limit_ceiling=self.max_export_events,
        )
        return events

    def event_detail(self, *, event_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"[0-9a-f]{64}", event_id):
            raise LogQueryError("event_id 格式无效")
        for event in self._scan(
            service_name=None, streams={"RUNTIME", "ACCESS"}, levels=_LEVELS,
            keyword=None, identifiers={}, http_status=None,
            started_at=datetime.min.replace(tzinfo=timezone.utc),
            ended_at=datetime.max.replace(tzinfo=timezone.utc),
        ):
            if event["event_id"] == event_id:
                return {
                    key: value for key, value in event.items()
                    if key != "_search_text"
                }
        return None

    def search_correlated(
        self, *, identifiers: set[str], limit: int = 500,
    ) -> list[dict[str, Any]]:
        normalized = {item.strip().casefold() for item in identifiers if item.strip()}
        if not normalized:
            return []
        start, end = self._window(None, None)
        events = self._scan(
            service_name=None, streams={"RUNTIME"}, levels=_LEVELS,
            keyword=None, identifiers={}, http_status=None,
            started_at=start, ended_at=end,
        )
        matched = [
            self._list_projection(event)
            for event in reversed(events)
            if any(item in event["_search_text"] for item in normalized)
        ]
        return matched[-limit:]

    def _scan(
        self, *, service_name: str | None, streams: set[str], levels: set[str],
        keyword: str | None, identifiers: dict[str, str | None],
        http_status: int | None, started_at: datetime, ended_at: datetime,
    ) -> list[dict[str, Any]]:
        remaining = self._max_total_scan_bytes
        events: list[dict[str, Any]] = []
        needle = str(keyword or "").strip().casefold()
        for ref in self._catalog.files():
            if remaining <= 0:
                break
            if service_name and ref.service_name != service_name:
                continue
            if ref.stream not in streams:
                continue
            budget = min(self._max_bytes_per_file, remaining)
            remaining -= min(ref.size, budget)
            for event in self._read_events(ref, budget=budget):
                timestamp = datetime.fromisoformat(event["timestamp"])
                if timestamp < started_at or timestamp > ended_at:
                    continue
                if event["level"] not in levels:
                    continue
                if needle and needle not in event["_search_text"]:
                    continue
                if any(
                    expected is not None and str(event.get(field) or "") != expected
                    for field, expected in identifiers.items()
                ):
                    continue
                if http_status is not None and event.get("http_status") != http_status:
                    continue
                events.append(event)
        events.sort(key=self._sort_key, reverse=True)
        return events

    def _read_events(self, ref: LogFileRef, *, budget: int) -> list[dict[str, Any]]:
        start = max(0, ref.size - budget)
        with ref.path.open("rb") as stream:
            stream.seek(start)
            payload = stream.read(budget)
        if start:
            marker = payload.find(b"\n")
            if marker < 0:
                return []
            start += marker + 1
            payload = payload[marker + 1:]
        header = _ACCESS_HEADER if ref.stream == "ACCESS" else _RUNTIME_HEADER
        records: list[tuple[int, re.Match[str] | None, dict | None, list[str]]] = []
        offset = start
        for raw_line in payload.splitlines(keepends=True):
            line = raw_line.rstrip(b"\r\n").decode("utf-8", errors="replace")
            line_offset = offset
            offset += len(raw_line)
            json_row = self._parse_json_line(line)
            match = header.match(line)
            if json_row is not None:
                records.append((line_offset, None, json_row, [line]))
            elif match:
                records.append((line_offset, match, None, [line]))
            elif line.lstrip().startswith("{"):
                records.append((line_offset, None, None, [line]))
            elif records and records[-1][1] is not None:
                records[-1][3].append(line)
            elif line.strip():
                records.append((line_offset, None, None, [line]))
        return [self._event(ref, *record) for record in records]

    def _event(
        self, ref: LogFileRef, offset: int, match: re.Match[str] | None,
        json_row: dict | None, lines: list[str],
    ) -> dict[str, Any]:
        source = "\n".join(lines)
        if json_row is not None:
            timestamp = self._json_timestamp(json_row, ref)
            level_value = json_row.get("level", "INFO")
            level = str(
                level_value.get("name", "INFO")
                if isinstance(level_value, dict) else level_value
            ).upper()
            extra = json_row.get("extra") if isinstance(json_row.get("extra"), dict) else {}
            process = str(json_row.get("process") or extra.get("process") or "unknown")
            location = str(json_row.get("location") or json_row.get("name") or "json-log")
            message = str(json_row.get("message") or json_row.get("text") or "结构化日志事件")
            values = {**extra, **json_row}
        elif match is not None:
            timestamp = datetime.strptime(
                match.group("timestamp"), "%Y-%m-%d %H:%M:%S.%f"
            ).astimezone()
            level = match.group("level").strip().upper()
            process = match.group("process").strip()
            location = "api-access" if ref.stream == "ACCESS" else match.group("location").strip()
            message = match.group("message")
            values = {}
        else:
            timestamp = datetime.fromtimestamp(ref.modified_at, tz=timezone.utc).astimezone()
            lowered = source.casefold()
            level = "ERROR" if any(
                marker in lowered for marker in (
                    "traceback", "exception", "error", "failed", "失败",
                )
            ) else "INFO"
            process, location = "supervisor", "unstructured-output"
            message = next((line.strip() for line in lines if line.strip()), "非结构化日志输出")
            values = {}
        identifiers = {
            name: self._field_or_extract(values, name, pattern, source)
            for name, pattern in _IDENTIFIERS.items()
        }
        status = self._field_or_extract(values, "http_status", _HTTP_STATUS, source)
        duration = self._field_or_extract(values, "duration_ms", _DURATION, source)
        if json_row is not None:
            sanitized_raw = _redact_text(
                json.dumps(
                    redact_recursive(json_row, max_chars=self._max_field_chars),
                    ensure_ascii=False,
                ),
                max_chars=self._max_detail_chars,
            )
        else:
            sanitized_raw = redact_recursive(
                source, max_chars=self._max_detail_chars,
            )
        sanitized_message = redact_recursive(
            message, max_chars=self._max_field_chars,
        )
        event_id = _event_id(ref, offset, source)
        return {
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "level": level if level in _LEVELS else "INFO",
            "service_name": ref.service_name,
            "process": redact_recursive(process, max_chars=256),
            "stream": ref.stream,
            "location": redact_recursive(location, max_chars=512),
            "message": sanitized_message,
            **identifiers,
            "event_name": self._field_or_extract(values, "event_name", _EVENT_NAME, source),
            "http_status": int(status) if str(status or "").isdigit() else None,
            "duration_ms": float(duration) if _is_number(duration) else None,
            "has_traceback": len(lines) > 1 or "traceback" in source.casefold(),
            "source_file": ref.relative_path,
            "raw": sanitized_raw,
            "traceback": redact_recursive(
                "\n".join(lines[1:]), max_chars=self._max_detail_chars,
            ) if len(lines) > 1 else None,
            "structured": redact_recursive(values, max_chars=self._max_field_chars),
            "_search_text": source.casefold(),
        }

    @staticmethod
    def _field_or_extract(
        values: dict[str, Any], name: str, pattern: re.Pattern[str], source: str,
    ) -> Any:
        value = values.get(name)
        if value is not None:
            return str(value)
        found = pattern.search(source)
        return found.group(1) if found else None

    @staticmethod
    def _parse_json_line(line: str) -> dict[str, Any] | None:
        if not line.lstrip().startswith("{"):
            return None
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _json_timestamp(row: dict[str, Any], ref: LogFileRef) -> datetime:
        value = row.get("timestamp") or row.get("time") or row.get("created_at")
        if isinstance(value, dict):
            value = value.get("repr") or value.get("timestamp")
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone()
        except (TypeError, ValueError):
            return datetime.fromtimestamp(ref.modified_at, tz=timezone.utc).astimezone()

    def _window(
        self, started_at: datetime | None, ended_at: datetime | None,
    ) -> tuple[datetime, datetime]:
        end = _aware(ended_at or datetime.now(timezone.utc))
        start = _aware(started_at or (end - self._max_window))
        if end < start:
            raise LogQueryError("ended_at 不能早于 started_at")
        if end - start > self._max_window:
            raise LogQueryError("日志查询时间窗口超过配置上限")
        return start, end

    @staticmethod
    def _sort_key(event: dict[str, Any]) -> tuple[str, str]:
        return str(event["timestamp"]), str(event["event_id"])

    @staticmethod
    def _fingerprint(filters: dict[str, Any]) -> str:
        return sha256(json.dumps(filters, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _encode_cursor(
        key: tuple[str, str], fingerprint: str,
        started_at: datetime, ended_at: datetime,
    ) -> str:
        payload = json.dumps(
            {
                "v": 1,
                "key": key,
                "q": fingerprint,
                "start": started_at.isoformat(),
                "end": ended_at.isoformat(),
            },
            separators=(",", ":"),
        )
        return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(
        cursor: str, fingerprint: str,
    ) -> tuple[tuple[str, str], datetime, datetime]:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
            key = payload["key"]
            if payload.get("v") != 1 or payload.get("q") != fingerprint:
                raise ValueError
            if not isinstance(key, list) or len(key) != 2:
                raise ValueError
            started_at = datetime.fromisoformat(payload["start"])
            ended_at = datetime.fromisoformat(payload["end"])
            if started_at.tzinfo is None or ended_at.tzinfo is None:
                raise ValueError
            return (str(key[0]), str(key[1])), started_at, ended_at
        except Exception as exc:
            raise LogQueryError("日志查询游标无效或不属于当前筛选条件") from exc

    @staticmethod
    def _list_projection(event: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in event.items()
            if key not in {"raw", "traceback", "structured", "_search_text"}
        }


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone()


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return value is not None
    except (TypeError, ValueError):
        return False


def _redact_text(value: str, *, max_chars: int) -> str:
    result = _PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", value)
    result = _BEARER.sub("Bearer [REDACTED]", result)
    result = _JWT.sub("[REDACTED JWT]", result)
    result = _URL_CREDENTIAL.sub(r"\1[REDACTED]:[REDACTED]@", result)
    result = _ORACLE_CREDENTIAL.sub(r"[REDACTED]/[REDACTED]@\3", result)
    result = _KEY_VALUE_SECRET.sub(
        lambda match: f"{match.group(1)}=[REDACTED]", result,
    )
    if len(result) > max_chars:
        return result[:max_chars] + f"…[TRUNCATED {len(result) - max_chars} chars]"
    return result


def redact_recursive(
    value: Any, *, max_chars: int = 4096, key: str | None = None,
    depth: int = 0,
) -> Any:
    """递归处理结构化字段与字符串，阻断 Secret 和超大载荷泄漏。"""
    if depth > 12:
        return "[TRUNCATED DEPTH]"
    if key and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if key and _LARGE_KEY.search(key):
        size = len(value) if hasattr(value, "__len__") else 1
        return f"[TRUNCATED size={size}]"
    if isinstance(value, str):
        return _redact_text(value, max_chars=max_chars)
    if isinstance(value, dict):
        return {
            str(item_key): redact_recursive(
                item, max_chars=max_chars, key=str(item_key), depth=depth + 1,
            )
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 100:
            return f"[TRUNCATED items={len(value)}]"
        return [
            redact_recursive(item, max_chars=max_chars, depth=depth + 1)
            for item in value
        ]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _redact_text(str(value), max_chars=max_chars)


def _event_id(ref: LogFileRef, offset: int, source: str) -> str:
    content_hash = sha256(source.encode("utf-8", errors="replace")).hexdigest()
    return sha256(
        f"{ref.device}:{ref.inode}|{offset}|{content_hash}".encode()
    ).hexdigest()
