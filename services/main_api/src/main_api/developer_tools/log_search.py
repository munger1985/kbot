"""按逻辑服务读取 runtime/access 两类本地调试日志。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import re
from typing import Literal


LogType = Literal["RUNTIME", "ACCESS"]
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
_ERROR_ID = re.compile(r"\berror_id[=:]\s*([0-9a-fA-F-]{32,36})")
_REQUEST_ID = re.compile(r"\brequest_id[=:]\s*([^\s|,]+)")
_TRACE_ID = re.compile(r"\btrace_id[=:]\s*([^\s|,]+)")
_HTTP_STATUS = re.compile(
    r"(?:status|Status Code|状态码)[=:]\s*(\d{3})"
)
_DURATION = re.compile(
    r"(?:duration_ms|Processing Time|处理耗时|耗时)[=:]\s*"
    r"([\d.]+)(?:\s*ms)?"
)
_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


@dataclass(frozen=True)
class LogFileRef:
    service_name: str
    log_type: LogType
    path: Path
    relative_path: str
    size: int
    modified_at: float


class LogFileCatalog:
    """只发现 `logs/<service>/{runtime,access}.log*`。"""

    def __init__(self, root: Path):
        self._root = root.resolve()

    def files(self) -> list[LogFileRef]:
        rows: list[LogFileRef] = []
        if not self._root.is_dir():
            return rows
        for service_dir in self._root.iterdir():
            if service_dir.is_symlink() or not service_dir.is_dir():
                continue
            for log_type, filename in (
                ("RUNTIME", "runtime.log"),
                ("ACCESS", "access.log"),
            ):
                for path in service_dir.glob(f"{filename}*"):
                    if path.is_symlink() or not path.is_file():
                        continue
                    resolved = path.resolve()
                    if not resolved.is_relative_to(self._root):
                        continue
                    stat = resolved.stat()
                    rows.append(
                        LogFileRef(
                            service_name=service_dir.name,
                            log_type=log_type,
                            path=resolved,
                            relative_path=str(
                                resolved.relative_to(self._root)
                            ),
                            size=stat.st_size,
                            modified_at=stat.st_mtime,
                        )
                    )
        return sorted(
            rows,
            key=lambda row: (
                row.service_name,
                row.log_type,
                row.relative_path,
            ),
        )


class LocalLogSearchService:
    """在有界尾部窗口内查询指定服务的一类日志。"""

    def __init__(
        self,
        *,
        log_root: Path,
        max_bytes_per_file: int = 2 * 1024 * 1024,
    ):
        self._catalog = LogFileCatalog(log_root)
        self._max_bytes_per_file = max_bytes_per_file

    def services(self) -> list[dict]:
        grouped: dict[str, dict] = {}
        for row in self._catalog.files():
            item = grouped.setdefault(
                row.service_name,
                {
                    "service_name": row.service_name,
                    "runtime": None,
                    "access": None,
                },
            )
            key = row.log_type.lower()
            summary = item[key] or {
                "log_type": row.log_type,
                "files": 0,
                "size": 0,
                "modified_at": None,
            }
            summary["files"] += 1
            summary["size"] += row.size
            modified = datetime.fromtimestamp(
                row.modified_at
            ).astimezone().isoformat()
            if not summary["modified_at"] or modified > summary["modified_at"]:
                summary["modified_at"] = modified
            item[key] = summary
        return [grouped[key] for key in sorted(grouped)]

    def search(
        self,
        *,
        service_name: str,
        log_type: str,
        levels: set[str] | None = None,
        keyword: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        normalized_type = log_type.upper()
        if normalized_type not in {"RUNTIME", "ACCESS"}:
            raise ValueError("log_type 只能是 RUNTIME 或 ACCESS")
        normalized_levels = {
            value.upper() for value in (levels or _LEVELS)
        } & _LEVELS
        needle = str(keyword or "").strip().casefold()
        events: list[dict] = []
        for ref in self._catalog.files():
            if (
                ref.service_name != service_name
                or ref.log_type != normalized_type
            ):
                continue
            for event in self._read_events(ref):
                if event["level"] not in normalized_levels:
                    continue
                if needle and needle not in event["raw"].casefold():
                    continue
                events.append(event)
        events.sort(
            key=lambda item: (item["timestamp"], item["event_id"]),
            reverse=True,
        )
        return events[:limit]

    def _read_events(self, ref: LogFileRef) -> list[dict]:
        start = max(0, ref.size - self._max_bytes_per_file)
        with ref.path.open("rb") as stream:
            stream.seek(start)
            payload = stream.read(self._max_bytes_per_file)
        text = payload.decode("utf-8", errors="replace")
        if start:
            text = text.split("\n", 1)[-1]

        header = (
            _ACCESS_HEADER
            if ref.log_type == "ACCESS"
            else _RUNTIME_HEADER
        )
        rows: list[tuple[int, re.Match, list[str]]] = []
        leading_lines: list[str] = []
        offset = start
        for line in text.splitlines():
            encoded_size = len(line.encode("utf-8", errors="replace")) + 1
            match = header.match(line)
            if match:
                rows.append((offset, match, [line]))
            elif rows:
                rows[-1][2].append(line)
            else:
                leading_lines.append(line)
            offset += encoded_size

        events = [
            self._event(ref, event_offset, match, lines)
            for event_offset, match, lines in rows
        ]
        if leading_lines and any(line.strip() for line in leading_lines):
            events.append(
                self._unstructured_event(ref, start, leading_lines)
            )
        return events

    @staticmethod
    def _event(
        ref: LogFileRef,
        offset: int,
        match: re.Match,
        lines: list[str],
    ) -> dict:
        raw = "\n".join(lines)
        status = _extract(_HTTP_STATUS, raw)
        duration = _extract(_DURATION, raw)
        level = match.group("level").strip()
        return {
            "event_id": _event_id(ref, offset),
            "timestamp": datetime.strptime(
                match.group("timestamp"), "%Y-%m-%d %H:%M:%S.%f"
            ).astimezone().isoformat(),
            "level": level,
            "service_name": ref.service_name,
            "process": match.group("process").strip(),
            "log_type": ref.log_type,
            "location": (
                "api-access"
                if ref.log_type == "ACCESS"
                else match.group("location").strip()
            ),
            "message": match.group("message"),
            "error_id": _extract(_ERROR_ID, raw),
            "request_id": _extract(_REQUEST_ID, raw),
            "trace_id": _extract(_TRACE_ID, raw),
            "http_status": int(status) if status else None,
            "duration_ms": float(duration) if duration else None,
            "has_traceback": len(lines) > 1,
            "source_file": ref.relative_path,
            "raw": raw,
        }

    @staticmethod
    def _unstructured_event(
        ref: LogFileRef,
        offset: int,
        lines: list[str],
    ) -> dict:
        """保留日志系统初始化前的解释器和进程监督输出。"""

        raw = "\n".join(lines)
        lowered = raw.casefold()
        error_markers = (
            "traceback",
            "exception",
            "error",
            "failed",
            "启动失败",
            "失败",
        )
        level = (
            "ERROR"
            if any(marker in lowered for marker in error_markers)
            else "INFO"
        )
        return {
            "event_id": _event_id(ref, offset),
            "timestamp": datetime.fromtimestamp(
                ref.modified_at
            ).astimezone().isoformat(),
            "level": level,
            "service_name": ref.service_name,
            "process": "supervisor",
            "log_type": ref.log_type,
            "location": "process-output",
            "message": next(
                (line.strip() for line in lines if line.strip()),
                "非结构化日志输出",
            ),
            "error_id": _extract(_ERROR_ID, raw),
            "request_id": _extract(_REQUEST_ID, raw),
            "trace_id": _extract(_TRACE_ID, raw),
            "http_status": None,
            "duration_ms": None,
            "has_traceback": "traceback" in lowered or len(lines) > 1,
            "source_file": ref.relative_path,
            "raw": raw,
        }


def _extract(pattern: re.Pattern, value: str) -> str | None:
    found = pattern.search(value)
    return found.group(1) if found else None


def _event_id(ref: LogFileRef, offset: int) -> str:
    return sha256(
        (
            f"{ref.relative_path}|{ref.modified_at}|{offset}"
        ).encode("utf-8")
    ).hexdigest()
