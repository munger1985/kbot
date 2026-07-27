"""KBot 逻辑服务日志初始化与多进程安全文件 Sink。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import fcntl
import os
from pathlib import Path
import re
import sys
from typing import TextIO

from loguru import logger


_SIZE_PATTERN = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>B|KB|MB|GB)\s*$",
    re.IGNORECASE,
)
_RETENTION_PATTERN = re.compile(
    r"^\s*(?P<value>\d+)\s*(?P<unit>day|days|hour|hours)\s*$",
    re.IGNORECASE,
)
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _rotation_bytes(value: str) -> int:
    match = _SIZE_PATTERN.match(value)
    if not match:
        raise ValueError(f"日志轮转大小格式无效：{value}")
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return int(
        float(match.group("value")) * units[match.group("unit").upper()]
    )


def _retention_delta(value: str) -> timedelta:
    match = _RETENTION_PATTERN.match(value)
    if not match:
        raise ValueError(f"日志保留周期格式无效：{value}")
    amount = int(match.group("value"))
    if match.group("unit").lower().startswith("day"):
        return timedelta(days=amount)
    return timedelta(hours=amount)


class MultiprocessRotatingSink:
    """使用目录锁协调多个独立进程写入和轮转同一个日志文件。"""

    def __init__(self, path: Path, *, rotation: str, retention: str):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._rotation_bytes = _rotation_bytes(rotation)
        self._retention = _retention_delta(retention)

    def write(self, message: str) -> None:
        payload = message.encode("utf-8", errors="replace")
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX)
            self._rotate_if_needed(len(payload))
            with self.path.open("ab") as stream:
                stream.write(payload)
                stream.flush()
        finally:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
            os.close(directory_fd)

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if (
            self.path.exists()
            and self.path.stat().st_size + incoming_bytes
            > self._rotation_bytes
        ):
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            archive = self.path.with_name(
                f"{self.path.name}.{timestamp}"
            )
            self.path.replace(archive)
            self.path.touch()
            self._remove_expired_archives()

    def _remove_expired_archives(self) -> None:
        cutoff = datetime.now().timestamp() - self._retention.total_seconds()
        pattern = f"{self.path.name}.*"
        for archive in self.path.parent.glob(pattern):
            if archive.is_file() and archive.stat().st_mtime < cutoff:
                archive.unlink()


@dataclass(frozen=True)
class LogConfig:
    """单个进程写入所属逻辑服务日志的配置。"""

    service: str
    process: str
    log_dir: str = "var/log"
    level: str = "INFO"
    rotation: str = "100 MB"
    retention: str = "10 days"
    console_output: bool = field(
        default_factory=lambda: _environment_flag(
            "KBOT_LOG_CONSOLE", True
        )
    )

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.fullmatch(self.service):
            raise ValueError(f"日志服务标识无效：{self.service}")
        if not _NAME_PATTERN.fullmatch(self.process):
            raise ValueError(f"日志进程标识无效：{self.process}")


class LogManager:
    """为进程安装 runtime/access 两个互斥日志 Sink。"""

    def __init__(self, config: LogConfig):
        self.config = config
        self._runtime_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "[{extra[process]}] "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>{extra[correlation]}"
        )
        self._access_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "[{extra[process]}] {message}"
        )

    def setup(self) -> None:
        service_dir = (
            Path(self.config.log_dir or "var/log") / self.config.service
        ).resolve()
        runtime_sink = MultiprocessRotatingSink(
            service_dir / "runtime.log",
            rotation=self.config.rotation,
            retention=self.config.retention,
        )
        access_sink = MultiprocessRotatingSink(
            service_dir / "access.log",
            rotation=self.config.rotation,
            retention=self.config.retention,
        )

        logger.remove()
        logger.configure(
            extra={
                "service": self.config.service,
                "process": self.config.process,
                "log_type": "runtime",
                "correlation": "",
            }
        )
        logger.add(
            runtime_sink.write,
            level=self.config.level,
            format=self._runtime_format,
            enqueue=False,
            backtrace=True,
            diagnose=False,
            filter=lambda record: (
                record["extra"].get("log_type", "runtime") == "runtime"
            ),
        )
        logger.add(
            access_sink.write,
            level=self.config.level,
            format=self._access_format,
            enqueue=False,
            backtrace=False,
            diagnose=False,
            filter=lambda record: (
                record["extra"].get("log_type") == "access"
            ),
        )
        if self.config.console_output:
            self._add_console_handler(sys.stderr)

    def _add_console_handler(self, stream: TextIO) -> None:
        logger.add(
            stream,
            level=self.config.level,
            enqueue=False,
            backtrace=True,
            diagnose=False,
        )
