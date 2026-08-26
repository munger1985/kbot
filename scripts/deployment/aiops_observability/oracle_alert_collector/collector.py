from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import oracledb


QUERY = """
SELECT *
FROM (
    SELECT
        ORIGINATING_TIMESTAMP,
        RECORD_ID,
        MESSAGE_TYPE,
        MESSAGE_LEVEL,
        MESSAGE_TEXT,
        PROBLEM_KEY,
        COMPONENT_ID,
        HOST_ID,
        CONTAINER_NAME,
        DATABASE_ID,
        SQL_ID,
        SESSION_ID
    FROM V$DIAG_ALERT_EXT
    WHERE ORIGINATING_TIMESTAMP > :last_timestamp
       OR (ORIGINATING_TIMESTAMP = :last_timestamp AND RECORD_ID > :last_record_id)
    ORDER BY ORIGINATING_TIMESTAMP, RECORD_ID
)
WHERE ROWNUM <= :max_rows
"""


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    service: str
    target_key: str
    poll_seconds: int
    initial_lookback_seconds: int
    max_rows: int
    username_file: Path
    password_file: Path
    output_file: Path
    checkpoint_file: Path
    health_file: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        data_dir = Path("/var/lib/kbot/oracle-alert")
        settings = cls(
            host=_required("ORACLE_HOST"),
            port=_bounded_int("ORACLE_PORT", 1521, 1, 65535),
            service=_required("ORACLE_SERVICE"),
            target_key=_required("ORACLE_TARGET_KEY"),
            poll_seconds=_bounded_int("ORACLE_POLL_SECONDS", 15, 5, 3600),
            initial_lookback_seconds=_bounded_int(
                "ORACLE_INITIAL_LOOKBACK_SECONDS", 900, 0, 86400
            ),
            max_rows=_bounded_int("ORACLE_MAX_ROWS", 1000, 1, 5000),
            username_file=Path("/run/secrets/oracle_username"),
            password_file=Path("/run/secrets/oracle_password"),
            output_file=data_dir / "alert.jsonl",
            checkpoint_file=data_dir / "checkpoint.json",
            health_file=data_dir / "health.json",
        )
        if not settings.target_key.replace("-", "").replace("_", "").isalnum():
            raise ValueError("ORACLE_TARGET_KEY只能包含字母、数字、连字符和下划线")
        return settings


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"缺少必要配置：{name}")
    return value


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name}必须介于{minimum}和{maximum}之间")
    return value


def _read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").rstrip("\r\n")
    if not value:
        raise ValueError(f"Secret为空：{path.name}")
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else str(value)
    return value


def _initial_checkpoint(settings: Settings) -> tuple[datetime, int]:
    return (
        datetime.now(timezone.utc)
        - timedelta(seconds=settings.initial_lookback_seconds),
        -1,
    )


def _load_checkpoint(settings: Settings) -> tuple[datetime, int]:
    if not settings.checkpoint_file.exists():
        return _initial_checkpoint(settings)
    payload = json.loads(settings.checkpoint_file.read_text(encoding="utf-8"))
    timestamp = datetime.fromisoformat(payload["originating_timestamp"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp, int(payload["record_id"])


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_health(settings: Settings, status: str, detail: str = "") -> None:
    _atomic_json_write(
        settings.health_file,
        {
            "status": status,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "detail": detail[:500],
        },
    )


def _append_rows(
    settings: Settings,
    columns: list[str],
    rows: list[tuple[Any, ...]],
) -> tuple[datetime, int] | None:
    if not rows:
        return None
    with settings.output_file.open("a", encoding="utf-8") as stream:
        for row in rows:
            payload = {
                key.lower(): _json_value(value)
                for key, value in zip(columns, row, strict=True)
            }
            payload.update(
                {
                    "schema": "ORACLE_ALERT_LOG.v1",
                    "source": "V$DIAG_ALERT_EXT",
                    "target_key": settings.target_key,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            stream.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    last = dict(zip(columns, rows[-1], strict=True))
    return last["ORIGINATING_TIMESTAMP"], int(last["RECORD_ID"])


def _collect_once(
    connection: oracledb.Connection,
    settings: Settings,
    checkpoint: tuple[datetime, int],
) -> tuple[datetime, int]:
    with connection.cursor() as cursor:
        cursor.execute(
            QUERY,
            last_timestamp=checkpoint[0],
            last_record_id=checkpoint[1],
            max_rows=settings.max_rows,
        )
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
    next_checkpoint = _append_rows(settings, columns, rows)
    if next_checkpoint is None:
        return checkpoint
    _atomic_json_write(
        settings.checkpoint_file,
        {
            "originating_timestamp": next_checkpoint[0].isoformat(),
            "record_id": next_checkpoint[1],
        },
    )
    return next_checkpoint


def run(settings: Settings) -> None:
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    settings.output_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint(settings)
    username = _read_secret(settings.username_file)
    password = _read_secret(settings.password_file)
    dsn = oracledb.makedsn(settings.host, settings.port, service_name=settings.service)

    while not stopping:
        try:
            with oracledb.connect(user=username, password=password, dsn=dsn) as connection:
                checkpoint = _collect_once(connection, settings, checkpoint)
            _write_health(settings, "healthy")
        except Exception as exc:  # noqa: BLE001
            # 日志不得输出连接串或凭据，仅保留驱动错误类型和受限错误文本。
            detail = str(exc).replace(username, "***").replace(password, "***")
            detail = f"{type(exc).__name__}: {detail}"
            _write_health(settings, "unhealthy", detail)
            print(f"Oracle Alert Log采集失败：{detail}", flush=True)
        deadline = time.monotonic() + settings.poll_seconds
        while not stopping and time.monotonic() < deadline:
            time.sleep(0.25)


if __name__ == "__main__":
    run(Settings.from_environment())
