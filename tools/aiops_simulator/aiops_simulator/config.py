"""模拟程序配置读取与校验。"""

from __future__ import annotations

import configparser
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SUPPORTED_ENGINES = ("oracle", "postgresql", "mysql")
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")


@dataclass(frozen=True)
class ScheduleSegment:
    """一天内连续的流量区间。"""

    start_minute: int
    end_minute: int
    operations_per_second: float


@dataclass(frozen=True)
class TrafficConfig:
    """单数据库的真实时间流量曲线。"""

    schedule: tuple[ScheduleSegment, ...]
    weekend_multiplier: float
    max_inflight: int

    def rate_at(self, minute_of_day: int, weekday: int) -> float:
        for segment in self.schedule:
            if segment.start_minute <= minute_of_day < segment.end_minute:
                multiplier = self.weekend_multiplier if weekday >= 5 else 1.0
                return segment.operations_per_second * multiplier
        raise RuntimeError("流量曲线没有覆盖当前时间")


@dataclass(frozen=True)
class DatabaseConfig:
    """单个外部演示数据库的连接配置。"""

    engine: str
    enabled: bool
    host: str
    port: int
    database: str
    username: str
    password: str
    schema: str
    min_pool_size: int
    max_pool_size: int
    traffic: TrafficConfig


@dataclass(frozen=True)
class SimulatorConfig:
    """完整运行配置。"""

    path: Path
    mode: str
    timezone_name: str
    timezone: ZoneInfo
    historical_year: int
    summary_interval_seconds: int
    statement_timeout_seconds: int
    retention_days: int
    cleanup_hour: int
    cleanup_batch_size: int
    random_seed: int
    databases: tuple[DatabaseConfig, ...]


def _required(parser: configparser.ConfigParser, section: str, option: str) -> str:
    if not parser.has_section(section):
        raise ValueError(f"缺少配置段[{section}]")
    value = parser.get(section, option, fallback="").strip()
    if not value or value == "CHANGE_ME":
        raise ValueError(f"[{section}] {option}未配置")
    return value


def _integer(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = parser.getint(section, option)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"[{section}] {option}必须在{minimum}到{maximum}之间"
        )
    return value


def _minute(value: str) -> int:
    if value == "24:00":
        return 24 * 60
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"时间格式无效：{value}")
    hour, minute = (int(item) for item in parts)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"时间超出范围：{value}")
    return hour * 60 + minute


def parse_schedule(raw: str) -> tuple[ScheduleSegment, ...]:
    """解析并验证完整覆盖24小时的流量曲线。"""

    segments: list[ScheduleSegment] = []
    for item in raw.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        try:
            period, rate_text = normalized.split("=", 1)
            start_text, end_text = period.split("-", 1)
            start = _minute(start_text.strip())
            end = _minute(end_text.strip())
            rate = float(rate_text.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"流量区间格式无效：{normalized}") from exc
        if start >= end:
            raise ValueError(f"流量区间起止时间无效：{normalized}")
        if not 0 <= rate <= 100:
            raise ValueError(f"每秒操作数必须在0到100之间：{normalized}")
        segments.append(ScheduleSegment(start, end, rate))

    segments.sort(key=lambda item: item.start_minute)
    if not segments or segments[0].start_minute != 0:
        raise ValueError("流量曲线必须从00:00开始")
    cursor = 0
    for segment in segments:
        if segment.start_minute != cursor:
            raise ValueError("流量曲线必须连续且不能重叠")
        cursor = segment.end_minute
    if cursor != 24 * 60:
        raise ValueError("流量曲线必须覆盖到24:00")
    return tuple(segments)


def _traffic(parser: configparser.ConfigParser, engine: str) -> TrafficConfig:
    section = f"traffic.{engine}"
    schedule = parse_schedule(_required(parser, section, "schedule"))
    weekend_multiplier = parser.getfloat(section, "weekend_multiplier")
    if not 0 <= weekend_multiplier <= 1:
        raise ValueError(f"[{section}] weekend_multiplier必须在0到1之间")
    max_inflight = _integer(
        parser, section, "max_inflight", minimum=1, maximum=32
    )
    return TrafficConfig(schedule, weekend_multiplier, max_inflight)


def _database(
    parser: configparser.ConfigParser, engine: str
) -> DatabaseConfig:
    section = f"database.{engine}"
    if not parser.has_section(section):
        raise ValueError(f"缺少配置段[{section}]")
    enabled = parser.getboolean(section, "enabled", fallback=False)
    if not enabled:
        return DatabaseConfig(
            engine=engine,
            enabled=False,
            host="",
            port=0,
            database="",
            username="",
            password="",
            schema="",
            min_pool_size=1,
            max_pool_size=1,
            traffic=_traffic(parser, engine),
        )

    default_port = {"oracle": 1521, "postgresql": 5432, "mysql": 3306}[engine]
    database_option = "service" if engine == "oracle" else "database"
    min_pool_size = _integer(
        parser, section, "min_pool_size", minimum=1, maximum=16
    )
    max_pool_size = _integer(
        parser, section, "max_pool_size", minimum=1, maximum=32
    )
    if min_pool_size > max_pool_size:
        raise ValueError(f"[{section}] min_pool_size不能大于max_pool_size")
    schema = _required(parser, section, "schema")
    if not IDENTIFIER.fullmatch(schema):
        raise ValueError(f"[{section}] schema不是安全的数据库标识符")
    return DatabaseConfig(
        engine=engine,
        enabled=True,
        host=_required(parser, section, "host"),
        port=parser.getint(section, "port", fallback=default_port),
        database=_required(parser, section, database_option),
        username=_required(parser, section, "username"),
        password=_required(parser, section, "password"),
        schema=schema,
        min_pool_size=min_pool_size,
        max_pool_size=max_pool_size,
        traffic=_traffic(parser, engine),
    )


def load_config(path: str | os.PathLike[str]) -> SimulatorConfig:
    """读取单一INI配置，并拒绝权限过宽或不完整的配置。"""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"配置文件不存在：{config_path}")
    file_mode = stat.S_IMODE(config_path.stat().st_mode)
    if file_mode & 0o077:
        raise ValueError("配置文件包含数据库密码，权限不得允许组或其他用户访问")

    parser = configparser.ConfigParser(interpolation=None)
    with config_path.open("r", encoding="utf-8") as stream:
        parser.read_file(stream)

    mode = _required(parser, "simulator", "mode").lower()
    if mode != "daily":
        raise ValueError("当前版本只实现daily日常流量模式")
    timezone_name = _required(parser, "simulator", "timezone")
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"时区不存在：{timezone_name}") from exc

    databases = tuple(_database(parser, engine) for engine in SUPPORTED_ENGINES)
    if not any(item.enabled for item in databases):
        raise ValueError("至少需要启用一个数据库")

    return SimulatorConfig(
        path=config_path,
        mode=mode,
        timezone_name=timezone_name,
        timezone=timezone,
        historical_year=_integer(
            parser, "simulator", "historical_year", minimum=2000, maximum=2100
        ),
        summary_interval_seconds=_integer(
            parser,
            "simulator",
            "summary_interval_seconds",
            minimum=10,
            maximum=3600,
        ),
        statement_timeout_seconds=_integer(
            parser,
            "simulator",
            "statement_timeout_seconds",
            minimum=1,
            maximum=300,
        ),
        retention_days=_integer(
            parser, "simulator", "retention_days", minimum=1, maximum=365
        ),
        cleanup_hour=_integer(
            parser, "simulator", "cleanup_hour", minimum=0, maximum=23
        ),
        cleanup_batch_size=_integer(
            parser,
            "simulator",
            "cleanup_batch_size",
            minimum=100,
            maximum=10000,
        ),
        random_seed=parser.getint("simulator", "random_seed"),
        databases=databases,
    )
