"""Top SQL 指标目录、输入合同与窗口指标计算，不将异构数据库指标混称。"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_ALL_DATABASES = ("ORACLE", "POSTGRESQL", "MYSQL")


@dataclass(frozen=True)
class SqlMetric:
    code: str
    label: str
    column: str
    unit: str
    databases: tuple[str, ...]
    counters: tuple[str, ...]
    scale: Decimal = Decimal(1)
    mean_of: str | None = None


SQL_METRICS = (
    SqlMetric("TOTAL_DURATION", "总耗时（秒）", "total_duration_seconds", "second", _ALL_DATABASES, ("@total_duration_microseconds",), Decimal("0.000001")),
    SqlMetric("MEAN_DURATION", "平均耗时（秒）", "mean_duration_seconds", "second/execution", _ALL_DATABASES, (), mean_of="TOTAL_DURATION"),
    SqlMetric("EXECUTION_COUNT", "执行次数", "execution_count", "execution", _ALL_DATABASES, ("@execution_count",)),
    SqlMetric("CPU_TIME", "CPU 消耗（秒）", "cpu_seconds", "second", ("ORACLE",), ("cpu_time",), Decimal("0.000001")),
    SqlMetric("MEAN_CPU_TIME", "每次 CPU（秒）", "mean_cpu_seconds", "second/execution", ("ORACLE",), (), mean_of="CPU_TIME"),
    SqlMetric("LOGICAL_READS", "逻辑读（块访问）", "buffer_gets", "oracle_block_access", ("ORACLE",), ("buffer_gets",)),
    SqlMetric("MEAN_LOGICAL_READS", "每次逻辑读（块访问）", "mean_buffer_gets", "oracle_block_access/execution", ("ORACLE",), (), mean_of="LOGICAL_READS"),
    SqlMetric("DISK_READS", "物理读（Oracle 块）", "disk_reads", "oracle_block", ("ORACLE",), ("disk_reads",)),
    SqlMetric("SHARED_BLOCK_ACCESSES", "共享块访问（命中＋读取）", "shared_block_accesses", "postgresql_block_access", ("POSTGRESQL",), ("shared_blks_hit", "shared_blks_read")),
    SqlMetric("SHARED_BLOCK_READS", "共享块读取（非磁盘读取保证）", "shared_blks_read", "postgresql_block", ("POSTGRESQL",), ("shared_blks_read",)),
    SqlMetric("TEMP_BLOCK_READS", "临时块读取", "temp_blks_read", "postgresql_block", ("POSTGRESQL",), ("temp_blks_read",)),
    SqlMetric("TEMP_BLOCK_WRITES", "临时块写入", "temp_blks_written", "postgresql_block", ("POSTGRESQL",), ("temp_blks_written",)),
    SqlMetric("ROWS_EXAMINED", "扫描行数", "rows_examined", "row", ("MYSQL",), ("rows_examined",)),
    SqlMetric("TEMP_DISK_TABLES", "临时磁盘表", "temp_disk_tables", "table", ("MYSQL",), ("temp_disk_tables",)),
    SqlMetric("LOCK_TIME", "锁等待（秒）", "lock_seconds", "second", ("MYSQL",), ("lock_time_picoseconds",), Decimal("0.000000000001")),
    SqlMetric("ROWS_PROCESSED", "处理／返回行数", "rows_processed", "row", _ALL_DATABASES, ("@rows_processed",)),
)
METRICS_BY_CODE = {metric.code: metric for metric in SQL_METRICS}


class TopSqlOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rank_metric: str = "TOTAL_DURATION"
    sort_direction: Literal["DESC", "ASC"] = "DESC"
    min_executions: int = Field(default=0, ge=0, le=2**63 - 1, strict=True)
    min_metric_value: Decimal | None = Field(default=None, ge=0, le=Decimal("1e30"), allow_inf_nan=False)
    min_cpu_seconds: Decimal | None = Field(default=None, ge=0, le=Decimal("1e30"), allow_inf_nan=False)
    min_logical_reads: int | None = Field(default=None, ge=0, le=2**63 - 1, strict=True)

    @field_validator("rank_metric")
    @classmethod
    def known_metric(cls, value: str) -> str:
        if value not in METRICS_BY_CODE:
            raise ValueError("不支持的 Top SQL 排名指标")
        return value

    @field_validator("min_metric_value", "min_cpu_seconds", "min_logical_reads", mode="before")
    @classmethod
    def no_boolean_threshold(cls, value):
        if isinstance(value, bool):
            raise ValueError("Top SQL 阈值必须为非负数，不能使用布尔值")
        return value

    def validate_database(self, database_type: str) -> None:
        requested = [self.rank_metric]
        if self.min_cpu_seconds is not None:
            requested.append("CPU_TIME")
        if self.min_logical_reads is not None:
            requested.append("LOGICAL_READS")
        for code in requested:
            if database_type not in METRICS_BY_CODE[code].databases:
                raise ValueError(f"当前数据库不支持指标：{METRICS_BY_CODE[code].label}")

    def frozen(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_defaults=True)

    def summary(self) -> str:
        conditions = []
        if self.min_executions:
            conditions.append(f"执行次数 ≥ {self.min_executions}")
        if self.min_metric_value is not None:
            conditions.append(f"所选指标 ≥ {self.min_metric_value}")
        if self.min_cpu_seconds is not None:
            conditions.append(f"CPU 消耗 ≥ {self.min_cpu_seconds} 秒")
        if self.min_logical_reads is not None:
            conditions.append(f"逻辑读 ≥ {self.min_logical_reads} 次块访问")
        direction = "降序" if self.sort_direction == "DESC" else "升序"
        return f"{METRICS_BY_CODE[self.rank_metric].label}，{direction}；筛选：" + ("；".join(conditions) or "无额外阈值")


OPTION_FIELDS = frozenset(TopSqlOptions.model_fields)


def metric_input_schema() -> tuple[dict[str, Any], ...]:
    return (
        {"name": "rank_metric", "label": "排序指标", "type": "select", "required": True,
         "default": "TOTAL_DURATION", "options": [
             {"value": metric.code, "label": metric.label, "supported_db_types": list(metric.databases)}
             for metric in SQL_METRICS],
         "help": "仅开放所选数据库支持的指标；缺少历史采集值时不可排名。"},
        {"name": "sort_direction", "label": "排序方向", "type": "select", "required": True,
         "default": "DESC", "options": [{"value": "DESC", "label": "降序（高资源优先）"}, {"value": "ASC", "label": "升序"}]},
        {"name": "min_executions", "label": "最低执行次数", "type": "integer", "required": False, "min": 0, "max": 2**63 - 1},
        {"name": "min_metric_value", "label": "所选指标最低值", "type": "number", "required": False, "min": 0,
         "help": "单位与排序指标一致；服务端先筛选，再截取 Top N。"},
        {"name": "min_cpu_seconds", "label": "最低 CPU 消耗（秒）", "type": "number", "required": False, "min": 0, "supported_db_types": ["ORACLE"]},
        {"name": "min_logical_reads", "label": "最低逻辑读（块访问）", "type": "integer", "required": False, "min": 0, "max": 2**63 - 1, "supported_db_types": ["ORACLE"]},
    )


def counter_value(statement, field: str) -> Decimal | None:
    raw = getattr(statement, field[1:], None) if field.startswith("@") else dict(
        getattr(statement, "database_metrics_json", None) or {}
    ).get(field)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() and value >= 0 else None


def metric_delta(metric: SqlMetric, before, after) -> tuple[Decimal | None, str | None]:
    values = []
    for field in metric.counters:
        left, right = counter_value(before, field), counter_value(after, field)
        if left is None or right is None:
            return None, "METRIC_NOT_COLLECTED"
        if right < left:
            return None, "METRIC_COUNTER_RESET"
        values.append(right - left)
    return sum(values, Decimal(0)) * metric.scale, None


def wire_decimal(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None

def finalize_metric_rows(rows, metrics, options: TopSqlOptions):
    """先聚合每个指标，再按阈值筛选和稳定排序；未知值不按零参与。"""
    ranked_metric = METRICS_BY_CODE[options.rank_metric]
    candidates = []
    for row in rows:
        values = row["metric_values"]
        for metric in metrics:
            if metric.mean_of:
                numerator = values.get(metric.mean_of)
                count = row["metric_executions"].get(metric.mean_of, 0)
                values[metric.code] = numerator / count if numerator is not None and count > 0 else None
                row["metric_gaps"][metric.code] = row["metric_gaps"].get(metric.mean_of, 0)
            value = values.get(metric.code)
            if metric.column != "execution_count":
                row[metric.column] = wire_decimal(value)
        row["metric_quality"] = {
            metric.code: "UNAVAILABLE" if values.get(metric.code) is None else
            "PARTIAL" if row["metric_gaps"].get(metric.code) else "OBSERVED"
            for metric in metrics
        }
        value = values.get(options.rank_metric)
        matches = (
            value is not None
            and row["execution_count"] >= options.min_executions
            and (options.min_metric_value is None or value >= options.min_metric_value)
            and (options.min_cpu_seconds is None or values.get("CPU_TIME") is not None and values["CPU_TIME"] >= options.min_cpu_seconds)
            and (options.min_logical_reads is None or values.get("LOGICAL_READS") is not None and values["LOGICAL_READS"] >= options.min_logical_reads)
        )
        if matches:
            candidates.append(row)
    direction = -1 if options.sort_direction == "DESC" else 1
    candidates.sort(key=lambda row: (
        direction * row["metric_values"][ranked_metric.code],
        row["statement_identity_type"], row["statement_identity_value"],
        str(row.get("schema_name", row["database_name"])), row["user_identifier"],
        str(row["top_level"]), str(row.get("plan_hash_value", "")),
        str(row.get("instance_key", "")),
    ))
    for row in rows:
        row["metric_values"] = {code: wire_decimal(value) for code, value in row["metric_values"].items()}
        row.pop("metric_executions", None)
    return candidates
