"""标准监控指标的领域常量。"""

METRIC_ROLE_AVAILABILITY = "AVAILABILITY"
METRIC_ROLE_STORAGE_UTILIZATION = "STORAGE_UTILIZATION"
METRIC_ROLE_STORAGE_USED = "STORAGE_USED"
METRIC_ROLE_STORAGE_MAX = "STORAGE_MAX"

DEFAULT_BASELINE_METRICS = (
    "db.availability",
    "db.cpu.utilization",
    "db.connection.active",
    "db.connection.utilization",
    "db.transaction.throughput",
    "db.response.latency",
    "db.storage.utilization",
    "db.storage.used_bytes",
    "db.storage.free_bytes",
    "db.storage.max_bytes",
    "db.error.rate",
    "host.cpu.utilization",
    "host.memory.utilization",
    "host.filesystem.utilization",
    "host.filesystem.used_bytes",
    "host.filesystem.available_bytes",
    "host.filesystem.size_bytes",
    "host.disk.io.utilization",
    "host.network.throughput",
    "runtime.cpu.utilization",
    "runtime.memory.utilization",
    "mysql.availability",
    "mysql.connection.utilization",
    "mysql.transaction.throughput",
    "mysql.sql.slow_query_rate",
    "mysql.lock.row_wait_rate",
    "mysql.innodb.buffer_pool_hit_percent",
    "postgresql.availability",
    "postgresql.connection.active",
    "postgresql.transaction.throughput",
    "postgresql.replication.lag_bytes",
    "postgresql.replication.slot_retained_bytes",
    "postgresql.database.size_bytes",
)
