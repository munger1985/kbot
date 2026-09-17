"""标准监控指标的领域常量。"""

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
    "host.disk.io.utilization",
    "host.network.throughput",
)
