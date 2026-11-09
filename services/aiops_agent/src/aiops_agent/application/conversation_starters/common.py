"""三引擎通用快捷入口的取证基线及输出合同。"""

COMMON_BASELINES = {
    "CURRENT_PERFORMANCE": {
        "ORACLE": ("db.instance.identity", "db.instance.performance", "db.session.active", "db.wait.class_summary"),
        "POSTGRESQL": ("db.instance.identity", "db.postgresql.instance.throughput", "db.session.active", "db.postgresql.connection.utilization", "db.postgresql.wait.current"),
        "MYSQL": ("db.instance.identity", "db.mysql.instance.throughput", "db.session.active", "db.mysql.connection.utilization", "db.wait.class_summary"),
    },
    "REPLICATION": {
        "ORACLE": ("db.instance.identity", "db.replication.status", "db.replication.lag"),
        "POSTGRESQL": ("db.instance.identity", "db.replication.status", "db.postgresql.replication.lag", "db.postgresql.wal.receiver", "db.postgresql.replication.replay", "db.postgresql.replication.slot_retention"),
        "MYSQL": ("db.instance.identity", "db.replication.status", "db.mysql.replication.channel_status", "db.mysql.replication.lag"),
    },
}

OUTPUT_SECTIONS = {
    "HEALTH_OVERVIEW": ("对象与时段", "负载、等待与会话", "容量、事务与复制", "当前告警", "诊断判断与必要处置"),
    "SESSIONS_BLOCKING": ("阻塞链与影响范围", "长事务与会话", "等待与锁", "诊断结论与必要处置"),
    "BACKUP_RECOVERY": ("备份记录与结果", "日志链与恢复能力", "恢复目标与演练", "检查结论"),
    "STORAGE_TREND": ("当前容量与口径", "逐日已用空间与增长", "预测结果（仅按请求）", "容量判断"),
    "CURRENT_PERFORMANCE": ("负载与吞吐", "会话与连接", "等待与阻塞", "诊断结论"),
    "REPLICATION": ("实例角色与复制拓扑", "发送、接收与应用状态", "延迟与日志积压", "复制错误与判断"),
    "PARAMETERS": ("内存与连接配置", "优化器与并行配置", "日志与持久性配置", "配置风险与建议"),
}

COMMON_CONSTRAINTS = (
    "按实例名称和观测时段描述事实并给出 DBA 判断；数值使用合理精度和单位，状态使用可读标签",
    "已成功采集且未发现慢 SQL、阻塞或长事务属于有效空结果，不因此标记部分完成",
    "缺少必需证据时先用授权的只读工具补采；仅在实际采集失败或能力不适用后说明原因，不猜测未知参数",
    "当前活动、累计统计与历史窗口增量必须分开解释；不能把累计等待量当作当前窗口等待",
    "复制零行需结合角色与发送、接收、应用事实区分未配置、不适用和未观测，不能直接判断复制健康",
    "MySQL 最近完成事务的提交至应用耗时不是当前复制延迟；正在应用事务年龄不是队列积压，空闲时不得用距最后提交时间推断延迟",
    "PostgreSQL 距最后回放时间仅表示回放新鲜度，空闲主库不代表复制故障；同时核对接收与回放 LSN 差",
    "参数保留原生名称，解释类别、单位与生效范围；未知业务约束不输出未经验证的参数修改命令",
)


def starter_measurement_semantics(tool_id: str, db_type: str) -> str:
    if tool_id in {"db.mysql.instance.throughput", "db.postgresql.instance.throughput"} or tool_id == "db.wait.class_summary":
        return "CUMULATIVE_SINCE_LOAD"
    return "CURRENT_ACTIVITY"
