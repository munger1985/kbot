"""空间趋势按数据库用量目标判定完成，辅助容量口径不替代物理上限。"""

from datetime import datetime
from decimal import Decimal

from aiops_agent.domain.evidence.observations import parse_numeric_sample


STORAGE_ANALYSIS_POLICY = {
    "scope": "趋势任务交付已用容量、变化量与增长速率；仅在用户明确要求时增加未来用量或耗尽时间目标。",
    "collection": "缺少相关容量口径时先尝试当前绑定和授权的只读来源；只有实际失败、无采样、无授权绑定或不能安全定位时才说明不可得，不要求用户猜挂载点或指标名。",
    "completion": "数据库用量时序满足请求窗口时，辅助使用率、上限或主机空间缺失仅限制对应判断，不阻断已完成的趋势分析。用户明确要求的耗尽时间仍须真实上限。",
    "capacity_semantics": "MySQL data_free/free_mb 是表空间内部可复用空间，不是文件系统可用容量；data_length+index_length/allocated_mb 是对象占用，不是磁盘总容量。InnoDB FILE_SIZE/ALLOCATED_SIZE 也不是磁盘上限，不要求表空间行数必然等于表数。",
    "time_to_limit": "仅引用同一来源、同一对象和相容口径的服务端 estimated_days_to_limit；字段为空则说明无法计算。禁止用 data_free、free_mb、当前已分配大小或逻辑库容量自行推算写满时间，也不能把数据库增长率与任意主机挂载点余量相除。",
    "recommendations": "建议不构成本轮实施目标；不为归档、分区或监控配置建议索取实施参数，不附带执行文档。用户明确要求方案或执行时另按冻结动作意图处理。",
    "host_resolution": "MySQL 本轮应采集 db.storage.locations，并沿当前 Agent 已授权系统的确认 RUNS_ON 关系采集主机指标。仅 storage_capacity_resolution.volumes 中已按 server_uuid、数据库路径和当前挂载映射匹配的卷可归因；其他文件系统不是本库承载盘。各卷分别引用其自己的使用率与增长预测，说明共享写入，不与逻辑库增长率混算。当前映射不证明历史路径一直不变。",
}


def storage_metric_scope(task_frame, monitoring):
    """只识别冻结标准目录中的角色与资源范围，不根据问题文字猜测。"""
    if task_frame.get("visualization_profile_id") != "storage.trend":
        return frozenset(), frozenset()
    primary, supporting = set(), set()
    for binding in monitoring.get("bindings", ()):
        for metric in binding.get("metrics", ()):
            code = metric.get("metric_code")
            if not code:
                continue
            role = metric.get("semantic_role")
            if metric.get("resource_scope", "DATABASE") == "DATABASE" and role in {"STORAGE_USED", "DATABASE_SIZE"}:
                primary.add(code)
            elif role in {"STORAGE_USED", "STORAGE_MAX", "STORAGE_FREE", "STORAGE_UTILIZATION", "STORAGE_LOCATION"}:
                supporting.add(code)
    return frozenset(primary), frozenset(supporting)


def storage_trend_ready(task_frame, monitoring, evidence):
    """辅助指标缺口不能否定有效的数据库容量时序，快照及事件零行不能代替时序。"""
    primary, _ = storage_metric_scope(task_frame, monitoring)
    for fact in evidence:
        if fact.tool_id != "metric.query_range" or fact.trust_level != "SOURCE_VERIFIED" or fact.truncated:
            continue
        names = [column.get("name") for column in fact.columns]
        for row in fact.rows:
            value = dict(zip(names, row))
            if value.get("metric_code") not in primary:
                continue
            coverage = parse_numeric_sample(value.get("coverage_ratio"))
            samples = parse_numeric_sample(value.get("sample_count"))
            if coverage is None or coverage < Decimal("0.8") or samples is None or samples < 2:
                continue
            if any(parse_numeric_sample(value.get(key)) is None for key in ("first", "latest", "change_per_day")):
                continue
            try:
                start = datetime.fromisoformat(str(value["window_start"]).replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(value["window_end"]).replace("Z", "+00:00"))
            except (KeyError, ValueError, TypeError):
                continue
            requested = task_frame.get("requested_window_seconds")
            if start.tzinfo is None or end.tzinfo is None or end <= start:
                continue
            if requested and (end - start).total_seconds() < requested * .95:
                continue
            return True
    return False


def supporting_storage_gap(gap, supporting_codes):
    """只降级明确指向辅助标准指标的缺口；全源失败和主用量缺失仍然阻断。"""
    return (
        gap.source_id == "monitoring.overview"
        and gap.step_id in supporting_codes
        and gap.code in {"SOURCE_NO_DATA", "SOURCE_QUERY_UNSUPPORTED", "SOURCE_TARGET_MAPPING_MISSING"}
    )
