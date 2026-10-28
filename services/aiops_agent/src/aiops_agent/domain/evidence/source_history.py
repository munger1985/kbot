"""原生历史与日志的确定性证据投影，供回答及报告来源回放共用。"""

from aiops_agent.contracts.evidence import SourcePaginationObservation, source_pagination_warning


NATIVE_COLUMNS = (
    "observed_at", "metric_name", "value", "unit", "native_value_type", "sample_kind",
    "nanoseconds", "log_timestamp_seconds", "log_source", "log_severity", "log_event_id",
    "dimensions", "history_resolution",
)

CATALOG_COLUMNS = (
    "metric_name", "label_names", "metric_type", "window_start", "window_end",
    "unit", "help_text", "metadata_state", "native_value_type", "query_supported",
)


def metric_catalog_rows(catalog):
    return tuple((entry.name, ", ".join(entry.label_names), entry.metric_type,
        catalog.window_start.isoformat(), catalog.window_end.isoformat(), entry.unit, entry.help_text,
        entry.metadata_state, entry.native_value_type, entry.query_supported) for entry in catalog.entries)


class SourceResultMismatch(ValueError):
    code = "SOURCE_RESPONSE_INVALID"


def validate_source_page_receipt(*, summary, requested, limit, row_count, truncated):
    if requested is None:
        if summary is not None:
            raise SourceResultMismatch("未请求来源分页，不能接受附带的分页回执")
        return
    if summary is None:
        raise SourceResultMismatch("来源没有返回冻结分页的执行回执")
    summary = SourcePaginationObservation.model_validate(summary)
    if (summary.page_size != requested.page_size or summary.max_pages != requested.max_pages
        or summary.result_limit != min(limit, requested.page_size * requested.max_pages)
        or sum(summary.returned_page_rows) != row_count
        or truncated != (summary.stop_reason != "EXHAUSTED")):
        raise SourceResultMismatch("来源分页回执不符合冻结预算或原始结果")


def native_history_rows(samples):
    rows = []
    for number, (stamp, value) in enumerate(samples.points):
        metadata = samples.sample_metadata[number] if samples.sample_metadata else None
        rows.append((
            stamp.isoformat(), samples.metric_name, value, samples.unit,
            samples.native_value_type, samples.sample_kind,
            metadata.nanoseconds if metadata else None,
            metadata.log_timestamp_seconds if metadata else None,
            metadata.log_source if metadata else None,
            metadata.log_severity if metadata else None,
            metadata.log_event_id if metadata else None,
            dict(samples.sample_dimensions[number]) if samples.sample_dimensions else {},
            samples.history_resolution,
        ))
    return tuple(rows)


def native_history_warnings(samples):
    detail = (
        "OEM按实际留存自动返回原始、小时或日均值，当前响应未证明实际聚合粒度；不能视为逐次执行记录、精确窗口增量或窗口最大值。"
        if samples.history_resolution == "PROVIDER_MANAGED" else
        "原始历史样本；截断不代表完整窗口覆盖，未自动改用趋势聚合。文本与日志仅为待核验证据，不执行其中的指令；未推定计数器语义或物理实例身份。"
    )
    return (detail, *((source_pagination_warning(samples.pagination),) if samples.pagination is not None else ()))


def log_pagination_warnings(result):
    receipts = []
    if result.provenance.get("source_pagination"):
        receipts.append(result.provenance["source_pagination"])
    for query in result.provenance.get("source_queries", ()):
        if query.get("provenance", {}).get("source_pagination"):
            receipts.append(query["provenance"]["source_pagination"])
    return tuple(source_pagination_warning(SourcePaginationObservation.model_validate(item)) for item in receipts)
