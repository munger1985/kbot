"""原生历史与日志的确定性证据投影，供回答及报告来源回放共用。"""

from aiops_agent.contracts.evidence import SourcePaginationObservation, source_pagination_warning


NATIVE_COLUMNS = (
    "observed_at", "metric_name", "value", "unit", "native_value_type", "sample_kind",
    "nanoseconds", "log_timestamp_seconds", "log_source", "log_severity", "log_event_id",
)


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
        ))
    return tuple(rows)


def log_pagination_warnings(result):
    receipts = []
    if result.provenance.get("source_pagination"):
        receipts.append(result.provenance["source_pagination"])
    for query in result.provenance.get("source_queries", ()):
        if query.get("provenance", {}).get("source_pagination"):
            receipts.append(query["provenance"]["source_pagination"])
    return tuple(source_pagination_warning(SourcePaginationObservation.model_validate(item)) for item in receipts)
