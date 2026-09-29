SELECT
    schema_name,
    digest,
    digest_text,
    count_star AS execution_count,
    sum_timer_wait AS total_latency_picoseconds,
    avg_timer_wait AS average_latency_picoseconds,
    max_timer_wait AS maximum_latency_picoseconds,
    sum_lock_time AS lock_time_picoseconds,
    sum_errors AS error_count,
    sum_warnings AS warning_count,
    sum_rows_examined AS rows_examined,
    sum_rows_sent AS rows_sent,
    sum_created_tmp_disk_tables AS temp_disk_tables,
    sum_sort_rows AS sort_rows
FROM performance_schema.events_statements_summary_by_digest
WHERE digest = :digest
ORDER BY sum_timer_wait DESC
LIMIT :limit
