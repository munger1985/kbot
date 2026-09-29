SELECT
    object_schema,
    object_name,
    index_name,
    count_read,
    count_write,
    sum_timer_read AS read_latency_picoseconds,
    sum_timer_write AS write_latency_picoseconds
FROM performance_schema.table_io_waits_summary_by_index_usage
WHERE object_schema NOT IN ('mysql', 'performance_schema', 'information_schema', 'sys')
ORDER BY count_read ASC, count_write DESC
LIMIT :limit
