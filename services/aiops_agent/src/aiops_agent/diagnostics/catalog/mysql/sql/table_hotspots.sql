SELECT
    object_schema,
    object_name,
    count_read,
    count_write,
    sum_timer_read AS read_latency_picoseconds,
    sum_timer_write AS write_latency_picoseconds,
    count_fetch,
    count_insert,
    count_update,
    count_delete
FROM performance_schema.table_io_waits_summary_by_table
WHERE object_schema NOT IN ('mysql', 'performance_schema', 'information_schema', 'sys')
ORDER BY sum_timer_read + sum_timer_write DESC
LIMIT :limit
