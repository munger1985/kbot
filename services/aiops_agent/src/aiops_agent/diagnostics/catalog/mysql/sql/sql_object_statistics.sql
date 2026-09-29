SELECT
    object_schema,
    object_name,
    count_read,
    count_write,
    sum_timer_read AS read_latency_picoseconds,
    sum_timer_write AS write_latency_picoseconds
FROM performance_schema.table_io_waits_summary_by_table
WHERE (:schema_name = '' OR object_schema = :schema_name)
ORDER BY sum_timer_read + sum_timer_write DESC
LIMIT :limit
