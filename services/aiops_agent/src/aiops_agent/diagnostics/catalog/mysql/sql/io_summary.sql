SELECT
    event_name,
    count_read,
    count_write,
    sum_number_of_bytes_read AS bytes_read,
    sum_number_of_bytes_write AS bytes_written,
    sum_timer_read AS read_latency_picoseconds,
    sum_timer_write AS write_latency_picoseconds
FROM performance_schema.file_summary_by_event_name
WHERE count_read + count_write > 0
ORDER BY sum_timer_read + sum_timer_write DESC
LIMIT :limit
