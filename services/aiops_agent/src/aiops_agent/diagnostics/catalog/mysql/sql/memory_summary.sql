SELECT
    event_name,
    current_count_used,
    current_number_of_bytes_used,
    high_count_used,
    high_number_of_bytes_used
FROM performance_schema.memory_summary_global_by_event_name
WHERE current_number_of_bytes_used > 0
ORDER BY current_number_of_bytes_used DESC
LIMIT :limit
