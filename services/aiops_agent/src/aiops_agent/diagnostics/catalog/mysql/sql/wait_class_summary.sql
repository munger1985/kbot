SELECT
    SUBSTRING_INDEX(event_name, '/', 2) AS wait_class,
    COUNT(*) AS event_count,
    SUM(count_star) AS wait_count,
    SUM(sum_timer_wait) AS total_wait_picoseconds,
    MAX(max_timer_wait) AS max_wait_picoseconds
FROM performance_schema.events_waits_summary_global_by_event_name
WHERE count_star > 0
GROUP BY SUBSTRING_INDEX(event_name, '/', 2)
ORDER BY total_wait_picoseconds DESC
LIMIT :limit
