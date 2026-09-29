SELECT
    wait_event_type,
    wait_event,
    COUNT(*) AS session_count,
    MAX(
        CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - query_start)) AS bigint)
    ) AS maximum_query_seconds
FROM pg_stat_activity
WHERE state = 'active'
  AND pid <> pg_backend_pid()
  AND wait_event_type IS NOT NULL
GROUP BY wait_event_type, wait_event
ORDER BY session_count DESC, maximum_query_seconds DESC, wait_event_type, wait_event
