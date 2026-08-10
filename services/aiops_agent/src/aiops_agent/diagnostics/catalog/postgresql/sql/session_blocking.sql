SELECT
    waiting.pid AS waiting_session_id,
    waiting.query AS waiting_statement,
    blocking.pid AS blocking_session_id,
    blocking.query AS blocking_statement,
    CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - waiting.query_start)) AS bigint) AS wait_seconds
FROM pg_stat_activity waiting
CROSS JOIN LATERAL unnest(pg_blocking_pids(waiting.pid)) AS blocker(pid)
JOIN pg_stat_activity blocking ON blocking.pid = blocker.pid
ORDER BY wait_seconds DESC
