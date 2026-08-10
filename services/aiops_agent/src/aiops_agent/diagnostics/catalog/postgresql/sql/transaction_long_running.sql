SELECT
    pid AS session_id,
    xact_start AS transaction_started_at,
    CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - xact_start)) AS bigint) AS elapsed_seconds,
    state AS transaction_state,
    CAST(0 AS bigint) AS rows_locked,
    CAST(0 AS bigint) AS rows_modified
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
  AND EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - xact_start)) >= :min_seconds
ORDER BY elapsed_seconds DESC
