SELECT
    threads.processlist_id AS session_id,
    history.event_id,
    history.event_name,
    history.digest,
    history.digest_text,
    CAST(history.timer_wait / 1000000000000 AS DECIMAL(20, 6)) AS elapsed_seconds,
    history.rows_examined,
    history.rows_sent,
    history.errors,
    history.warnings
FROM performance_schema.events_statements_history_long history
LEFT JOIN performance_schema.threads threads
    ON threads.thread_id = history.thread_id
WHERE history.digest IS NOT NULL
ORDER BY history.event_id DESC
LIMIT :limit
