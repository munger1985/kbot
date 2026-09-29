SELECT
    threads.processlist_id AS session_id,
    threads.processlist_user AS username,
    threads.processlist_db AS database_name,
    statements.digest,
    statements.digest_text,
    CAST(statements.timer_wait / 1000000000000 AS DECIMAL(20, 6)) AS elapsed_seconds,
    statements.rows_examined,
    statements.rows_sent
FROM performance_schema.events_statements_current statements
INNER JOIN performance_schema.threads threads
    ON threads.thread_id = statements.thread_id
WHERE statements.sql_text IS NOT NULL
ORDER BY statements.timer_wait DESC
LIMIT :limit
