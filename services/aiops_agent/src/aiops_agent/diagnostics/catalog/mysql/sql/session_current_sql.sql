SELECT
    threads.processlist_id AS session_id,
    threads.processlist_user AS username,
    threads.processlist_host AS client_host,
    threads.processlist_db AS database_name,
    statements.digest,
    statements.digest_text,
    threads.processlist_state AS session_state,
    CAST(statements.timer_wait / 1000000000000 AS DECIMAL(20, 6)) AS elapsed_seconds
FROM performance_schema.threads threads
LEFT JOIN performance_schema.events_statements_current statements
    ON statements.thread_id = threads.thread_id
WHERE threads.processlist_id = :session_id
LIMIT 1
