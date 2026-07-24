SELECT
    trx_mysql_thread_id AS session_id,
    trx_started AS transaction_started_at,
    TIMESTAMPDIFF(SECOND, trx_started, UTC_TIMESTAMP()) AS elapsed_seconds,
    trx_state AS transaction_state,
    trx_rows_locked AS rows_locked,
    trx_rows_modified AS rows_modified
FROM information_schema.innodb_trx
WHERE TIMESTAMPDIFF(SECOND, trx_started, UTC_TIMESTAMP()) >= :min_seconds
ORDER BY elapsed_seconds DESC
