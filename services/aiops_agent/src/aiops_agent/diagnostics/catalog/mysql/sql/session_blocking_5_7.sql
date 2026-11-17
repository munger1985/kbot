SELECT
    waiting.trx_mysql_thread_id AS waiting_session_id,
    waiting.trx_query AS waiting_statement,
    blocking.trx_mysql_thread_id AS blocking_session_id,
    blocking.trx_query AS blocking_statement,
    TIMESTAMPDIFF(SECOND, waiting.trx_wait_started, NOW()) AS wait_seconds
FROM information_schema.innodb_lock_waits AS waits
JOIN information_schema.innodb_trx AS waiting ON waiting.trx_id = waits.requesting_trx_id
JOIN information_schema.innodb_trx AS blocking ON blocking.trx_id = waits.blocking_trx_id
ORDER BY wait_seconds DESC
