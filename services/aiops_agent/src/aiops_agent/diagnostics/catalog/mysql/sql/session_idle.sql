SELECT
    plist.id AS session_id,
    plist.user AS username,
    plist.host AS client_host,
    plist.db AS database_name,
    plist.state AS session_state,
    plist.time AS idle_seconds,
    plist.info AS query_text
FROM information_schema.innodb_trx trx
INNER JOIN information_schema.processlist plist
    ON plist.id = trx.trx_mysql_thread_id
WHERE plist.command = 'Sleep'
ORDER BY plist.time DESC
