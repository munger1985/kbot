SELECT statements.THREAD_ID AS thread_id, statements.EVENT_ID AS event_id,
       threads.PROCESSLIST_ID AS connection_id,
       statements.DIGEST AS digest, statements.CURRENT_SCHEMA AS schema_name,
       statements.DIGEST_TEXT AS source_sql,
       :source_sql_sha256 AS expected_source_hash
FROM performance_schema.events_statements_current AS statements
JOIN performance_schema.threads AS threads ON threads.THREAD_ID = statements.THREAD_ID
WHERE statements.DIGEST = :digest AND statements.CURRENT_SCHEMA = :schema_name
  AND statements.END_EVENT_ID IS NULL AND threads.PROCESSLIST_ID <> CONNECTION_ID()
ORDER BY statements.THREAD_ID, statements.EVENT_ID
LIMIT 2
