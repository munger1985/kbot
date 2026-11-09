SELECT sources.* FROM (
SELECT statements.THREAD_ID AS thread_id, statements.EVENT_ID AS event_id,
       threads.PROCESSLIST_ID AS connection_id,
       statements.DIGEST AS digest, statements.CURRENT_SCHEMA AS schema_name,
       statements.DIGEST_TEXT AS source_sql,
       :source_sql_sha256 AS expected_source_hash,
       NULL AS sample_sql, NULL AS sample_digest, NULL AS sample_complete,
       NULL AS sample_captured_at, DATABASE() AS current_database
FROM performance_schema.events_statements_current AS statements
JOIN performance_schema.threads AS threads ON threads.THREAD_ID = statements.THREAD_ID
WHERE statements.DIGEST = :digest AND statements.CURRENT_SCHEMA = :schema_name
  AND statements.END_EVENT_ID IS NULL AND threads.PROCESSLIST_ID <> CONNECTION_ID()
UNION ALL
SELECT NULL, NULL, NULL, summary.DIGEST, summary.SCHEMA_NAME,
       summary.DIGEST_TEXT, :source_sql_sha256,
       summary.QUERY_SAMPLE_TEXT, STATEMENT_DIGEST(summary.QUERY_SAMPLE_TEXT),
       OCTET_LENGTH(summary.QUERY_SAMPLE_TEXT) < @@performance_schema_max_sql_text_length,
       summary.QUERY_SAMPLE_SEEN, DATABASE()
FROM performance_schema.events_statements_summary_by_digest AS summary
WHERE summary.DIGEST = :digest AND summary.SCHEMA_NAME = :schema_name
  AND summary.QUERY_SAMPLE_TEXT IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM performance_schema.events_statements_current AS active_statement
      JOIN performance_schema.threads AS active_thread ON active_thread.THREAD_ID = active_statement.THREAD_ID
      WHERE active_statement.DIGEST = :digest AND active_statement.CURRENT_SCHEMA = :schema_name
        AND active_statement.END_EVENT_ID IS NULL AND active_thread.PROCESSLIST_ID <> CONNECTION_ID()
  )
) AS sources
LIMIT 2
