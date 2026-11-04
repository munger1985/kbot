WITH tracked AS (
    SELECT value AS identity_key FROM jsonb_array_elements_text(CAST(:tracked_keys AS jsonb))
), excluded AS (
    SELECT value AS identity_key FROM jsonb_array_elements_text(CAST(:excluded_keys AS jsonb))
)
SELECT
    CAST(statements.queryid AS text) AS query_id,
    statements.dbid AS database_oid,
    database_name.datname AS database_name,
    statements.userid AS user_oid,
    role_name.rolname AS username,
    statements.toplevel AS top_level,
    statements.query AS normalized_statement,
    statements.plans AS plan_count,
    statements.total_plan_time AS total_plan_time_ms,
    statements.calls AS execution_count,
    statements.total_exec_time AS total_exec_time_ms,
    statements.mean_exec_time AS mean_exec_time_ms,
    statements.max_exec_time AS max_exec_time_ms,
    statements.rows AS rows_processed,
    statements.shared_blks_hit,
    statements.shared_blks_read,
    statements.shared_blks_dirtied,
    statements.shared_blks_written,
    statements.local_blks_hit,
    statements.local_blks_read,
    statements.local_blks_dirtied,
    statements.local_blks_written,
    statements.temp_blks_read,
    statements.temp_blks_written,
    statements.shared_blk_read_time
      + statements.local_blk_read_time AS block_read_time_ms,
    statements.shared_blk_write_time
      + statements.local_blk_write_time AS block_write_time_ms,
    statements.wal_records,
    statements.wal_fpi,
    statements.wal_bytes
FROM pg_stat_statements AS statements
LEFT JOIN pg_catalog.pg_database AS database_name
  ON database_name.oid = statements.dbid
LEFT JOIN pg_catalog.pg_roles AS role_name
  ON role_name.oid = statements.userid
WHERE statements.queryid IS NOT NULL
  AND statements.dbid = (SELECT oid FROM pg_catalog.pg_database WHERE datname = current_database())
  AND (CAST(statements.queryid AS text) || ':' || CAST(statements.dbid AS text) || ':' || CAST(statements.userid AS text) || ':' || CASE WHEN statements.toplevel THEN '1' ELSE '0' END IN (SELECT identity_key FROM tracked)
       OR CAST(statements.queryid AS text) || ':' || CAST(statements.dbid AS text) || ':' || CAST(statements.userid AS text) || ':' || CASE WHEN statements.toplevel THEN '1' ELSE '0' END NOT IN (SELECT identity_key FROM excluded))
ORDER BY CASE WHEN CAST(statements.queryid AS text) || ':' || CAST(statements.dbid AS text) || ':' || CAST(statements.userid AS text) || ':' || CASE WHEN statements.toplevel THEN '1' ELSE '0' END IN (SELECT identity_key FROM tracked) THEN 0 ELSE 1 END,
LEAST(
    ROW_NUMBER() OVER (ORDER BY statements.total_exec_time DESC, statements.queryid, statements.userid),
    ROW_NUMBER() OVER (ORDER BY statements.total_exec_time / NULLIF(statements.calls, 0) DESC NULLS LAST, statements.queryid, statements.userid),
    ROW_NUMBER() OVER (ORDER BY statements.calls DESC, statements.queryid, statements.userid),
    ROW_NUMBER() OVER (ORDER BY statements.rows DESC, statements.queryid, statements.userid),
    ROW_NUMBER() OVER (ORDER BY statements.shared_blks_hit + statements.shared_blks_read DESC, statements.queryid, statements.userid),
    ROW_NUMBER() OVER (ORDER BY statements.shared_blks_read DESC, statements.queryid, statements.userid),
    ROW_NUMBER() OVER (ORDER BY statements.temp_blks_read DESC, statements.queryid, statements.userid),
    ROW_NUMBER() OVER (ORDER BY statements.temp_blks_written DESC, statements.queryid, statements.userid)
), statements.total_exec_time DESC, statements.queryid DESC, statements.userid
LIMIT :limit
