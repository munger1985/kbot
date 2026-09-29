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
    statements.blk_read_time AS block_read_time_ms,
    statements.blk_write_time AS block_write_time_ms,
    statements.wal_records,
    statements.wal_fpi,
    statements.wal_bytes
FROM pg_stat_statements AS statements
LEFT JOIN pg_catalog.pg_database AS database_name
  ON database_name.oid = statements.dbid
LEFT JOIN pg_catalog.pg_roles AS role_name
  ON role_name.oid = statements.userid
WHERE statements.queryid = CAST(CAST(:query_id AS text) AS bigint)
  AND statements.dbid = :database_oid
  AND (:user_oid = 0 OR statements.userid = :user_oid)
  AND statements.toplevel = :top_level
ORDER BY statements.total_exec_time DESC
LIMIT :limit
