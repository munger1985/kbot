SELECT CAST(statements.queryid AS text) AS query_id,
       statements.dbid AS database_oid,
       statements.userid AS user_oid,
       statements.toplevel AS top_level,
       statements.query AS source_sql,
       CAST(:source_sql_sha256 AS text) AS expected_source_hash
FROM pg_stat_statements AS statements
WHERE statements.queryid = CAST(CAST(:query_id AS text) AS bigint)
  AND statements.dbid = :database_oid
  AND statements.userid = :user_oid
  AND statements.toplevel = :top_level
LIMIT 2
