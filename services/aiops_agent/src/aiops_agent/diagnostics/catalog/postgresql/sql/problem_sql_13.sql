WITH metrics AS (
SELECT CAST(dbid AS text) || ':' || CAST(userid AS text) || ':' || CAST(queryid AS text) AS sql_key,
       current_database() AS schema_name, query AS sql_text, 'NORMALIZED' AS text_kind,
       calls AS execution_count, total_exec_time AS total_elapsed_ms, mean_exec_time AS average_elapsed_ms,
       max_exec_time AS maximum_elapsed_ms, CAST(NULL AS numeric) AS average_cpu_ms,
       CASE WHEN current_setting('track_io_timing') = 'on' OR (blk_read_time + blk_write_time) > 0
            THEN (blk_read_time + blk_write_time) / NULLIF(calls, 0) ELSE NULL END AS average_io_ms,
       CAST(NULL AS numeric) AS average_lock_ms,
       (shared_blks_read + local_blks_read) / CAST(NULLIF(calls, 0) AS numeric) AS average_reads,
       CAST(NULL AS numeric) AS average_rows_examined,
       (temp_blks_read + temp_blks_written) / CAST(NULLIF(calls, 0) AS numeric) AS average_spills
FROM pg_stat_statements
WHERE queryid IS NOT NULL AND calls > 0
  AND dbid = (SELECT oid FROM pg_catalog.pg_database WHERE datname = current_database())
), qualified AS (
    SELECT * FROM metrics
    WHERE (average_elapsed_ms >= :average_elapsed_ms_limit)
       OR (average_cpu_ms >= :average_cpu_ms_limit)
       OR (average_io_ms >= :average_io_ms_limit)
       OR (average_lock_ms >= :average_lock_ms_limit)
       OR (average_reads >= :average_reads_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
       OR (average_rows_examined >= :average_rows_examined_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
       OR (average_spills >= :average_spills_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
), collection AS (
SELECT CASE WHEN current_setting('pg_stat_statements.track', true) IN ('top', 'all')
            THEN 'AVAILABLE' ELSE 'DISABLED' END AS collection_status
)
SELECT qualified.*, collection.collection_status
FROM collection LEFT JOIN qualified ON 1 = 1
ORDER BY qualified.sql_key
