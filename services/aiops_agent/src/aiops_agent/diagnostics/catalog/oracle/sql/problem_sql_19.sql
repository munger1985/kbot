WITH metrics AS (
SELECT sql_id || ':' || TO_CHAR(plan_hash_value) AS sql_key,
       CAST(NULL AS VARCHAR2(128)) AS schema_name, sql_text, 'CURSOR' AS text_kind,
       executions AS execution_count, elapsed_time / 1000 AS total_elapsed_ms,
       elapsed_time / 1000 / NULLIF(executions, 0) AS average_elapsed_ms,
       CAST(NULL AS NUMBER) AS maximum_elapsed_ms,
       cpu_time / 1000 / NULLIF(executions, 0) AS average_cpu_ms,
       user_io_wait_time / 1000 / NULLIF(executions, 0) AS average_io_ms,
       application_wait_time / 1000 / NULLIF(executions, 0) AS average_lock_ms,
       disk_reads / NULLIF(executions, 0) AS average_reads,
       CAST(NULL AS NUMBER) AS average_rows_examined, CAST(NULL AS NUMBER) AS average_spills
FROM v$sqlstats
WHERE sql_id IS NOT NULL AND executions > 0
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
SELECT 'AVAILABLE' AS collection_status FROM dual
)
SELECT qualified.*, collection.collection_status
FROM collection LEFT JOIN qualified ON 1 = 1
ORDER BY qualified.sql_key
