WITH sql_stats AS (
    SELECT con_id, sql_id, MAX(sql_text) AS sql_text,
           SUM(executions) AS executions, SUM(elapsed_time) AS elapsed_time,
           SUM(cpu_time) AS cpu_time, SUM(user_io_wait_time) AS user_io_wait_time,
           SUM(application_wait_time) AS application_wait_time, SUM(disk_reads) AS disk_reads
    FROM v$sqlstats
    WHERE sql_id IS NOT NULL AND executions > 0
    GROUP BY con_id, sql_id
), workareas AS (
    SELECT con_id, sql_id,
           SUM(total_executions) AS workarea_total_executions,
           SUM(multipasses_executions) AS workarea_multipass_executions,
           100.0 * SUM(multipasses_executions) / NULLIF(SUM(total_executions), 0) AS workarea_multipass_percent,
           MAX(NVL(max_tempseg_size, 0)) / 1048576.0 AS maximum_workarea_temp_mb
    FROM v$sql_workarea
    WHERE sql_id IS NOT NULL
    GROUP BY con_id, sql_id
), metrics AS (
    SELECT TO_CHAR(s.con_id) || ':' || s.sql_id AS sql_key,
           CAST(NULL AS VARCHAR2(128)) AS schema_name, s.sql_text, 'CURSOR' AS text_kind,
           s.executions AS execution_count, s.elapsed_time / 1000 AS total_elapsed_ms,
           s.elapsed_time / 1000 / NULLIF(s.executions, 0) AS average_elapsed_ms,
           CAST(NULL AS NUMBER) AS maximum_elapsed_ms,
           s.cpu_time / 1000 / NULLIF(s.executions, 0) AS average_cpu_ms,
           s.user_io_wait_time / 1000 / NULLIF(s.executions, 0) AS average_io_ms,
           s.application_wait_time / 1000 / NULLIF(s.executions, 0) AS average_lock_ms,
           s.disk_reads / NULLIF(s.executions, 0) AS average_reads,
           CAST(NULL AS NUMBER) AS average_rows_examined, CAST(NULL AS NUMBER) AS average_spills,
           w.workarea_total_executions, w.workarea_multipass_executions,
           w.workarea_multipass_percent, w.maximum_workarea_temp_mb
    FROM sql_stats s LEFT JOIN workareas w ON w.con_id = s.con_id AND w.sql_id = s.sql_id
), qualified AS (
    SELECT * FROM metrics
    WHERE (average_elapsed_ms >= :average_elapsed_ms_limit)
       OR (average_cpu_ms >= :average_cpu_ms_limit)
       OR (average_io_ms >= :average_io_ms_limit)
       OR (average_lock_ms >= :average_lock_ms_limit)
       OR (average_reads >= :average_reads_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
       OR (average_rows_examined >= :average_rows_examined_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
       OR (average_spills >= :average_spills_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
       OR (workarea_multipass_percent >= :workarea_multipass_percent_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
       OR (maximum_workarea_temp_mb >= :maximum_workarea_temp_mb_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
), collection AS (
    SELECT 'AVAILABLE' AS collection_status FROM dual
)
SELECT qualified.*, collection.collection_status
FROM collection LEFT JOIN qualified ON 1 = 1
ORDER BY qualified.sql_key
