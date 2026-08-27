SELECT
    ranked.sql_id,
    ranked.plan_hash_value,
    ranked.executions,
    ranked.elapsed_seconds,
    ranked.cpu_seconds,
    ranked.buffer_gets,
    ranked.disk_reads,
    ranked.rows_processed,
    ranked.last_active_time
FROM (
    SELECT
        sql_id,
        plan_hash_value,
        executions,
        ROUND(elapsed_time / 1000000, 3) AS elapsed_seconds,
        ROUND(cpu_time / 1000000, 3) AS cpu_seconds,
        buffer_gets,
        disk_reads,
        rows_processed,
        last_active_time
    FROM v$sqlstats
    WHERE sql_id IS NOT NULL
      AND elapsed_time > 0
    ORDER BY elapsed_time DESC, sql_id
) ranked
WHERE ROWNUM <= :limit
