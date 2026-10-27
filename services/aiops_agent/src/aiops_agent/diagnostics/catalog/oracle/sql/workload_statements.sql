SELECT
    ranked.sql_id,
    ranked.sql_text,
    ranked.executions,
    ranked.elapsed_time,
    ranked.rows_processed,
    ranked.plan_hash_value
FROM (
    SELECT
        sql_id,
        sql_text,
        executions,
        elapsed_time,
        rows_processed,
        plan_hash_value
    FROM v$sqlstats
    WHERE con_id = TO_NUMBER(SYS_CONTEXT('USERENV', 'CON_ID'))
      AND sql_id IS NOT NULL
      AND elapsed_time >= 0
    ORDER BY elapsed_time DESC, sql_id
) ranked
WHERE ROWNUM <= :limit
