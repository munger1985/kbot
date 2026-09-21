SELECT
    sql_id,
    sql_exec_id,
    sql_exec_start,
    instance_id,
    status,
    username,
    elapsed_seconds,
    cpu_seconds,
    sql_plan_hash_value,
    last_refresh_time
FROM (
    SELECT
        sql_id,
        sql_exec_id,
        sql_exec_start,
        inst_id AS instance_id,
        status,
        username,
        ROUND(elapsed_time / 1000000, 3) AS elapsed_seconds,
        ROUND(cpu_time / 1000000, 3) AS cpu_seconds,
        sql_plan_hash_value,
        last_refresh_time
    FROM gv$sql_monitor
    WHERE sql_id IS NOT NULL
    ORDER BY last_refresh_time DESC NULLS LAST,
             sql_exec_start DESC NULLS LAST,
             sql_exec_id DESC
)
WHERE ROWNUM <= :limit
