SELECT *
FROM (
    SELECT
        w.inst_id AS instance_id,
        w.con_id AS container_id,
        w.sql_id,
        SUM(w.total_executions) AS total_executions,
        SUM(w.optimal_executions) AS optimal_executions,
        SUM(w.onepass_executions) AS onepass_executions,
        SUM(w.multipasses_executions) AS multipasses_executions,
        ROUND(MAX(NVL(w.last_tempseg_size, 0)) / 1048576, 2) AS last_temp_mb,
        ROUND(MAX(NVL(w.max_tempseg_size, 0)) / 1048576, 2) AS max_temp_mb,
        ROUND(MAX(w.active_time) / 100, 3) AS max_workarea_avg_active_seconds
    FROM gv$sql_workarea w
    WHERE w.sql_id IS NOT NULL
    GROUP BY w.inst_id, w.con_id, w.sql_id
    HAVING SUM(w.onepass_executions) > 0
        OR SUM(w.multipasses_executions) > 0
        OR MAX(NVL(w.max_tempseg_size, 0)) >= :min_temp_mb * 1048576
    ORDER BY multipasses_executions DESC, max_temp_mb DESC, w.sql_id
)
WHERE ROWNUM <= :limit
