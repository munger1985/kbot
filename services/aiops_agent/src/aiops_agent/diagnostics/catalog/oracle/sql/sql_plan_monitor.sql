SELECT
    sql_id,
    sql_exec_id,
    sql_exec_start,
    sql_plan_hash_value,
    plan_line_id,
    plan_operation,
    plan_options,
    plan_object_owner,
    plan_object_name,
    plan_cardinality,
    starts,
    output_rows,
    first_refresh_time,
    last_refresh_time,
    status
FROM (
    SELECT
        sql_id,
        sql_exec_id,
        sql_exec_start,
        sql_plan_hash_value,
        plan_line_id,
        plan_operation,
        plan_options,
        plan_object_owner,
        plan_object_name,
        plan_cardinality,
        starts,
        output_rows,
        first_refresh_time,
        last_refresh_time,
        status
    FROM gv$sql_plan_monitor
    WHERE sql_id = :sql_id
    ORDER BY sql_exec_start DESC, sql_exec_id DESC, plan_line_id
)
WHERE ROWNUM <= :limit
