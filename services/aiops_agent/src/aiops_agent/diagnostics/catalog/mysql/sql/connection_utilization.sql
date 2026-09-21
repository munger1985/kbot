SELECT
    'connections' AS resource_name,
    CAST(connected.variable_value AS SIGNED) AS current_utilization,
    CAST(max_used.variable_value AS SIGNED) AS max_utilization,
    CAST(max_conn.variable_value AS CHAR) AS limit_value,
    ROUND(
        100 * CAST(connected.variable_value AS DECIMAL)
        / CAST(max_conn.variable_value AS DECIMAL),
        3
    ) AS utilization_percent
FROM performance_schema.global_status connected
INNER JOIN performance_schema.global_status max_used
    ON max_used.variable_name = 'Max_used_connections'
INNER JOIN performance_schema.global_variables max_conn
    ON max_conn.variable_name = 'max_connections'
WHERE connected.variable_name = 'Threads_connected'
