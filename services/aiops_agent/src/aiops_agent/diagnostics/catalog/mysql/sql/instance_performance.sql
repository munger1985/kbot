SELECT
    @@server_uuid AS server_uuid,
    CAST(MAX(CASE WHEN variable_name = 'Uptime' THEN variable_value END) AS SIGNED) AS uptime_seconds,
    CAST(MAX(CASE WHEN variable_name = 'Questions' THEN variable_value END) AS SIGNED) AS questions,
    CAST(MAX(CASE WHEN variable_name = 'Com_commit' THEN variable_value END) AS SIGNED) AS commits,
    CAST(MAX(CASE WHEN variable_name = 'Com_rollback' THEN variable_value END) AS SIGNED) AS rollbacks,
    CAST(MAX(CASE WHEN variable_name = 'Threads_connected' THEN variable_value END) AS SIGNED) AS threads_connected,
    CAST(MAX(CASE WHEN variable_name = 'Threads_running' THEN variable_value END) AS SIGNED) AS threads_running,
    CAST(MAX(CASE WHEN variable_name = 'Created_tmp_disk_tables' THEN variable_value END) AS SIGNED) AS temp_disk_tables,
    CAST(MAX(CASE WHEN variable_name = 'Select_scan' THEN variable_value END) AS SIGNED) AS full_table_scans
FROM performance_schema.global_status
WHERE variable_name IN (
    'Uptime', 'Questions', 'Com_commit', 'Com_rollback',
    'Threads_connected', 'Threads_running',
    'Created_tmp_disk_tables', 'Select_scan'
)
