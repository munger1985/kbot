SELECT
    MAX(CASE WHEN variable_name = 'log_bin' THEN variable_value END) AS log_bin,
    MAX(CASE WHEN variable_name = 'binlog_format' THEN variable_value END) AS binlog_format,
    MAX(CASE WHEN variable_name = 'gtid_mode' THEN variable_value END) AS gtid_mode,
    MAX(CASE WHEN variable_name = 'binlog_expire_logs_seconds' THEN variable_value END) AS binlog_expire_logs_seconds,
    MAX(CASE WHEN variable_name = 'sync_binlog' THEN variable_value END) AS sync_binlog
FROM performance_schema.global_variables
WHERE variable_name IN (
    'log_bin', 'binlog_format', 'gtid_mode',
    'binlog_expire_logs_seconds', 'sync_binlog'
)
