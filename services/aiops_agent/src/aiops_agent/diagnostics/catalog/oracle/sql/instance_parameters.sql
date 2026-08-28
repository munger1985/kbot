SELECT
    name AS parameter_name,
    display_value AS parameter_value,
    isdefault AS is_default,
    ismodified AS modification_source,
    issys_modifiable AS runtime_modifiable
FROM v$parameter
WHERE name IN (
    'processes',
    'sessions',
    'open_cursors',
    'sga_target',
    'memory_target',
    'pga_aggregate_target',
    'db_recovery_file_dest_size',
    'archive_lag_target',
    'undo_retention',
    'fast_start_mttr_target',
    'statistics_level',
    'optimizer_mode',
    'parallel_degree_policy'
)
ORDER BY name
