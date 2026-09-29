SELECT
    variable_name,
    variable_value
FROM performance_schema.global_variables
WHERE variable_name IN (
    'max_connections', 'innodb_buffer_pool_size', 'innodb_flush_log_at_trx_commit',
    'sync_binlog', 'tmp_table_size', 'max_heap_table_size',
    'optimizer_switch', 'sql_mode', 'transaction_isolation'
)
ORDER BY variable_name
