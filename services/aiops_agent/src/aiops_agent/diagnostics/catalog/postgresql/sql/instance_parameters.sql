SELECT
    name,
    setting,
    NULLIF(unit, '') AS unit,
    context,
    source,
    pending_restart
FROM pg_settings
WHERE name IN (
    'archive_mode',
    'autovacuum',
    'autovacuum_analyze_scale_factor',
    'autovacuum_freeze_max_age',
    'autovacuum_max_workers',
    'autovacuum_naptime',
    'autovacuum_vacuum_scale_factor',
    'checkpoint_completion_target',
    'checkpoint_timeout',
    'effective_cache_size',
    'effective_io_concurrency',
    'jit',
    'log_min_duration_statement',
    'maintenance_work_mem',
    'max_connections',
    'max_parallel_workers',
    'max_parallel_workers_per_gather',
    'max_wal_size',
    'max_worker_processes',
    'min_wal_size',
    'random_page_cost',
    'shared_buffers',
    'synchronous_commit',
    'track_io_timing',
    'track_wal_io_timing',
    'vacuum_freeze_table_age',
    'wal_buffers',
    'work_mem'
)
ORDER BY name
