SELECT
    pid,
    datid AS database_oid,
    datname AS database_name,
    relid AS relation_oid,
    phase,
    heap_blks_total,
    heap_blks_scanned,
    heap_blks_vacuumed,
    index_vacuum_count,
    max_dead_tuples,
    num_dead_tuples
FROM pg_stat_progress_vacuum
ORDER BY pid
