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
    max_dead_tuple_bytes,
    dead_tuple_bytes,
    num_dead_item_ids,
    indexes_total,
    indexes_processed
FROM pg_stat_progress_vacuum
ORDER BY pid
