SELECT
    stats.relid AS relation_oid,
    stats.indexrelid AS index_oid,
    stats.schemaname AS schema_name,
    stats.relname AS table_name,
    stats.indexrelname AS index_name,
    stats.idx_scan,
    stats.idx_tup_read,
    stats.idx_tup_fetch,
    pg_relation_size(stats.indexrelid) AS index_bytes,
    index_meta.indisunique AS is_unique,
    index_meta.indisvalid AS is_valid,
    index_meta.indisready AS is_ready,
    index_meta.indislive AS is_live
FROM pg_stat_user_indexes stats
JOIN pg_index index_meta
  ON index_meta.indexrelid = stats.indexrelid
ORDER BY index_bytes DESC, stats.schemaname, stats.indexrelname
LIMIT :limit
