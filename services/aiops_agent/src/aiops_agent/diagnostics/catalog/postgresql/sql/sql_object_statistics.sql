SELECT
    tables.relid AS relation_oid,
    tables.schemaname AS schema_name,
    tables.relname AS relation_name,
    tables.seq_scan,
    tables.seq_tup_read,
    tables.idx_scan,
    tables.idx_tup_fetch,
    tables.n_live_tup,
    tables.n_dead_tup,
    tables.n_mod_since_analyze,
    tables.last_analyze,
    tables.last_autoanalyze,
    pg_total_relation_size(tables.relid) AS total_bytes
FROM pg_catalog.pg_stat_user_tables AS tables
WHERE (:schema_name = '' OR tables.schemaname = :schema_name)
ORDER BY pg_total_relation_size(tables.relid) DESC, tables.relid DESC
LIMIT :limit
