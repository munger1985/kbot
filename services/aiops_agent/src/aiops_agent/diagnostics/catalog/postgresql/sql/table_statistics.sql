SELECT
    relid AS relation_oid,
    schemaname AS schema_name,
    relname AS table_name,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_tup_hot_upd,
    n_live_tup,
    n_dead_tup,
    n_mod_since_analyze,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze,
    vacuum_count,
    autovacuum_count,
    analyze_count,
    autoanalyze_count
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC, n_mod_since_analyze DESC, schemaname, relname
LIMIT :limit
