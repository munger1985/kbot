SELECT
    relid AS relation_oid,
    schemaname AS schema_name,
    relname AS table_name,
    n_live_tup,
    n_dead_tup,
    n_mod_since_analyze,
    last_analyze,
    last_autoanalyze,
    analyze_count,
    autoanalyze_count
FROM pg_stat_user_tables
ORDER BY n_mod_since_analyze DESC, schemaname, relname
LIMIT :limit
