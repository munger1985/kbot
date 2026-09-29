SELECT
    relid AS relation_oid,
    schemaname AS schema_name,
    relname AS table_name,
    n_live_tup,
    n_dead_tup,
    CASE
        WHEN n_live_tup + n_dead_tup = 0 THEN NULL
        ELSE ROUND(
            CAST(n_dead_tup AS numeric) * 100 / (n_live_tup + n_dead_tup),
            2
        )
    END AS estimated_dead_tuple_percent,
    pg_total_relation_size(relid) AS total_bytes,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
ORDER BY estimated_dead_tuple_percent DESC NULLS LAST,
         total_bytes DESC,
         schemaname,
         relname
LIMIT :limit
