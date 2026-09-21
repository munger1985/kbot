SELECT
    schemaname AS schema_name,
    relname AS table_name,
    n_live_tup AS live_tuples,
    n_dead_tup AS dead_tuples,
    CASE
        WHEN n_live_tup + n_dead_tup = 0 THEN 0
        ELSE ROUND(
            100.0 * n_dead_tup / (n_live_tup + n_dead_tup),
            2
        )
    END AS dead_tuple_percent,
    last_autovacuum,
    last_vacuum
FROM pg_stat_user_tables
WHERE n_dead_tup >= 1000
ORDER BY n_dead_tup DESC
