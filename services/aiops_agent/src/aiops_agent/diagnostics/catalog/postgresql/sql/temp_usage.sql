SELECT
    datid AS database_oid,
    datname AS database_name,
    temp_files,
    temp_bytes,
    stats_reset
FROM pg_stat_database
WHERE datname IS NOT NULL
ORDER BY temp_bytes DESC, datname
