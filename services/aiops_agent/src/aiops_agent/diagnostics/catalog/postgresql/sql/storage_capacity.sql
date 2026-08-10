SELECT
    datname AS database_name,
    ROUND(CAST(pg_database_size(datname) AS numeric) / 1048576, 2) AS allocated_mb,
    CAST(0 AS bigint) AS table_count
FROM pg_database
WHERE datistemplate = false
ORDER BY datname
