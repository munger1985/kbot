WITH temp_files AS (
    SELECT
        tablespace_name,
        SUM(bytes) AS allocated_bytes,
        SUM(GREATEST(bytes, maxbytes)) AS maximum_bytes,
        COUNT(*) AS file_count
    FROM dba_temp_files
    GROUP BY tablespace_name
), temp_usage AS (
    SELECT
        tablespace_name,
        SUM(bytes_used) AS used_bytes,
        SUM(bytes_free) AS free_bytes
    FROM v$temp_space_header
    GROUP BY tablespace_name
)
SELECT
    temp_files.tablespace_name,
    ROUND(temp_files.allocated_bytes / 1048576, 2) AS allocated_mb,
    ROUND(NVL(temp_usage.used_bytes, 0) / 1048576, 2) AS used_mb,
    ROUND(NVL(temp_usage.free_bytes, 0) / 1048576, 2) AS free_mb,
    ROUND(
        NVL(temp_usage.used_bytes, 0)
        / NULLIF(temp_files.allocated_bytes, 0)
        * 100,
        2
    ) AS used_percent,
    ROUND(temp_files.maximum_bytes / 1048576, 2) AS maximum_mb,
    temp_files.file_count
FROM temp_files
LEFT JOIN temp_usage
  ON temp_usage.tablespace_name = temp_files.tablespace_name
ORDER BY used_percent DESC, temp_files.tablespace_name
