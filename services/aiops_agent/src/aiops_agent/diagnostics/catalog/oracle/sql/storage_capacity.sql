SELECT
    tablespace_name,
    ROUND(SUM(bytes) / 1048576, 2) AS allocated_mb,
    ROUND(SUM(maxbytes) / 1048576, 2) AS maximum_mb,
    COUNT(*) AS file_count
FROM dba_data_files
GROUP BY tablespace_name
ORDER BY tablespace_name
