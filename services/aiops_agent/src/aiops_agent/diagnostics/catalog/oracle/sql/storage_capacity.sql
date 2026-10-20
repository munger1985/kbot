WITH data_file_detail AS (
    SELECT
        tablespace_name,
        bytes,
        maxbytes,
        CASE
            WHEN file_name LIKE '+%' THEN 'ASM'
            ELSE 'FILESYSTEM'
        END AS storage_type,
        CASE
            WHEN file_name LIKE '+%'
            THEN REGEXP_SUBSTR(file_name, '[^/+]+', 2, 1)
        END AS asm_diskgroup,
        CASE
            WHEN file_name NOT LIKE '+%'
            THEN REGEXP_REPLACE(file_name, '[/\\][^/\\]+$', '')
        END AS datafile_directory
    FROM dba_data_files
),
data_files AS (
    SELECT
        tablespace_name,
        SUM(bytes) AS allocated_bytes,
        SUM(GREATEST(bytes, maxbytes)) AS maximum_bytes,
        COUNT(*) AS file_count,
        CASE
            WHEN MIN(storage_type) = MAX(storage_type) THEN MIN(storage_type)
            ELSE 'MIXED'
        END AS storage_type,
        LISTAGG(DISTINCT asm_diskgroup, ', ')
            WITHIN GROUP (ORDER BY asm_diskgroup) AS asm_diskgroups,
        LISTAGG(DISTINCT datafile_directory, ', ')
            WITHIN GROUP (ORDER BY datafile_directory) AS datafile_directories
    FROM data_file_detail
    GROUP BY tablespace_name
),
free_space AS (
    SELECT
        tablespace_name,
        SUM(bytes) AS free_bytes
    FROM dba_free_space
    GROUP BY tablespace_name
)
SELECT
    data_files.tablespace_name,
    ROUND(data_files.allocated_bytes / 1048576, 2) AS allocated_mb,
    ROUND(
        (data_files.allocated_bytes - NVL(free_space.free_bytes, 0))
        / 1048576,
        2
    ) AS used_mb,
    ROUND(NVL(free_space.free_bytes, 0) / 1048576, 2) AS free_mb,
    ROUND(
        (data_files.allocated_bytes - NVL(free_space.free_bytes, 0))
        / NULLIF(data_files.allocated_bytes, 0)
        * 100,
        2
    ) AS used_percent,
    ROUND(data_files.maximum_bytes / 1048576, 2) AS maximum_mb,
    ROUND(
        (
            data_files.maximum_bytes
            - data_files.allocated_bytes
            + NVL(free_space.free_bytes, 0)
        ) / 1048576,
        2
    ) AS maximum_headroom_mb,
    data_files.file_count,
    data_files.storage_type,
    data_files.asm_diskgroups,
    data_files.datafile_directories
FROM data_files
LEFT JOIN free_space
    ON free_space.tablespace_name = data_files.tablespace_name
ORDER BY used_percent DESC, data_files.tablespace_name
