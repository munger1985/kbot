SELECT location_kind, storage_name, file_path, server_uuid
FROM (
    SELECT 'DATA_DIRECTORY' AS location_kind, 'datadir' AS storage_name,
           @@datadir AS file_path, @@server_uuid AS server_uuid
    UNION ALL
    SELECT 'TABLESPACE', TABLESPACE_NAME, FILE_NAME, @@server_uuid
    FROM information_schema.FILES
    WHERE ENGINE = 'InnoDB' AND FILE_TYPE = 'TABLESPACE'
      AND FILE_NAME IS NOT NULL
) AS locations
ORDER BY location_kind, storage_name, file_path
LIMIT :limit
