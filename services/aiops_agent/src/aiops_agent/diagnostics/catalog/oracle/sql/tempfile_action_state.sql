SELECT f.file_name,
       f.tablespace_name,
       ROUND(f.bytes / 1048576) AS current_size_mb,
       ROUND(f.maxbytes / 1048576) AS current_max_size_mb,
       f.autoextensible,
       ROUND(f.increment_by * t.block_size / 1048576) AS current_next_mb,
       TO_NUMBER(:new_size_mb) AS requested_size_mb,
       TO_NUMBER(:next_mb) AS requested_next_mb,
       TO_NUMBER(:max_size_mb) AS requested_max_size_mb,
       f.status,
       'ONLINE' AS online_status
  FROM dba_temp_files f
  JOIN dba_tablespaces t
    ON t.tablespace_name = f.tablespace_name
 WHERE f.file_name = :file_name
   AND f.status = 'AVAILABLE'
