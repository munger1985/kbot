WITH target_index AS (
    SELECT owner,
           index_name,
           table_owner,
           table_name,
           tablespace_name,
           status,
           index_type,
           partitioned,
           visibility,
           degree,
           leaf_blocks,
           num_rows,
           last_analyzed
      FROM dba_indexes
     WHERE owner = UPPER(:schema_name)
       AND index_name = UPPER(:index_name)
),
segment_space AS (
    SELECT owner,
           segment_name AS index_name,
           CEIL(SUM(bytes) / 1048576) AS index_size_mb
      FROM dba_segments
     WHERE owner = UPPER(:schema_name)
       AND segment_name = UPPER(:index_name)
       AND segment_type = 'INDEX'
     GROUP BY owner, segment_name
),
free_space AS (
    SELECT tablespace_name,
           FLOOR(SUM(bytes) / 1048576) AS tablespace_free_mb
      FROM dba_free_space
     GROUP BY tablespace_name
),
table_locks AS (
    SELECT o.owner AS table_owner,
           o.object_name AS table_name,
           COUNT(*) AS active_table_locks
      FROM gv$locked_object l
      JOIN dba_objects o
        ON o.object_id = l.object_id
     GROUP BY o.owner, o.object_name
)
SELECT i.owner,
       i.index_name,
       i.table_owner,
       i.table_name,
       i.tablespace_name,
       i.status,
       i.index_type,
       i.partitioned,
       i.visibility,
       i.degree,
       i.leaf_blocks,
       i.num_rows,
       i.last_analyzed,
       NVL(s.index_size_mb, 0) AS index_size_mb,
       NVL(f.tablespace_free_mb, 0) AS tablespace_free_mb,
       NVL(l.active_table_locks, 0) AS active_table_locks,
       CASE
           WHEN i.index_type IN ('NORMAL', 'NORMAL/REV') THEN 'YES'
           ELSE 'NO'
       END AS online_supported,
       CASE
           WHEN NVL(s.index_size_mb, 0) > 0
            AND NVL(f.tablespace_free_mb, 0) >= CEIL(s.index_size_mb * 1.2)
           THEN 'YES'
           ELSE 'NO'
       END AS space_sufficient
  FROM target_index i
  LEFT JOIN segment_space s
    ON s.owner = i.owner
   AND s.index_name = i.index_name
  LEFT JOIN free_space f
    ON f.tablespace_name = i.tablespace_name
  LEFT JOIN table_locks l
    ON l.table_owner = i.table_owner
   AND l.table_name = i.table_name
