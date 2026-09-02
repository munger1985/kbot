WITH target_partition AS (
    SELECT p.index_owner AS owner,
           p.index_name,
           p.partition_name,
           p.tablespace_name,
           p.status,
           p.leaf_blocks,
           p.num_rows,
           p.last_analyzed,
           i.table_owner,
           i.table_name,
           i.index_type,
           i.partitioned
      FROM dba_ind_partitions p
      JOIN dba_indexes i
        ON i.owner = p.index_owner
       AND i.index_name = p.index_name
     WHERE p.index_owner = UPPER(:schema_name)
       AND p.index_name = UPPER(:index_name)
       AND p.partition_name = UPPER(:partition_name)
),
segment_space AS (
    SELECT owner,
           segment_name AS index_name,
           partition_name,
           CEIL(SUM(bytes) / 1048576) AS partition_size_mb
      FROM dba_segments
     WHERE owner = UPPER(:schema_name)
       AND segment_name = UPPER(:index_name)
       AND partition_name = UPPER(:partition_name)
       AND segment_type = 'INDEX PARTITION'
     GROUP BY owner, segment_name, partition_name
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
SELECT p.owner,
       p.index_name,
       p.partition_name,
       p.table_owner,
       p.table_name,
       p.tablespace_name,
       p.status,
       p.index_type,
       p.partitioned,
       p.leaf_blocks,
       p.num_rows,
       p.last_analyzed,
       NVL(s.partition_size_mb, 0) AS partition_size_mb,
       NVL(f.tablespace_free_mb, 0) AS tablespace_free_mb,
       NVL(l.active_table_locks, 0) AS active_table_locks,
       CASE
           WHEN p.index_type IN ('NORMAL', 'NORMAL/REV') THEN 'YES'
           ELSE 'NO'
       END AS online_supported,
       CASE
           WHEN NVL(s.partition_size_mb, 0) > 0
            AND NVL(f.tablespace_free_mb, 0) >= CEIL(s.partition_size_mb * 1.2)
           THEN 'YES'
           ELSE 'NO'
       END AS space_sufficient
  FROM target_partition p
  LEFT JOIN segment_space s
    ON s.owner = p.owner
   AND s.index_name = p.index_name
   AND s.partition_name = p.partition_name
  LEFT JOIN free_space f
    ON f.tablespace_name = p.tablespace_name
  LEFT JOIN table_locks l
    ON l.table_owner = p.table_owner
   AND l.table_name = p.table_name
