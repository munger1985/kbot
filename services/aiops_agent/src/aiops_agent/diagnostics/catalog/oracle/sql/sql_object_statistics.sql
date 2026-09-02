WITH plan_tables AS (
    SELECT DISTINCT object_owner AS owner,
                    object_name AS table_name
      FROM gv$sql_plan
     WHERE sql_id = :sql_id
       AND object_owner IS NOT NULL
       AND object_name IS NOT NULL
       AND object_type LIKE 'TABLE%'
)
SELECT p.owner,
       p.table_name,
       t.partitioned,
       t.temporary,
       s.num_rows,
       s.blocks,
       s.avg_row_len,
       s.sample_size,
       s.last_analyzed,
       s.stale_stats,
       s.stattype_locked,
       s.global_stats,
       s.user_stats
  FROM plan_tables p
  JOIN dba_tables t
    ON t.owner = p.owner
   AND t.table_name = p.table_name
  LEFT JOIN dba_tab_statistics s
    ON s.owner = p.owner
   AND s.table_name = p.table_name
   AND s.partition_name IS NULL
   AND s.subpartition_name IS NULL
   AND s.object_type = 'TABLE'
 ORDER BY p.owner,
          p.table_name
