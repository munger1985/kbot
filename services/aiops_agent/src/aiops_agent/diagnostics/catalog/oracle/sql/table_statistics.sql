SELECT t.owner,
       t.table_name,
       t.partitioned,
       t.temporary,
       s.last_analyzed,
       s.stale_stats,
       s.stattype_locked
  FROM dba_tables t
  LEFT JOIN dba_tab_statistics s
    ON s.owner = t.owner
   AND s.table_name = t.table_name
   AND s.partition_name IS NULL
   AND s.subpartition_name IS NULL
   AND s.object_type = 'TABLE'
 WHERE t.owner = UPPER(:schema_name)
   AND t.table_name = UPPER(:table_name)
   AND t.nested = 'NO'
   AND t.secondary = 'N'
