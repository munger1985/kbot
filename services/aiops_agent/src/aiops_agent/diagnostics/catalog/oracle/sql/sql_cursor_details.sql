SELECT *
  FROM (
        SELECT database_identity.dbid,
               inst_id,
               con_id AS container_id,
               sql_id,
               child_number,
               plan_hash_value,
               parsing_schema_name,
               module,
               action,
               executions,
               elapsed_time / 1000000 AS elapsed_seconds,
               cpu_time / 1000000 AS cpu_seconds,
               buffer_gets,
               disk_reads,
               direct_writes,
               rows_processed,
               fetches,
               sorts,
               loads,
               invalidations,
               is_bind_sensitive,
               is_bind_aware,
               sql_profile,
               sql_plan_baseline,
               first_load_time,
               last_load_time,
               last_active_time,
               sql_text
          FROM gv$sql
         CROSS JOIN v$database database_identity
         WHERE sql_id = :sql_id
         ORDER BY last_active_time DESC NULLS LAST,
                  inst_id,
                  child_number
       )
 WHERE ROWNUM <= :limit
