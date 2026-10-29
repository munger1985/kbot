SELECT *
  FROM (
        SELECT database_identity.dbid,
               cursor_stats.inst_id,
               cursor_stats.con_id AS container_id,
               cursor_stats.sql_id,
               cursor_stats.child_number,
               cursor_stats.plan_hash_value,
               cursor_stats.parsing_schema_name,
               cursor_stats.module,
               cursor_stats.action,
               cursor_stats.executions,
               cursor_stats.elapsed_time / 1000000 AS elapsed_seconds,
               cursor_stats.cpu_time / 1000000 AS cpu_seconds,
               cursor_stats.buffer_gets,
               cursor_stats.disk_reads,
               cursor_stats.direct_writes,
               cursor_stats.rows_processed,
               cursor_stats.fetches,
               cursor_stats.sorts,
               cursor_stats.loads,
               cursor_stats.invalidations,
               cursor_stats.is_bind_sensitive,
               cursor_stats.is_bind_aware,
               cursor_stats.sql_profile,
               cursor_stats.sql_plan_baseline,
               cursor_stats.first_load_time,
               cursor_stats.last_load_time,
               cursor_stats.last_active_time,
               cursor_stats.sql_text
          FROM gv$sql cursor_stats
         CROSS JOIN v$database database_identity
         WHERE cursor_stats.sql_id = :sql_id
         ORDER BY cursor_stats.last_active_time DESC NULLS LAST,
                  cursor_stats.inst_id,
                  cursor_stats.child_number
       )
 WHERE ROWNUM <= :limit
