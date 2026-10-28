SELECT *
  FROM (
        SELECT database_identity.dbid,
               plan.inst_id,
               plan.con_id AS container_id,
               plan.sql_id,
               plan.child_number,
               plan.plan_hash_value,
               plan.id AS plan_line_id,
               plan.parent_id,
               plan.operation,
               plan.options,
               plan.object_owner,
               plan.object_name,
               plan.object_type,
               plan.cardinality,
               plan.cost,
               plan.last_starts,
               plan.last_output_rows,
               plan.last_elapsed_time,
               plan.last_cr_buffer_gets,
               plan.last_disk_reads,
               plan.last_disk_writes
          FROM gv$sql_plan_statistics_all plan
         CROSS JOIN v$database database_identity
         WHERE database_identity.dbid = :dbid
           AND plan.inst_id = :inst_id
           AND plan.con_id = :container_id
           AND plan.sql_id = :sql_id
           AND plan.child_number = :child_number
           AND plan.plan_hash_value = :plan_hash_value
         ORDER BY plan.id
       )
 WHERE ROWNUM <= :limit
