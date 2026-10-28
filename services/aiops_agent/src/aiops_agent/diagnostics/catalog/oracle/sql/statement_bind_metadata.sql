SELECT *
  FROM (
        SELECT database_identity.dbid,
               binds.inst_id,
               binds.con_id AS container_id,
               binds.sql_id,
               binds.child_number,
               cursor_stats.plan_hash_value,
               binds.position AS bind_position,
               binds.datatype_string AS datatype_name,
               binds.was_captured,
               binds.last_captured
          FROM gv$sql_bind_capture binds
          JOIN gv$sql cursor_stats
            ON cursor_stats.inst_id = binds.inst_id
           AND cursor_stats.con_id = binds.con_id
           AND cursor_stats.sql_id = binds.sql_id
           AND cursor_stats.child_number = binds.child_number
           AND cursor_stats.child_address = binds.child_address
         CROSS JOIN v$database database_identity
         WHERE database_identity.dbid = :dbid
           AND binds.inst_id = :inst_id
           AND binds.con_id = :container_id
           AND binds.sql_id = :sql_id
           AND binds.child_number = :child_number
           AND cursor_stats.plan_hash_value = :plan_hash_value
         ORDER BY binds.position
       )
 WHERE ROWNUM <= :limit
