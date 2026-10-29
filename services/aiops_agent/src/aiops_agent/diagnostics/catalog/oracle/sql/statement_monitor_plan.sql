SELECT *
  FROM (
        SELECT d.dbid, p.inst_id, p.con_id AS container_id, p.sql_id,
               s.child_number, p.sql_plan_hash_value AS plan_hash_value,
               p.sql_exec_id,
               TO_CHAR(p.sql_exec_start, 'YYYY-MM-DD"T"HH24:MI:SS') AS sql_exec_start_native,
               p.process_name, p.sid AS session_id, m.session_serial# AS session_serial,
               p.plan_line_id, p.plan_operation, p.plan_options,
               p.plan_object_owner, p.plan_object_name, p.plan_cardinality,
               p.starts, p.output_rows, p.physical_read_bytes, p.physical_write_bytes,
               p.workarea_mem, p.workarea_max_mem, p.workarea_tempseg, p.workarea_max_tempseg,
               p.status
          FROM gv$sql_plan_monitor p
          JOIN gv$sql_monitor m ON m.inst_id = p.inst_id AND m.con_id = p.con_id
           AND m.key = p.key AND m.sql_id = p.sql_id
           AND m.sql_exec_id = p.sql_exec_id AND m.sql_exec_start = p.sql_exec_start
           AND m.sql_child_address = p.sql_child_address
          JOIN gv$sql s ON s.inst_id = p.inst_id AND s.con_id = p.con_id
           AND s.sql_id = p.sql_id AND s.child_address = p.sql_child_address
           AND s.child_number = :child_number
           AND s.plan_hash_value = p.sql_plan_hash_value
         CROSS JOIN v$database d
         WHERE d.dbid = :dbid AND p.inst_id = :inst_id AND p.con_id = :container_id
           AND p.sql_id = :sql_id AND p.sql_plan_hash_value = :plan_hash_value
           AND p.sql_exec_id = :sql_exec_id
           AND p.sql_exec_start = TO_DATE(:sql_exec_start_native, 'YYYY-MM-DD"T"HH24:MI:SS')
         ORDER BY p.plan_line_id, p.process_name, p.sid
       )
 WHERE ROWNUM <= :limit
