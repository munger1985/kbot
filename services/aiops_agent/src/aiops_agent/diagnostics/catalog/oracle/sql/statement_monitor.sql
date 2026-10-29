SELECT *
  FROM (
        SELECT d.dbid, m.inst_id, m.con_id AS container_id, m.sql_id,
               s.child_number, m.sql_plan_hash_value AS plan_hash_value,
               m.sql_exec_id,
               TO_CHAR(m.sql_exec_start, 'YYYY-MM-DD"T"HH24:MI:SS') AS sql_exec_start_native,
               m.status, m.elapsed_time AS elapsed_time_us, m.cpu_time AS cpu_time_us,
               m.user_io_wait_time AS user_io_wait_time_us,
               m.application_wait_time AS application_wait_time_us,
               m.concurrency_wait_time AS concurrency_wait_time_us,
               m.cluster_wait_time AS cluster_wait_time_us,
               m.buffer_gets, m.disk_reads, m.physical_read_bytes, m.physical_write_bytes,
               m.px_servers_allocated
          FROM gv$sql_monitor m
          JOIN gv$sql s ON s.inst_id = m.inst_id AND s.con_id = m.con_id
           AND s.sql_id = m.sql_id AND s.child_address = m.sql_child_address
           AND s.plan_hash_value = m.sql_plan_hash_value
         CROSS JOIN v$database d
         WHERE d.dbid = :dbid AND m.inst_id = :inst_id AND m.con_id = :container_id
           AND m.sql_id = :sql_id AND s.child_number = :child_number
           AND m.sql_plan_hash_value = :plan_hash_value AND m.px_qcsid IS NULL
         ORDER BY m.sql_exec_start DESC, m.sql_exec_id DESC
       )
 WHERE ROWNUM <= :limit
