SELECT *
  FROM (
        SELECT d.dbid, a.inst_id, a.con_id AS container_id, a.sql_id,
               a.sql_child_number AS child_number, a.sql_plan_hash_value AS plan_hash_value,
               a.session_state, a.wait_class, a.event, a.sql_plan_line_id AS plan_line_id,
               a.blocking_inst_id, a.blocking_session, a.blocking_session_status,
               COUNT(*) AS sample_count, SUM(COUNT(*)) OVER () AS total_sample_count,
               MIN(a.sample_time_utc) AS first_sample_time_utc,
               MAX(a.sample_time_utc) AS last_sample_time_utc,
               :begin_time AS requested_begin_time, :end_time AS requested_end_time
          FROM gv$active_session_history a CROSS JOIN v$database d
         WHERE d.dbid = :dbid AND a.inst_id = :inst_id AND a.con_id = :container_id
           AND a.sql_id = :sql_id AND a.sql_child_number = :child_number
           AND a.sql_plan_hash_value = :plan_hash_value
           AND a.sample_time_utc >= SYS_EXTRACT_UTC(
               TO_TIMESTAMP_TZ(:begin_time, 'YYYY-MM-DD"T"HH24:MI:SSTZH:TZM'))
           AND a.sample_time_utc < SYS_EXTRACT_UTC(
               TO_TIMESTAMP_TZ(:end_time, 'YYYY-MM-DD"T"HH24:MI:SSTZH:TZM'))
         GROUP BY d.dbid, a.inst_id, a.con_id, a.sql_id, a.sql_child_number,
                  a.sql_plan_hash_value, a.session_state, a.wait_class, a.event,
                  a.sql_plan_line_id, a.blocking_inst_id, a.blocking_session,
                  a.blocking_session_status
         ORDER BY sample_count DESC
       )
 WHERE ROWNUM <= :limit
