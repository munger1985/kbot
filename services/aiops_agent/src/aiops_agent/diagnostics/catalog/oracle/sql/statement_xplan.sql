SELECT s.dbid, s.inst_id, s.container_id, s.sql_id, s.child_number,
       s.plan_hash_value, x.plan_table_output
  FROM (
        SELECT d.dbid, c.inst_id, c.con_id AS container_id, c.sql_id,
               c.child_number, c.plan_hash_value
          FROM gv$sql c CROSS JOIN v$database d CROSS JOIN v$instance i
         WHERE d.dbid = :dbid AND c.inst_id = :inst_id
           AND c.inst_id = i.instance_number
           AND c.con_id = :container_id
           AND c.con_id = TO_NUMBER(SYS_CONTEXT('USERENV', 'CON_ID'))
           AND c.sql_id = :sql_id AND c.child_number = :child_number
           AND c.plan_hash_value = :plan_hash_value
       ) s,
       TABLE(dbms_xplan.display_cursor(
           s.sql_id, s.child_number, 'ALLSTATS LAST +PREDICATE +ALIAS +OUTLINE'
       )) x
