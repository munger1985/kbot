SELECT *
  FROM (
        SELECT inst_id,
               sql_id,
               child_number,
               plan_hash_value,
               id AS plan_line_id,
               parent_id,
               depth,
               position,
               operation,
               options,
               object_owner,
               object_name,
               object_type,
               cardinality,
               bytes,
               cost,
               cpu_cost,
               io_cost,
               temp_space,
               access_predicates,
               filter_predicates,
               projection,
               timestamp
          FROM gv$sql_plan
         WHERE sql_id = :sql_id
         ORDER BY inst_id,
                  child_number,
                  plan_hash_value,
                  id
       )
 WHERE ROWNUM <= :limit
