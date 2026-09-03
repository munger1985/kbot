SELECT plan_table_output
  FROM TABLE(
        dbms_xplan.display_cursor(
            :sql_id,
            NULL,
            'ALLSTATS LAST'
        )
       )
