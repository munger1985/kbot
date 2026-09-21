SELECT dbms_sqltune.report_sql_monitor(
            sql_id => :sql_id,
            sql_exec_start => CASE
                WHEN :sql_exec_start = '' THEN CAST(NULL AS DATE)
                ELSE CAST(
                    to_timestamp_tz(
                        :sql_exec_start,
                        'YYYY-MM-DD"T"HH24:MI:SS TZH:TZM'
                    ) AS DATE
                )
            END,
            sql_exec_id => CASE
                WHEN :sql_exec_id = 0 THEN NULL
                ELSE :sql_exec_id
            END,
            report_level => 'TYPICAL',
            type => 'HTML'
       ) AS output
  FROM dual
