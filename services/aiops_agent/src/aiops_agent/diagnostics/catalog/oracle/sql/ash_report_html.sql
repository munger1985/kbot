SELECT output
  FROM TABLE(
        dbms_workload_repository.ash_report_html(
            (SELECT dbid FROM v$database),
            (SELECT instance_number FROM v$instance),
            to_timestamp_tz(
                :begin_time,
                'YYYY-MM-DD"T"HH24:MI:SS TZH:TZM'
            ),
            to_timestamp_tz(
                :end_time,
                'YYYY-MM-DD"T"HH24:MI:SS TZH:TZM'
            ),
            0,
            0
        )
       )
