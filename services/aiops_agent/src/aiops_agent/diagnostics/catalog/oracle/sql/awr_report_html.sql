SELECT output
  FROM TABLE(
        dbms_workload_repository.awr_report_html(
            (SELECT dbid FROM v$database),
            (SELECT instance_number FROM v$instance),
            :begin_snapshot_id,
            :end_snapshot_id,
            0
        )
       )
