SELECT output
  FROM TABLE(
        dbms_workload_repository.awr_report_html(
            (SELECT dbid FROM v$database),
            :instance_number,
            :begin_snapshot_id,
            :end_snapshot_id,
            0
        )
       )
