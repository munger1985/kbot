SELECT output
  FROM TABLE(
        dbms_workload_repository.awr_diff_report_html(
            (SELECT dbid FROM v$database),
            (SELECT instance_number FROM v$instance),
            :baseline_begin_snapshot_id,
            :baseline_end_snapshot_id,
            :after_begin_snapshot_id,
            :after_end_snapshot_id
        )
       )
