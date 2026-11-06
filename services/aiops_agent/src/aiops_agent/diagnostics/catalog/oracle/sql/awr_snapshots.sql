SELECT
    snap_id AS snapshot_id,
    instance_number AS instance_number,
    sys_extract_utc(begin_interval_time_tz) AS begin_time,
    sys_extract_utc(end_interval_time_tz) AS end_time,
    startup_time AS startup_time
  FROM dba_hist_snapshot
 WHERE dbid = (SELECT dbid FROM v$database)
   AND instance_number = (SELECT instance_number FROM v$instance)
 ORDER BY snap_id DESC
