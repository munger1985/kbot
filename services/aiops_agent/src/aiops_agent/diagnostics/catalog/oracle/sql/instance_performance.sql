SELECT
    metric_name,
    ROUND(value, 3) AS metric_value,
    metric_unit,
    ROUND(intsize_csec / 100, 0) AS window_seconds
FROM v$sysmetric
WHERE group_id = 2
  AND metric_name IN (
      'Host CPU Utilization (%)',
      'Database CPU Time Ratio',
      'Database Wait Time Ratio',
      'Physical Read Total Bytes Per Sec',
      'Physical Write Total Bytes Per Sec',
      'Physical Read Total IO Requests Per Sec',
      'Physical Write Total IO Requests Per Sec',
      'Average Active Sessions'
  )
ORDER BY metric_name
