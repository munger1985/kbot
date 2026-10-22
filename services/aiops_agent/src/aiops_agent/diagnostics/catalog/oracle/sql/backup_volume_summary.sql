SELECT
    COUNT(*) AS job_count,
    NVL(SUM(CASE WHEN status LIKE 'COMPLETED%' THEN 1 ELSE 0 END), 0) AS successful_job_count,
    NVL(SUM(CASE WHEN status NOT LIKE 'COMPLETED%' THEN 1 ELSE 0 END), 0) AS failed_job_count,
    ROUND(NVL(SUM(input_bytes), 0) / 1048576, 2) AS input_mb,
    ROUND(NVL(SUM(output_bytes), 0) / 1048576, 2) AS output_mb,
    MIN(start_time) AS first_start_time,
    MAX(end_time) AS last_end_time
FROM v$rman_backup_job_details
WHERE start_time >= SYSDATE - :days
