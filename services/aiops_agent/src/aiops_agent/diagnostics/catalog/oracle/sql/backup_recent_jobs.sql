SELECT
    session_key,
    input_type,
    status,
    start_time,
    end_time,
    ROUND(elapsed_seconds, 2) AS elapsed_seconds,
    ROUND(input_bytes / 1048576, 2) AS input_mb,
    ROUND(output_bytes / 1048576, 2) AS output_mb,
    output_device_type
FROM (
    SELECT
        session_key,
        input_type,
        status,
        start_time,
        end_time,
        elapsed_seconds,
        input_bytes,
        output_bytes,
        output_device_type
    FROM v$rman_backup_job_details
    WHERE start_time >= SYSDATE - :days
    ORDER BY start_time DESC
)
WHERE ROWNUM <= :limit
