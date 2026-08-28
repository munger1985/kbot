SELECT
    thread# AS thread_number,
    group# AS group_number,
    sequence# AS sequence_number,
    ROUND(bytes / 1048576, 2) AS size_mb,
    members AS member_count,
    archived AS archived,
    status AS status,
    first_time AS first_change_at,
    next_time AS next_change_at
FROM v$log
ORDER BY thread#, group#
