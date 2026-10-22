SELECT
    name,
    ROUND(space_limit / 1048576, 2) AS space_limit_mb,
    ROUND(space_used / 1048576, 2) AS space_used_mb,
    ROUND(space_reclaimable / 1048576, 2) AS space_reclaimable_mb,
    number_of_files,
    CASE
        WHEN space_limit > 0
        THEN ROUND(space_used * 100 / space_limit, 2)
        ELSE NULL
    END AS utilization_percent
FROM v$recovery_file_dest
