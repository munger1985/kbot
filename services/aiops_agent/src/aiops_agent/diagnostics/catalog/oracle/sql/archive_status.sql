SELECT
    d.database_role,
    d.open_mode,
    d.log_mode,
    d.force_logging,
    d.flashback_on,
    r.name AS recovery_file_dest,
    ROUND(r.space_limit / 1024 / 1024, 2) AS fra_limit_mb,
    ROUND(r.space_used / 1024 / 1024, 2) AS fra_used_mb,
    ROUND(r.space_reclaimable / 1024 / 1024, 2) AS fra_reclaimable_mb,
    r.number_of_files AS fra_file_count
FROM v$database d
LEFT JOIN v$recovery_file_dest r ON 1 = 1
