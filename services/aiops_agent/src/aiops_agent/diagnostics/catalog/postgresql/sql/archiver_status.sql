SELECT
    current_setting('archive_mode') AS archive_mode,
    CASE
        WHEN current_setting('archive_mode') IN ('on', 'always')
         AND (
             COALESCE(NULLIF(TRIM(current_setting('archive_command', true)), ''), '(disabled)') <> '(disabled)'
             OR COALESCE(NULLIF(TRIM(current_setting('archive_library', true)), ''), '(disabled)') <> '(disabled)'
         )
        THEN 'YES'
        ELSE 'NO'
    END AS archive_command_configured,
    archived_count,
    last_archived_wal,
    last_archived_time,
    failed_count,
    last_failed_wal,
    last_failed_time,
    stats_reset
FROM pg_stat_archiver
