WITH parameter_values AS (
    SELECT
        MAX(CASE WHEN name = 'db_unique_name' THEN value END) AS db_unique_name,
        MAX(CASE WHEN name = 'service_names' THEN value END) AS service_names,
        MAX(CASE WHEN name = 'db_domain' THEN value END) AS db_domain,
        MAX(CASE WHEN name = 'remote_login_passwordfile' THEN value END) AS remote_login_passwordfile,
        MAX(CASE WHEN name = 'log_archive_config' THEN value END) AS log_archive_config,
        MAX(CASE WHEN name = 'log_archive_dest_1' THEN value END) AS log_archive_dest_1,
        MAX(CASE WHEN name = 'log_archive_dest_2' THEN value END) AS log_archive_dest_2,
        MAX(CASE WHEN name = 'log_archive_dest_state_2' THEN value END) AS log_archive_dest_state_2,
        MAX(CASE WHEN name = 'fal_server' THEN value END) AS fal_server,
        MAX(CASE WHEN name = 'fal_client' THEN value END) AS fal_client,
        MAX(CASE WHEN name = 'standby_file_management' THEN value END) AS standby_file_management,
        MAX(CASE WHEN name = 'db_file_name_convert' THEN value END) AS db_file_name_convert,
        MAX(CASE WHEN name = 'log_file_name_convert' THEN value END) AS log_file_name_convert,
        MAX(CASE WHEN name = 'db_create_file_dest' THEN value END) AS db_create_file_dest,
        MAX(CASE WHEN name = 'db_recovery_file_dest' THEN value END) AS db_recovery_file_dest,
        MAX(CASE WHEN name = 'db_recovery_file_dest_size' THEN value END) AS db_recovery_file_dest_size,
        MAX(CASE WHEN name = 'dg_broker_start' THEN value END) AS dg_broker_start
    FROM v$parameter
    WHERE name IN (
        'db_unique_name',
        'service_names',
        'db_domain',
        'remote_login_passwordfile',
        'log_archive_config',
        'log_archive_dest_1',
        'log_archive_dest_2',
        'log_archive_dest_state_2',
        'fal_server',
        'fal_client',
        'standby_file_management',
        'db_file_name_convert',
        'log_file_name_convert',
        'db_create_file_dest',
        'db_recovery_file_dest',
        'db_recovery_file_dest_size',
        'dg_broker_start'
    )
),
online_redo AS (
    SELECT
        thread# AS thread_number,
        COUNT(*) AS online_group_count,
        ROUND(MIN(bytes) / 1048576) AS min_size_mb,
        ROUND(MAX(bytes) / 1048576) AS max_size_mb
    FROM v$log
    GROUP BY thread#
),
standby_redo AS (
    SELECT
        thread# AS thread_number,
        COUNT(*) AS standby_group_count,
        ROUND(MIN(bytes) / 1048576) AS min_size_mb,
        ROUND(MAX(bytes) / 1048576) AS max_size_mb
    FROM v$standby_log
    GROUP BY thread#
),
redo_summary AS (
    SELECT
        COUNT(*) AS online_redo_threads,
        SUM(o.online_group_count) AS online_redo_groups,
        MIN(o.min_size_mb) AS online_redo_min_size_mb,
        MAX(o.max_size_mb) AS online_redo_max_size_mb,
        SUM(NVL(s.standby_group_count, 0)) AS standby_redo_groups,
        MIN(s.min_size_mb) AS standby_redo_min_size_mb,
        MAX(s.max_size_mb) AS standby_redo_max_size_mb,
        SUM(o.online_group_count + 1) AS required_standby_redo_groups,
        SUM(GREATEST(o.online_group_count + 1 - NVL(s.standby_group_count, 0), 0)) AS standby_redo_shortage,
        LISTAGG(
            TO_CHAR(o.thread_number) || ':' ||
            TO_CHAR(o.online_group_count) || ':' ||
            TO_CHAR(o.online_group_count + 1) || ':' ||
            TO_CHAR(GREATEST(o.online_group_count + 1 - NVL(s.standby_group_count, 0), 0)) || ':' ||
            TO_CHAR(o.max_size_mb),
            ','
        ) WITHIN GROUP (ORDER BY o.thread_number) AS redo_thread_plan
    FROM online_redo o
    LEFT JOIN standby_redo s ON s.thread_number = o.thread_number
)
SELECT
    d.name AS database_name,
    p.db_unique_name,
    i.instance_name,
    i.host_name,
    p.service_names,
    p.db_domain,
    d.platform_name,
    d.cdb,
    SYS_CONTEXT('USERENV', 'CON_NAME') AS container_name,
    TO_NUMBER(SYS_CONTEXT('USERENV', 'CON_ID')) AS container_id,
    d.database_role,
    d.open_mode,
    d.log_mode,
    d.force_logging,
    d.flashback_on,
    d.protection_mode,
    d.switchover_status,
    p.remote_login_passwordfile,
    p.log_archive_config,
    p.log_archive_dest_1,
    p.log_archive_dest_2,
    p.log_archive_dest_state_2,
    p.fal_server,
    p.fal_client,
    p.standby_file_management,
    p.db_file_name_convert,
    p.log_file_name_convert,
    p.db_create_file_dest,
    p.db_recovery_file_dest,
    p.db_recovery_file_dest_size,
    p.dg_broker_start,
    r.space_limit AS fra_space_limit_bytes,
    r.space_used AS fra_space_used_bytes,
    rs.online_redo_threads,
    rs.online_redo_groups,
    rs.online_redo_min_size_mb,
    rs.online_redo_max_size_mb,
    rs.standby_redo_groups,
    rs.standby_redo_min_size_mb,
    rs.standby_redo_max_size_mb,
    rs.required_standby_redo_groups,
    rs.standby_redo_shortage,
    rs.redo_thread_plan
FROM v$database d
CROSS JOIN v$instance i
CROSS JOIN parameter_values p
CROSS JOIN redo_summary rs
LEFT JOIN v$recovery_file_dest r ON 1 = 1
