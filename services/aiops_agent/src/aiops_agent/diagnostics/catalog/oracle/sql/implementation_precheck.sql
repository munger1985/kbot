WITH parameter_values AS (
    SELECT
        MAX(CASE WHEN name = 'db_unique_name' THEN value END) AS db_unique_name,
        MAX(CASE WHEN name = 'cluster_database' THEN value END) AS cluster_database,
        MAX(CASE WHEN name = 'db_recovery_file_dest' THEN value END) AS db_recovery_file_dest,
        MAX(CASE WHEN name = 'db_recovery_file_dest_size' THEN value END) AS db_recovery_file_dest_size,
        MAX(CASE WHEN name = 'compatible' THEN value END) AS compatible,
        MAX(CASE WHEN name = 'service_names' THEN value END) AS service_names
    FROM v$parameter
    WHERE name IN (
        'db_unique_name', 'cluster_database', 'db_recovery_file_dest',
        'db_recovery_file_dest_size', 'compatible', 'service_names'
    )
),
database_size AS (
    SELECT NVL(SUM(bytes), 0) AS datafile_bytes FROM v$datafile
),
redo_summary AS (
    SELECT COUNT(DISTINCT thread#) AS redo_threads, COUNT(*) AS redo_groups
    FROM v$log
),
backup_summary AS (
    SELECT
        COUNT(*) AS backup_set_count,
        MAX(completion_time) AS last_backup_time,
        CASE WHEN COUNT(*) > 0 THEN 'AVAILABLE' ELSE 'NOT_FOUND' END AS backup_chain_status
    FROM v$backup_set
    WHERE completion_time >= SYSDATE - 30
),
registry_summary AS (
    SELECT
        LISTAGG(comp_id || ':' || version || ':' || status, ',')
            WITHIN GROUP (ORDER BY comp_id) AS component_summary,
        SUM(CASE WHEN status <> 'VALID' THEN 1 ELSE 0 END) AS invalid_component_count
    FROM dba_registry
),
charset_summary AS (
    SELECT
        MAX(CASE WHEN parameter = 'NLS_CHARACTERSET' THEN value END) AS character_set,
        MAX(CASE WHEN parameter = 'NLS_NCHAR_CHARACTERSET' THEN value END) AS national_character_set
    FROM nls_database_parameters
    WHERE parameter IN ('NLS_CHARACTERSET', 'NLS_NCHAR_CHARACTERSET')
),
rman_summary AS (
    SELECT LISTAGG(name || '=' || value, '; ')
        WITHIN GROUP (ORDER BY name) AS rman_configuration
    FROM v$rman_configuration
)
SELECT
    d.name AS database_name,
    p.db_unique_name,
    i.instance_name,
    i.host_name,
    i.version,
    d.platform_name,
    d.database_role,
    d.open_mode,
    d.log_mode,
    d.force_logging,
    d.flashback_on,
    d.protection_mode,
    d.switchover_status,
    d.cdb,
    p.cluster_database,
    p.compatible,
    p.service_names,
    p.db_recovery_file_dest,
    p.db_recovery_file_dest_size,
    s.datafile_bytes,
    r.redo_threads,
    r.redo_groups,
    b.backup_set_count,
    TO_CHAR(b.last_backup_time, 'YYYY-MM-DD HH24:MI:SS') AS last_backup_time,
    b.backup_chain_status,
    g.component_summary,
    g.invalid_component_count,
    c.character_set,
    c.national_character_set,
    m.rman_configuration
FROM v$database d
CROSS JOIN v$instance i
CROSS JOIN parameter_values p
CROSS JOIN database_size s
CROSS JOIN redo_summary r
CROSS JOIN backup_summary b
CROSS JOIN registry_summary g
CROSS JOIN charset_summary c
CROSS JOIN rman_summary m
