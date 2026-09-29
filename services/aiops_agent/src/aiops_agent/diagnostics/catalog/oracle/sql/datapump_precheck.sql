WITH parameter_values AS (
    SELECT
        MAX(CASE WHEN name = 'db_unique_name' THEN value END) AS db_unique_name,
        MAX(CASE WHEN name = 'compatible' THEN value END) AS compatible,
        MAX(CASE WHEN name = 'service_names' THEN value END) AS service_names
    FROM v$parameter
    WHERE name IN ('db_unique_name', 'compatible', 'service_names')
),
database_size AS (
    SELECT NVL(SUM(bytes), 0) AS datafile_bytes
    FROM v$datafile
),
charset_summary AS (
    SELECT
        MAX(CASE WHEN parameter = 'NLS_CHARACTERSET' THEN value END)
            AS character_set,
        MAX(CASE WHEN parameter = 'NLS_NCHAR_CHARACTERSET' THEN value END)
            AS national_character_set
    FROM nls_database_parameters
    WHERE parameter IN ('NLS_CHARACTERSET', 'NLS_NCHAR_CHARACTERSET')
),
directory_summary AS (
    SELECT
        MAX(directory_name) AS datapump_directory_name,
        MAX(directory_path) AS datapump_directory_path
    FROM dba_directories
    WHERE directory_name = 'DATA_PUMP_DIR'
      AND directory_path IS NOT NULL
),
schema_rows AS (
    SELECT
        username,
        COUNT(*) OVER () AS source_schema_count,
        ROW_NUMBER() OVER (ORDER BY username) AS row_number_value
    FROM dba_users
    WHERE oracle_maintained = 'N'
      AND common = 'NO'
),
schema_summary AS (
    SELECT
        CASE
            WHEN NVL(MAX(source_schema_count), 0) <= 30
            THEN LISTAGG(username, ',') WITHIN GROUP (ORDER BY username)
        END AS source_schemas,
        NVL(MAX(source_schema_count), 0) AS source_schema_count
    FROM schema_rows
    WHERE row_number_value <= 30
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
    d.cdb,
    p.compatible,
    p.service_names,
    s.datafile_bytes,
    c.character_set,
    c.national_character_set,
    x.datapump_directory_name,
    x.datapump_directory_path,
    u.source_schemas,
    u.source_schema_count
FROM v$database d
CROSS JOIN v$instance i
CROSS JOIN parameter_values p
CROSS JOIN database_size s
CROSS JOIN charset_summary c
CROSS JOIN directory_summary x
CROSS JOIN schema_summary u
