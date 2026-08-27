WITH parameter_values AS (
    SELECT
        MAX(CASE WHEN name = 'max_pdb_sessions' THEN TO_NUMBER(value) END) AS max_pdb_sessions,
        MAX(CASE WHEN name = 'sessions' THEN TO_NUMBER(value) END) AS instance_sessions
    FROM v$parameter
    WHERE name IN ('max_pdb_sessions', 'sessions')
), session_count AS (
    SELECT COUNT(*) AS current_sessions
    FROM v$session
), effective_limit AS (
    SELECT
        session_count.current_sessions,
        CASE
            WHEN NVL(parameter_values.max_pdb_sessions, 0) > 0
            THEN parameter_values.max_pdb_sessions
            ELSE parameter_values.instance_sessions
        END AS limit_sessions
    FROM parameter_values
    CROSS JOIN session_count
)
SELECT
    'sessions' AS resource_name,
    current_sessions AS current_utilization,
    CAST(NULL AS NUMBER) AS max_utilization,
    TO_CHAR(limit_sessions) AS limit_value,
    ROUND(100 * current_sessions / limit_sessions, 3) AS utilization_percent
FROM effective_limit
WHERE limit_sessions > 0
