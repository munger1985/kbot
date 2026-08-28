WITH undo_files AS (
    SELECT
        tablespace_name,
        SUM(bytes) AS allocated_bytes
    FROM dba_data_files
    WHERE tablespace_name IN (
        SELECT tablespace_name
        FROM dba_tablespaces
        WHERE contents = 'UNDO'
    )
    GROUP BY tablespace_name
), undo_extents AS (
    SELECT
        tablespace_name,
        SUM(CASE WHEN status = 'ACTIVE' THEN bytes ELSE 0 END) AS active_bytes,
        SUM(CASE WHEN status = 'UNEXPIRED' THEN bytes ELSE 0 END) AS unexpired_bytes,
        SUM(CASE WHEN status = 'EXPIRED' THEN bytes ELSE 0 END) AS expired_bytes
    FROM dba_undo_extents
    GROUP BY tablespace_name
)
SELECT
    undo_files.tablespace_name,
    ROUND(undo_files.allocated_bytes / 1048576, 2) AS allocated_mb,
    ROUND(NVL(undo_extents.active_bytes, 0) / 1048576, 2) AS active_mb,
    ROUND(NVL(undo_extents.unexpired_bytes, 0) / 1048576, 2) AS unexpired_mb,
    ROUND(NVL(undo_extents.expired_bytes, 0) / 1048576, 2) AS expired_mb,
    ROUND(
        (
            NVL(undo_extents.active_bytes, 0)
            + NVL(undo_extents.unexpired_bytes, 0)
        ) / NULLIF(undo_files.allocated_bytes, 0) * 100,
        2
    ) AS retained_percent
FROM undo_files
LEFT JOIN undo_extents
  ON undo_extents.tablespace_name = undo_files.tablespace_name
ORDER BY retained_percent DESC, undo_files.tablespace_name
