SELECT
    owner,
    object_type,
    status,
    COUNT(*) AS object_count
FROM dba_objects
WHERE status <> 'VALID'
GROUP BY owner, object_type, status
ORDER BY object_count DESC, owner, object_type
