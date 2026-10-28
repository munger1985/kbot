SELECT schema_name, table_name, relation_kind, native_object_id, container_id
FROM (
    SELECT objects.owner AS schema_name,
           objects.object_name AS table_name,
           objects.object_type AS relation_kind,
           TO_CHAR(objects.object_id) AS native_object_id,
           TO_NUMBER(SYS_CONTEXT('USERENV', 'CON_ID')) AS container_id
    FROM all_objects objects
    WHERE objects.object_name = :table_name
      AND (:schema_name IS NULL OR objects.owner = :schema_name)
      AND objects.object_type IN ('TABLE', 'VIEW', 'MATERIALIZED VIEW', 'SYNONYM')
      AND objects.subobject_name IS NULL
    ORDER BY objects.owner, objects.object_name, objects.object_type, objects.object_id
)
WHERE ROWNUM <= :limit
