SELECT schema_name, table_name, relation_kind,
       dependency_schema_name, dependency_name, dependency_kind, reference_scope
FROM (
    SELECT dependencies.owner AS schema_name, dependencies.name AS table_name,
           dependencies.type AS relation_kind,
           dependencies.referenced_owner AS dependency_schema_name,
           dependencies.referenced_name AS dependency_name,
           dependencies.referenced_type AS dependency_kind,
           CASE WHEN dependencies.referenced_link_name IS NULL THEN 'LOCAL' ELSE 'REMOTE' END AS reference_scope
    FROM all_dependencies dependencies
    WHERE dependencies.owner = :schema_name AND dependencies.name = :table_name
      AND dependencies.type IN ('VIEW', 'MATERIALIZED VIEW')
      AND dependencies.referenced_type IN ('TABLE', 'VIEW', 'MATERIALIZED VIEW', 'SYNONYM')
    UNION
    SELECT synonyms.owner AS schema_name, synonyms.synonym_name AS table_name,
           'SYNONYM' AS relation_kind,
           synonyms.table_owner AS dependency_schema_name,
           synonyms.table_name AS dependency_name,
           'UNKNOWN' AS dependency_kind,
           CASE WHEN synonyms.db_link IS NULL THEN 'LOCAL' ELSE 'REMOTE' END AS reference_scope
    FROM all_synonyms synonyms
    WHERE synonyms.owner = :schema_name AND synonyms.synonym_name = :table_name
    ORDER BY dependency_schema_name, dependency_name, relation_kind
)
WHERE ROWNUM <= :limit
