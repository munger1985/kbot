SELECT schema_name, table_name, relation_name, index_name,
       column_position, column_name, index_status, partitioned
FROM (
    SELECT tables.owner AS schema_name, tables.table_name,
           tables.table_name AS relation_name, indexes.index_name,
           columns.column_position, columns.column_name,
           indexes.status AS index_status, indexes.partitioned
    FROM all_tables tables
    LEFT JOIN all_indexes indexes
      ON indexes.table_owner = tables.owner
     AND indexes.table_name = tables.table_name
    LEFT JOIN all_ind_columns columns
      ON columns.index_owner = indexes.owner
     AND columns.index_name = indexes.index_name
    WHERE tables.owner = :schema_name AND tables.table_name = :table_name
    ORDER BY indexes.index_name, columns.column_position
)
WHERE ROWNUM <= :limit
