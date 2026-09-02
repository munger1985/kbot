SELECT owner,
       object_name,
       object_type,
       status,
       last_ddl_time
  FROM dba_objects
 WHERE owner = UPPER(:schema_name)
   AND object_name = UPPER(:object_name)
   AND object_type = UPPER(:object_type)
