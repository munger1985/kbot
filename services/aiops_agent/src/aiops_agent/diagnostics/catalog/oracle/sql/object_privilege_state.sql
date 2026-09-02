SELECT o.owner,
       o.object_name,
       o.object_type,
       u.username AS grantee_name,
       UPPER(:privilege) AS privilege,
       CASE WHEN p.privilege IS NULL THEN 'NO' ELSE 'YES' END AS is_granted,
       u.oracle_maintained,
       u.common
  FROM dba_objects o
  JOIN dba_users u
    ON u.username = UPPER(:grantee_name)
  LEFT JOIN dba_tab_privs p
    ON p.owner = o.owner
   AND p.table_name = o.object_name
   AND p.grantee = u.username
   AND p.privilege = UPPER(:privilege)
 WHERE o.owner = UPPER(:schema_name)
   AND o.object_name = UPPER(:object_name)
   AND o.object_type = UPPER(:object_type)
   AND o.status = 'VALID'
   AND u.oracle_maintained = 'N'
   AND u.common = 'NO'
