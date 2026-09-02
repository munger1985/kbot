SELECT u.username AS grantee_name,
       UPPER(:privilege) AS privilege,
       CASE WHEN p.privilege IS NULL THEN 'NO' ELSE 'YES' END AS is_granted,
       u.oracle_maintained,
       u.common
  FROM dba_users u
  LEFT JOIN dba_sys_privs p
    ON p.grantee = u.username
   AND p.privilege = UPPER(:privilege)
 WHERE u.username = UPPER(:grantee_name)
   AND u.oracle_maintained = 'N'
   AND u.common = 'NO'
