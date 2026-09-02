SELECT username,
       account_status,
       lock_date,
       expiry_date,
       profile,
       authentication_type,
       oracle_maintained,
       common
  FROM dba_users
 WHERE username = UPPER(:username)
   AND oracle_maintained = 'N'
   AND common = 'NO'
