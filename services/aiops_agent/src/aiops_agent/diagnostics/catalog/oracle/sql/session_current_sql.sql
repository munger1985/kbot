SELECT s.inst_id AS instance_id,
       s.sid AS session_id,
       s.serial# AS serial_number,
       s.sql_id,
       s.username,
       s.status
  FROM gv$session s
 WHERE s.inst_id = :instance_id
   AND s.sid = :session_id
   AND s.type = 'USER'
