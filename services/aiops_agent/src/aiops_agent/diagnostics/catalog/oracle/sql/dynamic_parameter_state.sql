SELECT p.name AS parameter_name,
       p.display_value AS current_value,
       p.issys_modifiable,
       p.isinstance_modifiable,
       UPPER(:parameter_value) AS requested_value
  FROM v$parameter p
 WHERE p.name = LOWER(:parameter_name)
   AND p.issys_modifiable = 'IMMEDIATE'
