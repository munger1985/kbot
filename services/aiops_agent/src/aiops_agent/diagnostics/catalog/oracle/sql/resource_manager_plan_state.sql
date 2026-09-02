SELECT p.plan AS resource_plan_name,
       p.status,
       p.mandatory,
       NVL((SELECT value
              FROM v$parameter
             WHERE name = 'resource_manager_plan'), '<NONE>') AS current_plan_name
  FROM dba_rsrc_plans p
 WHERE p.plan = UPPER(:resource_plan_name)
   AND p.status = 'ACTIVE'
