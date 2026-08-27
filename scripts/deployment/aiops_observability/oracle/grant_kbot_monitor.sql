-- 已有KBot AIOps Oracle监控用户的完整授权脚本。
-- 使用SYSDBA连接到实际被监控PDB后执行，禁止在CDB$ROOT运行。

SET ECHO OFF
SET FEEDBACK ON
SET HEADING ON
SET VERIFY OFF
WHENEVER SQLERROR EXIT SQL.SQLCODE

DECLARE
    current_container VARCHAR2(128);
    current_user_name VARCHAR2(128);
    monitor_user_count NUMBER;
BEGIN
    current_container := SYS_CONTEXT('USERENV', 'CON_NAME');
    current_user_name := SYS_CONTEXT('USERENV', 'CURRENT_USER');

    IF current_user_name <> 'SYS' THEN
        RAISE_APPLICATION_ERROR(
            -20001,
            '必须使用SYSDBA执行本脚本，当前用户为' || current_user_name
        );
    END IF;

    IF current_container = 'CDB$ROOT' THEN
        RAISE_APPLICATION_ERROR(
            -20002,
            '禁止在CDB$ROOT授权，请先切换到实际被监控PDB'
        );
    END IF;

    SELECT COUNT(*)
    INTO monitor_user_count
    FROM DBA_USERS
    WHERE USERNAME = 'KBOT_MONITOR';

    IF monitor_user_count <> 1 THEN
        RAISE_APPLICATION_ERROR(
            -20003,
            '当前PDB不存在kbot_monitor，请先执行建用户脚本'
        );
    END IF;
END;
/

PROMPT 当前容器：
SELECT SYS_CONTEXT('USERENV', 'CON_NAME') AS CURRENT_CONTAINER FROM DUAL;

GRANT CREATE SESSION TO kbot_monitor;

-- Oracle Exporter默认指标。
GRANT SELECT ON SYS.DBA_TABLESPACE_USAGE_METRICS TO kbot_monitor;
GRANT SELECT ON SYS.DBA_TABLESPACES               TO kbot_monitor;
GRANT SELECT ON SYS.V_$SYSTEM_WAIT_CLASS          TO kbot_monitor;
GRANT SELECT ON SYS.V_$ASM_DISKGROUP_STAT         TO kbot_monitor;
GRANT SELECT ON SYS.V_$DATAFILE                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$SYSSTAT                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$PROCESS                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$WAITCLASSMETRIC            TO kbot_monitor;
GRANT SELECT ON SYS.V_$SESSION                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$RESOURCE_LIMIT             TO kbot_monitor;

-- KBot AIOps实例身份及当前累计Top SQL诊断。
GRANT SELECT ON SYS.V_$INSTANCE                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$DATABASE                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$SQLSTATS                   TO kbot_monitor;

-- KBot AIOps当前会话、阻塞链、长事务和表空间容量诊断。
GRANT SELECT ON SYS.GV_$SESSION                   TO kbot_monitor;
GRANT SELECT ON SYS.GV_$TRANSACTION               TO kbot_monitor;
GRANT SELECT ON SYS.DBA_DATA_FILES                TO kbot_monitor;
GRANT SELECT ON SYS.DBA_FREE_SPACE                TO kbot_monitor;

-- KBot AIOps实时性能、内存和归档/FRA诊断。
GRANT SELECT ON SYS.V_$SYSMETRIC                  TO kbot_monitor;
GRANT SELECT ON SYS.V_$RSRCPDBMETRIC              TO kbot_monitor;
GRANT SELECT ON SYS.V_$PARAMETER                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$SGA                        TO kbot_monitor;
GRANT SELECT ON SYS.V_$PGASTAT                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$RECOVERY_FILE_DEST         TO kbot_monitor;

-- KBot Oracle Alert Collector。
GRANT SELECT ON SYS.V_$DIAG_ALERT_EXT             TO kbot_monitor;

PROMPT 完整授权已执行，正在验证授权清单……

DECLARE
    session_grant_count NUMBER;
    object_grant_count NUMBER;
BEGIN
    SELECT COUNT(*)
    INTO session_grant_count
    FROM DBA_SYS_PRIVS
    WHERE grantee = 'KBOT_MONITOR'
      AND privilege = 'CREATE SESSION';

    SELECT COUNT(*)
    INTO object_grant_count
    FROM DBA_TAB_PRIVS
    WHERE grantee = 'KBOT_MONITOR'
      AND owner = 'SYS'
      AND privilege = 'SELECT'
      AND table_name IN (
          'DBA_TABLESPACE_USAGE_METRICS',
          'DBA_TABLESPACES',
          'V_$SYSTEM_WAIT_CLASS',
          'V_$ASM_DISKGROUP_STAT',
          'V_$DATAFILE',
          'V_$SYSSTAT',
          'V_$PROCESS',
          'V_$WAITCLASSMETRIC',
          'V_$SESSION',
          'V_$RESOURCE_LIMIT',
          'V_$INSTANCE',
          'V_$DATABASE',
          'V_$SQLSTATS',
          'GV_$SESSION',
          'GV_$TRANSACTION',
          'DBA_DATA_FILES',
          'DBA_FREE_SPACE',
          'V_$SYSMETRIC',
          'V_$RSRCPDBMETRIC',
          'V_$PARAMETER',
          'V_$SGA',
          'V_$PGASTAT',
          'V_$RECOVERY_FILE_DEST',
          'V_$DIAG_ALERT_EXT'
      );

    IF session_grant_count <> 1 OR object_grant_count <> 24 THEN
        RAISE_APPLICATION_ERROR(
            -20004,
            'kbot_monitor完整授权清单验证失败'
        );
    END IF;
END;
/

PROMPT kbot_monitor完整授权清单验证通过。

SELECT owner, table_name, privilege
FROM DBA_TAB_PRIVS
WHERE grantee = 'KBOT_MONITOR'
  AND owner = 'SYS'
  AND privilege = 'SELECT'
ORDER BY table_name;

EXIT SUCCESS
