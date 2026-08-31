-- KBot AIOps Oracle监控用户初始化脚本。
-- 使用SYSDBA连接到实际被监控PDB后执行，禁止在CDB$ROOT运行。

SET ECHO OFF
SET FEEDBACK ON
SET HEADING ON
SET VERIFY OFF
WHENEVER SQLERROR EXIT SQL.SQLCODE

DECLARE
    current_container VARCHAR2(128);
    current_user_name VARCHAR2(128);
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
            '禁止在CDB$ROOT创建监控用户，请先切换到实际被监控PDB'
        );
    END IF;
END;
/

PROMPT 当前容器：
SELECT SYS_CONTEXT('USERENV', 'CON_NAME') AS CURRENT_CONTAINER FROM DUAL;

ACCEPT KBOT_MONITOR_PASSWORD CHAR PROMPT '请输入kbot_monitor密码（不得包含双引号）：' HIDE

CREATE USER kbot_monitor
    IDENTIFIED BY "&KBOT_MONITOR_PASSWORD"
    ACCOUNT UNLOCK;

UNDEFINE KBOT_MONITOR_PASSWORD

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

-- KBot AIOps实例身份及当前累计Top SQL诊断；不包含AWR/Diagnostics Pack历史视图。
GRANT SELECT ON SYS.V_$INSTANCE                   TO kbot_monitor;
GRANT SELECT ON SYS.GV_$INSTANCE                  TO kbot_monitor;
GRANT SELECT ON SYS.V_$DATABASE                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$VERSION                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$OPTION                     TO kbot_monitor;
GRANT SELECT ON SYS.V_$PDBS                       TO kbot_monitor;
GRANT SELECT ON SYS.V_$CONTAINERS                 TO kbot_monitor;
GRANT SELECT ON SYS.V_$SERVICES                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$DIAG_INFO                  TO kbot_monitor;
GRANT SELECT ON SYS.V_$SQLSTATS                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$SQL_PLAN                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$SQL_WORKAREA_ACTIVE        TO kbot_monitor;

-- KBot AIOps当前配置、持久化配置和参数有效值诊断。
GRANT SELECT ON SYS.V_$SYSTEM_PARAMETER           TO kbot_monitor;
GRANT SELECT ON SYS.V_$SPPARAMETER                TO kbot_monitor;
GRANT SELECT ON SYS.V_$PARAMETER_VALID_VALUES     TO kbot_monitor;

-- KBot AIOps当前活跃会话、阻塞链和表空间容量诊断。
GRANT SELECT ON SYS.GV_$SESSION                   TO kbot_monitor;
GRANT SELECT ON SYS.GV_$PROCESS                   TO kbot_monitor;
GRANT SELECT ON SYS.GV_$TRANSACTION               TO kbot_monitor;
GRANT SELECT ON SYS.V_$TRANSACTION                TO kbot_monitor;
GRANT SELECT ON SYS.V_$LOCK                       TO kbot_monitor;
GRANT SELECT ON SYS.GV_$LOCK                      TO kbot_monitor;
GRANT SELECT ON SYS.V_$LOCKED_OBJECT              TO kbot_monitor;
GRANT SELECT ON SYS.V_$SESSION_EVENT              TO kbot_monitor;
GRANT SELECT ON SYS.V_$SESSTAT                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$STATNAME                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$SESSION_LONGOPS            TO kbot_monitor;
GRANT SELECT ON SYS.DBA_DATA_FILES                TO kbot_monitor;
GRANT SELECT ON SYS.DBA_FREE_SPACE                TO kbot_monitor;
GRANT SELECT ON SYS.DBA_SEGMENTS                  TO kbot_monitor;
GRANT SELECT ON SYS.DBA_TAB_STATISTICS            TO kbot_monitor;
GRANT SELECT ON SYS.DBA_IND_STATISTICS            TO kbot_monitor;

-- KBot AIOps实时性能、内存和归档/FRA诊断。
GRANT SELECT ON SYS.V_$SYSMETRIC                  TO kbot_monitor;
GRANT SELECT ON SYS.V_$RSRCPDBMETRIC              TO kbot_monitor;
GRANT SELECT ON SYS.V_$PARAMETER                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$OSSTAT                     TO kbot_monitor;
GRANT SELECT ON SYS.V_$SYSTEM_EVENT               TO kbot_monitor;
GRANT SELECT ON SYS.GV_$SYSTEM_EVENT              TO kbot_monitor;
GRANT SELECT ON SYS.GV_$SYSSTAT                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$SYS_TIME_MODEL             TO kbot_monitor;
GRANT SELECT ON SYS.V_$EVENTMETRIC                TO kbot_monitor;
GRANT SELECT ON SYS.V_$FILEMETRIC                 TO kbot_monitor;
GRANT SELECT ON SYS.V_$IOSTAT_FILE                TO kbot_monitor;
GRANT SELECT ON SYS.V_$IOSTAT_FUNCTION            TO kbot_monitor;
GRANT SELECT ON SYS.V_$SGA                        TO kbot_monitor;
GRANT SELECT ON SYS.V_$SGAINFO                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$SGASTAT                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$PGASTAT                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$MEMORY_DYNAMIC_COMPONENTS  TO kbot_monitor;
GRANT SELECT ON SYS.V_$MEMORY_RESIZE_OPS          TO kbot_monitor;
GRANT SELECT ON SYS.V_$SGA_TARGET_ADVICE          TO kbot_monitor;
GRANT SELECT ON SYS.V_$PGA_TARGET_ADVICE          TO kbot_monitor;
GRANT SELECT ON SYS.V_$RECOVERY_FILE_DEST         TO kbot_monitor;
GRANT SELECT ON SYS.V_$FLASH_RECOVERY_AREA_USAGE  TO kbot_monitor;

-- KBot AIOps TEMP/UNDO、Redo、维护、备份和Data Guard诊断。
GRANT SELECT ON SYS.DBA_TEMP_FILES                TO kbot_monitor;
GRANT SELECT ON SYS.V_$TEMP_SPACE_HEADER          TO kbot_monitor;
GRANT SELECT ON SYS.DBA_UNDO_EXTENTS              TO kbot_monitor;
GRANT SELECT ON SYS.V_$TEMPFILE                   TO kbot_monitor;
GRANT SELECT ON SYS.V_$CONTROLFILE                TO kbot_monitor;
GRANT SELECT ON SYS.V_$LOG                        TO kbot_monitor;
GRANT SELECT ON SYS.V_$LOGFILE                    TO kbot_monitor;
GRANT SELECT ON SYS.V_$STANDBY_LOG                TO kbot_monitor;
GRANT SELECT ON SYS.V_$ARCHIVED_LOG               TO kbot_monitor;
GRANT SELECT ON SYS.V_$ARCHIVE_DEST               TO kbot_monitor;
GRANT SELECT ON SYS.V_$ARCHIVE_DEST_STATUS        TO kbot_monitor;
GRANT SELECT ON SYS.V_$ARCHIVE_GAP                TO kbot_monitor;
GRANT SELECT ON SYS.DBA_SCHEDULER_JOB_RUN_DETAILS TO kbot_monitor;
GRANT SELECT ON SYS.DBA_SCHEDULER_JOBS            TO kbot_monitor;
GRANT SELECT ON SYS.DBA_SCHEDULER_RUNNING_JOBS    TO kbot_monitor;
GRANT SELECT ON SYS.DBA_AUTOTASK_CLIENT           TO kbot_monitor;
GRANT SELECT ON SYS.DBA_AUTOTASK_JOB_HISTORY      TO kbot_monitor;
GRANT SELECT ON SYS.DBA_OBJECTS                   TO kbot_monitor;
GRANT SELECT ON SYS.DBA_ERRORS                    TO kbot_monitor;
GRANT SELECT ON SYS.DBA_REGISTRY                  TO kbot_monitor;
GRANT SELECT ON SYS.V_$RMAN_BACKUP_JOB_DETAILS    TO kbot_monitor;
GRANT SELECT ON SYS.V_$BACKUP_SET                 TO kbot_monitor;
GRANT SELECT ON SYS.V_$BACKUP_PIECE               TO kbot_monitor;
GRANT SELECT ON SYS.V_$RECOVER_FILE               TO kbot_monitor;
GRANT SELECT ON SYS.V_$DATAFILE_HEADER            TO kbot_monitor;
GRANT SELECT ON SYS.V_$DATABASE_BLOCK_CORRUPTION  TO kbot_monitor;
GRANT SELECT ON SYS.V_$BLOCK_CHANGE_TRACKING      TO kbot_monitor;
GRANT SELECT ON SYS.V_$DATAGUARD_STATS            TO kbot_monitor;
GRANT SELECT ON SYS.V_$DATAGUARD_PROCESS          TO kbot_monitor;

-- KBot AIOps账号状态、Profile和授权缺口诊断；不读取SYS.USER$等内部基表。
GRANT SELECT ON SYS.DBA_USERS                     TO kbot_monitor;
GRANT SELECT ON SYS.DBA_PROFILES                  TO kbot_monitor;
GRANT SELECT ON SYS.DBA_ROLE_PRIVS                TO kbot_monitor;
GRANT SELECT ON SYS.DBA_SYS_PRIVS                 TO kbot_monitor;
GRANT SELECT ON SYS.DBA_TAB_PRIVS                 TO kbot_monitor;
GRANT SELECT ON SYS.DBA_COL_PRIVS                 TO kbot_monitor;

-- KBot Oracle Alert Collector。
GRANT SELECT ON SYS.V_$DIAG_ALERT_EXT             TO kbot_monitor;

PROMPT 已创建kbot_monitor并完成最小权限授权。
PROMPT 正在验证授权清单……

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
          'GV_$INSTANCE',
          'V_$DATABASE',
          'V_$VERSION',
          'V_$OPTION',
          'V_$PDBS',
          'V_$CONTAINERS',
          'V_$SERVICES',
          'V_$DIAG_INFO',
          'V_$SQLSTATS',
          'V_$SQL_PLAN',
          'V_$SQL_WORKAREA_ACTIVE',
          'V_$SYSTEM_PARAMETER',
          'V_$SPPARAMETER',
          'V_$PARAMETER_VALID_VALUES',
          'GV_$SESSION',
          'GV_$PROCESS',
          'GV_$TRANSACTION',
          'V_$TRANSACTION',
          'V_$LOCK',
          'GV_$LOCK',
          'V_$LOCKED_OBJECT',
          'V_$SESSION_EVENT',
          'V_$SESSTAT',
          'V_$STATNAME',
          'V_$SESSION_LONGOPS',
          'DBA_DATA_FILES',
          'DBA_FREE_SPACE',
          'DBA_SEGMENTS',
          'DBA_TAB_STATISTICS',
          'DBA_IND_STATISTICS',
          'V_$SYSMETRIC',
          'V_$RSRCPDBMETRIC',
          'V_$PARAMETER',
          'V_$OSSTAT',
          'V_$SYSTEM_EVENT',
          'GV_$SYSTEM_EVENT',
          'GV_$SYSSTAT',
          'V_$SYS_TIME_MODEL',
          'V_$EVENTMETRIC',
          'V_$FILEMETRIC',
          'V_$IOSTAT_FILE',
          'V_$IOSTAT_FUNCTION',
          'V_$SGA',
          'V_$SGAINFO',
          'V_$SGASTAT',
          'V_$PGASTAT',
          'V_$MEMORY_DYNAMIC_COMPONENTS',
          'V_$MEMORY_RESIZE_OPS',
          'V_$SGA_TARGET_ADVICE',
          'V_$PGA_TARGET_ADVICE',
          'V_$RECOVERY_FILE_DEST',
          'V_$FLASH_RECOVERY_AREA_USAGE',
          'DBA_TEMP_FILES',
          'V_$TEMP_SPACE_HEADER',
          'DBA_UNDO_EXTENTS',
          'V_$TEMPFILE',
          'V_$CONTROLFILE',
          'V_$LOG',
          'V_$LOGFILE',
          'V_$STANDBY_LOG',
          'V_$ARCHIVED_LOG',
          'V_$ARCHIVE_DEST',
          'V_$ARCHIVE_DEST_STATUS',
          'V_$ARCHIVE_GAP',
          'DBA_SCHEDULER_JOB_RUN_DETAILS',
          'DBA_SCHEDULER_JOBS',
          'DBA_SCHEDULER_RUNNING_JOBS',
          'DBA_AUTOTASK_CLIENT',
          'DBA_AUTOTASK_JOB_HISTORY',
          'DBA_OBJECTS',
          'DBA_ERRORS',
          'DBA_REGISTRY',
          'V_$RMAN_BACKUP_JOB_DETAILS',
          'V_$BACKUP_SET',
          'V_$BACKUP_PIECE',
          'V_$RECOVER_FILE',
          'V_$DATAFILE_HEADER',
          'V_$DATABASE_BLOCK_CORRUPTION',
          'V_$BLOCK_CHANGE_TRACKING',
          'V_$DATAGUARD_STATS',
          'V_$DATAGUARD_PROCESS',
          'DBA_USERS',
          'DBA_PROFILES',
          'DBA_ROLE_PRIVS',
          'DBA_SYS_PRIVS',
          'DBA_TAB_PRIVS',
          'DBA_COL_PRIVS',
          'V_$DIAG_ALERT_EXT'
      );

    IF session_grant_count <> 1 OR object_grant_count <> 99 THEN
        RAISE_APPLICATION_ERROR(-20003, 'kbot_monitor授权清单验证失败');
    END IF;
END;
/

EXIT SUCCESS
