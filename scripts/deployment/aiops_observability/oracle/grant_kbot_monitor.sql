-- 已有KBot AIOps Oracle诊断用户的完整授权脚本。
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

-- AIOps使用专用只读账号诊断数据库，包括V$/GV$、DBA/CDB目录以及AWR/ASH视图。
GRANT CREATE SESSION TO kbot_monitor;
GRANT SELECT ANY DICTIONARY TO kbot_monitor;

PROMPT 完整诊断授权已执行，正在验证系统权限……

DECLARE
    system_grant_count NUMBER;
BEGIN
    SELECT COUNT(*)
    INTO system_grant_count
    FROM DBA_SYS_PRIVS
    WHERE grantee = 'KBOT_MONITOR'
      AND privilege IN ('CREATE SESSION', 'SELECT ANY DICTIONARY');

    IF system_grant_count <> 2 THEN
        RAISE_APPLICATION_ERROR(
            -20004,
            'kbot_monitor系统权限验证失败'
        );
    END IF;
END;
/

PROMPT kbot_monitor数据库诊断权限验证通过。

SELECT privilege
FROM DBA_SYS_PRIVS
WHERE grantee = 'KBOT_MONITOR'
  AND privilege IN ('CREATE SESSION', 'SELECT ANY DICTIONARY')
ORDER BY privilege;

EXIT SUCCESS
