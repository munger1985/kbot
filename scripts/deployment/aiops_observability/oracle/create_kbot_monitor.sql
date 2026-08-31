-- KBot AIOps Oracle诊断用户初始化脚本。
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
            '禁止在CDB$ROOT创建诊断用户，请先切换到实际被监控PDB'
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

-- AIOps使用专用只读账号诊断数据库，包括V$/GV$、DBA/CDB目录以及AWR/ASH视图。
GRANT CREATE SESSION TO kbot_monitor;
GRANT SELECT ANY DICTIONARY TO kbot_monitor;

PROMPT 已创建kbot_monitor并完成数据库诊断授权。
PROMPT 正在验证系统权限……

DECLARE
    system_grant_count NUMBER;
BEGIN
    SELECT COUNT(*)
    INTO system_grant_count
    FROM DBA_SYS_PRIVS
    WHERE grantee = 'KBOT_MONITOR'
      AND privilege IN ('CREATE SESSION', 'SELECT ANY DICTIONARY');

    IF system_grant_count <> 2 THEN
        RAISE_APPLICATION_ERROR(-20003, 'kbot_monitor系统权限验证失败');
    END IF;
END;
/

SELECT privilege
FROM DBA_SYS_PRIVS
WHERE grantee = 'KBOT_MONITOR'
  AND privilege IN ('CREATE SESSION', 'SELECT ANY DICTIONARY')
ORDER BY privilege;

EXIT SUCCESS
