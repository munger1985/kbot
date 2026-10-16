-- KBot 4.1：收敛跨 App 共享平台视图。
-- 仅重建视图，不修改 KBOT_PLATFORM_DOMAIN 业务数据。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK
SET SERVEROUTPUT ON
SET VERIFY OFF
SET SQLBLANKLINES ON
SET DEFINE OFF

PROMPT === 检查共享平台 Domain 表合同 ===

DECLARE
    l_required_column_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_required_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_PLATFORM_DOMAIN'
       AND COLUMN_NAME IN (
           'DOMAIN_ID', 'NAME', 'STATUS', 'DESCRIPTION', 'ROW_VERSION',
           'CREATED_AT', 'UPDATED_AT'
       );
    IF l_required_column_count <> 7 THEN
        raise_application_error(
            -20620,
            'KBOT_PLATFORM_DOMAIN 缺少 KBot 4.1 共享视图所需列'
        );
    END IF;
END;
/

CREATE OR REPLACE VIEW KBOT_V_PLATFORM_DOMAIN AS
SELECT
    DOMAIN_ID,
    NAME,
    STATUS,
    DESCRIPTION,
    ROW_VERSION,
    CREATED_AT,
    UPDATED_AT
FROM KBOT_PLATFORM_DOMAIN;

DECLARE
    l_status VARCHAR2(16 CHAR);
BEGIN
    SELECT STATUS
      INTO l_status
      FROM USER_OBJECTS
     WHERE OBJECT_NAME = 'KBOT_V_PLATFORM_DOMAIN'
       AND OBJECT_TYPE = 'VIEW';
    IF l_status <> 'VALID' THEN
        raise_application_error(-20621, 'KBOT_V_PLATFORM_DOMAIN 重建后仍无效');
    END IF;
END;
/

PROMPT === KBot 4.1 共享平台视图已收敛 ===
