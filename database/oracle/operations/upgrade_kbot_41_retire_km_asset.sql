-- KBot 4.0 -> 4.1：从主产品 Schema 退役 KM Asset App。
--
-- 使用 KBot Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- 执行前必须：
--   1. 停止 KBot 4.0 KM API、Worker、Slack Worker 及其他 KBot 写入服务；
--   2. 使用 Data Pump 完成当前 Schema 的可恢复备份并验证 dump；
--   3. 确认目标后续运行 KBot 4.1，而 KM 继续使用独立的 km4.0 分支和数据库。
--
-- Oracle DDL 会自动提交，删除 KM 私有表后的回退方式是恢复升级前备份，不是 ROLLBACK。
-- 本脚本不删除共享 Domain、Knowledge Core、Data Query、Agent Runtime 历史或托管凭据；
-- 这些共享数据可能包含 KM 历史引用，保留它们可避免跨基础能力的数据损失。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK
SET SERVEROUTPUT ON
SET VERIFY OFF
SET SQLBLANKLINES ON
SET DEFINE OFF

ALTER SESSION SET TIME_ZONE = '+00:00';
ALTER SESSION SET DDL_LOCK_TIMEOUT = 60;

PROMPT === KBot 4.1 KM Asset 退役预检 ===

DECLARE
    l_missing_foundation_count PLS_INTEGER;
    l_unknown_object_count PLS_INTEGER;
    l_external_fk_count PLS_INTEGER;
    l_external_dependency_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_missing_foundation_count
      FROM (
          SELECT 'KBOT_PLATFORM_APP' object_name FROM DUAL
          UNION ALL SELECT 'KBOT_PLATFORM_USER' FROM DUAL
          UNION ALL SELECT 'KBOT_PLATFORM_USER_CREDENTIAL' FROM DUAL
          UNION ALL SELECT 'KBOT_PERMISSION' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_ROLE' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_ROLE_PERMISSION' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_DOMAIN' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_MEMBER' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_MEMBER_ROLE' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_MEMBER_ROLE_SCOPE' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_API_CLIENT' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_API_CREDENTIAL' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_API_CLIENT_SCOPE' FROM DUAL
          UNION ALL SELECT 'KBOT_APP_API_CLIENT_AGENT' FROM DUAL
          UNION ALL SELECT 'KBOT_PLATFORM_USER_ROLE' FROM DUAL
          UNION ALL SELECT 'KBOT_ACCESS_AUDIT_EVENT' FROM DUAL
      ) required_object
     WHERE NOT EXISTS (
         SELECT 1
           FROM USER_TABLES existing_table
          WHERE existing_table.TABLE_NAME = required_object.object_name
     );

    IF l_missing_foundation_count <> 0 THEN
        raise_application_error(
            -20610,
            '平台授权基础表不完整，拒绝执行 KBot 4.1 KM 退役升级'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_unknown_object_count
      FROM USER_OBJECTS object_row
     WHERE object_row.OBJECT_TYPE IN ('TABLE', 'VIEW')
       AND (
           object_row.OBJECT_NAME LIKE 'KBOT_KM\_%' ESCAPE '\'
           OR object_row.OBJECT_NAME LIKE 'KBOT_V_KM\_%' ESCAPE '\'
       )
       AND object_row.OBJECT_NAME NOT IN (
           'KBOT_KM_SOURCE',
           'KBOT_KM_ASSET',
           'KBOT_KM_ASSET_REVISION',
           'KBOT_KM_ATTACHMENT',
           'KBOT_KM_JOB',
           'KBOT_KM_AGENT',
           'KBOT_KM_AGENT_VERSION',
           'KBOT_KM_SLACK_INBOX',
           'KBOT_KM_SLACK_THREAD',
           'KBOT_KM_SLACK_DELIVERY',
           'KBOT_V_KM_ASSET_CURRENT',
           'KBOT_V_KM_ASSET_SEARCHABLE'
       );

    IF l_unknown_object_count <> 0 THEN
        raise_application_error(
            -20611,
            '发现不属于 KBot 4.0 清单的 KM 表或视图，拒绝猜测删除'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_external_fk_count
      FROM USER_CONSTRAINTS child_constraint
      JOIN USER_CONSTRAINTS parent_constraint
        ON parent_constraint.CONSTRAINT_NAME = child_constraint.R_CONSTRAINT_NAME
       AND parent_constraint.OWNER = child_constraint.R_OWNER
     WHERE child_constraint.CONSTRAINT_TYPE = 'R'
       AND parent_constraint.TABLE_NAME IN (
           'KBOT_KM_SOURCE',
           'KBOT_KM_ASSET',
           'KBOT_KM_ASSET_REVISION',
           'KBOT_KM_ATTACHMENT',
           'KBOT_KM_JOB',
           'KBOT_KM_AGENT',
           'KBOT_KM_AGENT_VERSION',
           'KBOT_KM_SLACK_INBOX',
           'KBOT_KM_SLACK_THREAD',
           'KBOT_KM_SLACK_DELIVERY'
       )
       AND child_constraint.TABLE_NAME NOT IN (
           'KBOT_KM_SOURCE',
           'KBOT_KM_ASSET',
           'KBOT_KM_ASSET_REVISION',
           'KBOT_KM_ATTACHMENT',
           'KBOT_KM_JOB',
           'KBOT_KM_AGENT',
           'KBOT_KM_AGENT_VERSION',
           'KBOT_KM_SLACK_INBOX',
           'KBOT_KM_SLACK_THREAD',
           'KBOT_KM_SLACK_DELIVERY'
       );

    IF l_external_fk_count <> 0 THEN
        raise_application_error(
            -20612,
            '存在非 KM 表指向 KM 私有表的外键，拒绝级联删除'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_external_dependency_count
      FROM USER_DEPENDENCIES dependency_row
     WHERE dependency_row.REFERENCED_NAME IN (
           'KBOT_KM_SOURCE',
           'KBOT_KM_ASSET',
           'KBOT_KM_ASSET_REVISION',
           'KBOT_KM_ATTACHMENT',
           'KBOT_KM_JOB',
           'KBOT_KM_AGENT',
           'KBOT_KM_AGENT_VERSION',
           'KBOT_KM_SLACK_INBOX',
           'KBOT_KM_SLACK_THREAD',
           'KBOT_KM_SLACK_DELIVERY'
       )
       AND dependency_row.NAME NOT IN (
           'KBOT_V_KM_ASSET_CURRENT',
           'KBOT_V_KM_ASSET_SEARCHABLE'
       )
       AND dependency_row.NAME NOT LIKE 'BIN$%';

    IF l_external_dependency_count <> 0 THEN
        raise_application_error(
            -20613,
            '存在非 KM 视图对象依赖 KM 私有表，拒绝继续升级'
        );
    END IF;

    dbms_output.put_line('预检通过：未发现未知 KM 对象或外部依赖。');
END;
/

PROMPT === 删除 KM 私有视图和表 ===

DECLARE
    PROCEDURE drop_view_if_exists(p_view_name VARCHAR2) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*) INTO l_count
          FROM USER_VIEWS
         WHERE VIEW_NAME = p_view_name;
        IF l_count = 1 THEN
            EXECUTE IMMEDIATE 'DROP VIEW ' || DBMS_ASSERT.SIMPLE_SQL_NAME(p_view_name);
            dbms_output.put_line('已删除视图 ' || p_view_name);
        END IF;
    END;

    PROCEDURE drop_table_if_exists(p_table_name VARCHAR2) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*) INTO l_count
          FROM USER_TABLES
         WHERE TABLE_NAME = p_table_name;
        IF l_count = 1 THEN
            EXECUTE IMMEDIATE
                'DROP TABLE ' || DBMS_ASSERT.SIMPLE_SQL_NAME(p_table_name)
                || ' CASCADE CONSTRAINTS PURGE';
            dbms_output.put_line('已删除表 ' || p_table_name);
        END IF;
    END;
BEGIN
    drop_view_if_exists('KBOT_V_KM_ASSET_SEARCHABLE');
    drop_view_if_exists('KBOT_V_KM_ASSET_CURRENT');

    drop_table_if_exists('KBOT_KM_SLACK_DELIVERY');
    drop_table_if_exists('KBOT_KM_SLACK_THREAD');
    drop_table_if_exists('KBOT_KM_SLACK_INBOX');
    drop_table_if_exists('KBOT_KM_JOB');
    drop_table_if_exists('KBOT_KM_ATTACHMENT');
    drop_table_if_exists('KBOT_KM_AGENT_VERSION');
    drop_table_if_exists('KBOT_KM_AGENT');
    drop_table_if_exists('KBOT_KM_ASSET_REVISION');
    drop_table_if_exists('KBOT_KM_ASSET');
    drop_table_if_exists('KBOT_KM_SOURCE');
END;
/

PROMPT === 删除 km_asset 授权边界和 App 自有账号 ===

DECLARE
    l_owned_user_count PLS_INTEGER := 0;
    l_app_member_count PLS_INTEGER := 0;
    l_app_count PLS_INTEGER := 0;
    l_remaining_count PLS_INTEGER := 0;
BEGIN
    SELECT COUNT(*) INTO l_owned_user_count
      FROM KBOT_PLATFORM_USER
     WHERE OWNER_APP_ID = 'km_asset';

    SELECT COUNT(*) INTO l_app_member_count
      FROM KBOT_APP_MEMBER
     WHERE APP_ID = 'km_asset';

    SELECT COUNT(*) INTO l_app_count
      FROM KBOT_PLATFORM_APP
     WHERE APP_ID = 'km_asset';

    DELETE FROM KBOT_APP_API_CLIENT_AGENT
     WHERE CLIENT_ID IN (
         SELECT client.CLIENT_ID
           FROM KBOT_APP_API_CLIENT client
          WHERE client.APP_ID = 'km_asset'
             OR client.SUBJECT_USER_ID IN (
                 SELECT user_account.USER_ID
                   FROM KBOT_PLATFORM_USER user_account
                  WHERE user_account.OWNER_APP_ID = 'km_asset'
             )
     );

    DELETE FROM KBOT_APP_API_CLIENT_SCOPE
     WHERE CLIENT_ID IN (
         SELECT client.CLIENT_ID
           FROM KBOT_APP_API_CLIENT client
          WHERE client.APP_ID = 'km_asset'
             OR client.SUBJECT_USER_ID IN (
                 SELECT user_account.USER_ID
                   FROM KBOT_PLATFORM_USER user_account
                  WHERE user_account.OWNER_APP_ID = 'km_asset'
             )
     );

    DELETE FROM KBOT_APP_API_CREDENTIAL
     WHERE CLIENT_ID IN (
         SELECT client.CLIENT_ID
           FROM KBOT_APP_API_CLIENT client
          WHERE client.APP_ID = 'km_asset'
             OR client.SUBJECT_USER_ID IN (
                 SELECT user_account.USER_ID
                   FROM KBOT_PLATFORM_USER user_account
                  WHERE user_account.OWNER_APP_ID = 'km_asset'
             )
     );

    DELETE FROM KBOT_APP_API_CLIENT client
     WHERE client.APP_ID = 'km_asset'
        OR client.SUBJECT_USER_ID IN (
            SELECT user_account.USER_ID
              FROM KBOT_PLATFORM_USER user_account
             WHERE user_account.OWNER_APP_ID = 'km_asset'
        );

    DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE scope_row
     WHERE scope_row.APP_ID = 'km_asset'
        OR scope_row.USER_ID IN (
            SELECT user_account.USER_ID
              FROM KBOT_PLATFORM_USER user_account
             WHERE user_account.OWNER_APP_ID = 'km_asset'
        );

    DELETE FROM KBOT_APP_MEMBER_ROLE role_binding
     WHERE role_binding.APP_ID = 'km_asset'
        OR role_binding.USER_ID IN (
            SELECT user_account.USER_ID
              FROM KBOT_PLATFORM_USER user_account
             WHERE user_account.OWNER_APP_ID = 'km_asset'
        );

    DELETE FROM KBOT_APP_MEMBER member_row
     WHERE member_row.APP_ID = 'km_asset'
        OR member_row.USER_ID IN (
            SELECT user_account.USER_ID
              FROM KBOT_PLATFORM_USER user_account
             WHERE user_account.OWNER_APP_ID = 'km_asset'
        );

    DELETE FROM KBOT_PLATFORM_USER_ROLE platform_role
     WHERE platform_role.USER_ID IN (
         SELECT user_account.USER_ID
           FROM KBOT_PLATFORM_USER user_account
          WHERE user_account.OWNER_APP_ID = 'km_asset'
     );

    DELETE FROM KBOT_PLATFORM_USER_CREDENTIAL credential
     WHERE credential.USER_ID IN (
         SELECT user_account.USER_ID
           FROM KBOT_PLATFORM_USER user_account
          WHERE user_account.OWNER_APP_ID = 'km_asset'
     );

    DELETE FROM KBOT_APP_ROLE_PERMISSION WHERE APP_ID = 'km_asset';
    DELETE FROM KBOT_PERMISSION WHERE APP_ID = 'km_asset';
    DELETE FROM KBOT_APP_ROLE WHERE APP_ID = 'km_asset';
    DELETE FROM KBOT_APP_DOMAIN WHERE APP_ID = 'km_asset';
    DELETE FROM KBOT_PLATFORM_USER WHERE OWNER_APP_ID = 'km_asset';
    DELETE FROM KBOT_PLATFORM_APP WHERE APP_ID = 'km_asset';

    SELECT SUM(item_count)
      INTO l_remaining_count
      FROM (
          SELECT COUNT(*) item_count
            FROM KBOT_PLATFORM_APP WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_PLATFORM_USER
           WHERE OWNER_APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_PERMISSION WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_ROLE WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_DOMAIN WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_MEMBER WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_MEMBER_ROLE WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_MEMBER_ROLE_SCOPE
           WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_API_CLIENT WHERE APP_ID = 'km_asset'
      );

    IF l_remaining_count <> 0 THEN
        raise_application_error(-20614, 'km_asset 授权数据删除不完整');
    END IF;

    IF l_app_count > 0 OR l_owned_user_count > 0 OR l_app_member_count > 0 THEN
        INSERT INTO KBOT_ACCESS_AUDIT_EVENT (
            EVENT_ID, ACTOR_ID, APP_ID, ACTION, RESOURCE_TYPE,
            RESOURCE_ID, SUMMARY_JSON, AFFECTED_MEMBER_COUNT, CREATED_AT
        ) VALUES (
            RAWTOHEX(SYS_GUID()),
            'migration:kbot-4.1-km-retirement',
            'km_asset',
            'RETIRED_APP_AUTHORIZATION_PURGED',
            'PLATFORM_APP',
            'km_asset',
            '{"physical_delete":true,"shared_foundation_retained":true,'
                || '"owned_user_count":' || l_owned_user_count || '}',
            l_app_member_count,
            SYSTIMESTAMP
        );
    END IF;

    COMMIT;
    dbms_output.put_line(
        'km_asset 授权边界已删除：App 自有用户=' || l_owned_user_count
        || '，App 成员=' || l_app_member_count
    );
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
/

PROMPT === 验证 KBot 4.1 KM Asset 已退役 ===

DECLARE
    l_private_object_count PLS_INTEGER;
    l_authorization_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_private_object_count
      FROM USER_OBJECTS object_row
     WHERE object_row.OBJECT_TYPE IN ('TABLE', 'VIEW')
       AND (
           object_row.OBJECT_NAME LIKE 'KBOT_KM\_%' ESCAPE '\'
           OR object_row.OBJECT_NAME LIKE 'KBOT_V_KM\_%' ESCAPE '\'
       );

    SELECT SUM(item_count)
      INTO l_authorization_count
      FROM (
          SELECT COUNT(*) item_count
            FROM KBOT_PLATFORM_APP WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_PLATFORM_USER
           WHERE OWNER_APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_PERMISSION WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_ROLE WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_DOMAIN WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_MEMBER WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_MEMBER_ROLE WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_MEMBER_ROLE_SCOPE
           WHERE APP_ID = 'km_asset'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_API_CLIENT WHERE APP_ID = 'km_asset'
      );

    IF l_private_object_count <> 0 OR l_authorization_count <> 0 THEN
        raise_application_error(-20615, 'KBot 4.1 KM Asset 退役验证失败');
    END IF;

    dbms_output.put_line(
        'KBot 4.1 KM Asset 退役验证通过；共享 Domain、KC、DQ、Agent Runtime 和凭据未删除。'
    );
END;
/

PROMPT === KBot 4.1 KM Asset Schema 升级完成 ===
