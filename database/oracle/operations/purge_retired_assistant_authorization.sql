-- 物理删除已退役 assistant App 的授权数据和 App 自有用户。
-- 使用 SQL Developer Run Script（F5）执行。
-- 脚本会同时删除 assistant 自有用户在其他 App 的显式授权和 API Client，
-- 但不删除其他 App 自有用户或平台用户。任何步骤失败都整体回滚。

WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;
SET SERVEROUTPUT ON;

DECLARE
    l_owned_user_count NUMBER := 0;
    l_app_member_count NUMBER := 0;
    l_remaining_count NUMBER := 0;
    l_target_count NUMBER := 0;
BEGIN
    SELECT COUNT(*) INTO l_owned_user_count
      FROM KBOT_PLATFORM_USER
     WHERE OWNER_APP_ID = 'assistant';

    SELECT COUNT(*) INTO l_app_member_count
      FROM KBOT_APP_MEMBER
     WHERE APP_ID = 'assistant';

    SELECT COUNT(*) INTO l_target_count
      FROM KBOT_PLATFORM_APP
     WHERE APP_ID = 'assistant';

    DELETE FROM KBOT_APP_API_CLIENT_AGENT
     WHERE CLIENT_ID IN (
         SELECT client.CLIENT_ID
           FROM KBOT_APP_API_CLIENT client
          WHERE client.APP_ID = 'assistant'
             OR client.SUBJECT_USER_ID IN (
                 SELECT user_account.USER_ID
                   FROM KBOT_PLATFORM_USER user_account
                  WHERE user_account.OWNER_APP_ID = 'assistant'
             )
     );

    DELETE FROM KBOT_APP_API_CLIENT_SCOPE
     WHERE CLIENT_ID IN (
         SELECT client.CLIENT_ID
           FROM KBOT_APP_API_CLIENT client
          WHERE client.APP_ID = 'assistant'
             OR client.SUBJECT_USER_ID IN (
                 SELECT user_account.USER_ID
                   FROM KBOT_PLATFORM_USER user_account
                  WHERE user_account.OWNER_APP_ID = 'assistant'
             )
     );

    DELETE FROM KBOT_APP_API_CREDENTIAL
     WHERE CLIENT_ID IN (
         SELECT client.CLIENT_ID
           FROM KBOT_APP_API_CLIENT client
          WHERE client.APP_ID = 'assistant'
             OR client.SUBJECT_USER_ID IN (
                 SELECT user_account.USER_ID
                   FROM KBOT_PLATFORM_USER user_account
                  WHERE user_account.OWNER_APP_ID = 'assistant'
             )
     );

    DELETE FROM KBOT_APP_API_CLIENT client
     WHERE client.APP_ID = 'assistant'
        OR client.SUBJECT_USER_ID IN (
            SELECT user_account.USER_ID
              FROM KBOT_PLATFORM_USER user_account
             WHERE user_account.OWNER_APP_ID = 'assistant'
        );

    DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE scope_row
     WHERE scope_row.APP_ID = 'assistant'
        OR scope_row.USER_ID IN (
            SELECT user_account.USER_ID
              FROM KBOT_PLATFORM_USER user_account
             WHERE user_account.OWNER_APP_ID = 'assistant'
        );

    DELETE FROM KBOT_APP_MEMBER_ROLE role_binding
     WHERE role_binding.APP_ID = 'assistant'
        OR role_binding.USER_ID IN (
            SELECT user_account.USER_ID
              FROM KBOT_PLATFORM_USER user_account
             WHERE user_account.OWNER_APP_ID = 'assistant'
        );

    DELETE FROM KBOT_APP_MEMBER member_row
     WHERE member_row.APP_ID = 'assistant'
        OR member_row.USER_ID IN (
            SELECT user_account.USER_ID
              FROM KBOT_PLATFORM_USER user_account
             WHERE user_account.OWNER_APP_ID = 'assistant'
        );

    DELETE FROM KBOT_PLATFORM_USER_ROLE platform_role
     WHERE platform_role.USER_ID IN (
         SELECT user_account.USER_ID
           FROM KBOT_PLATFORM_USER user_account
          WHERE user_account.OWNER_APP_ID = 'assistant'
     );

    DELETE FROM KBOT_PLATFORM_USER_CREDENTIAL credential
     WHERE credential.USER_ID IN (
         SELECT user_account.USER_ID
           FROM KBOT_PLATFORM_USER user_account
          WHERE user_account.OWNER_APP_ID = 'assistant'
     );

    DELETE FROM KBOT_APP_ROLE_PERMISSION
     WHERE APP_ID = 'assistant';
    DELETE FROM KBOT_PERMISSION
     WHERE APP_ID = 'assistant';
    DELETE FROM KBOT_APP_ROLE
     WHERE APP_ID = 'assistant';
    DELETE FROM KBOT_APP_DOMAIN
     WHERE APP_ID = 'assistant';
    DELETE FROM KBOT_PLATFORM_USER
     WHERE OWNER_APP_ID = 'assistant';
    DELETE FROM KBOT_PLATFORM_APP
     WHERE APP_ID = 'assistant';

    SELECT SUM(item_count) INTO l_remaining_count
      FROM (
          SELECT COUNT(*) item_count
            FROM KBOT_PLATFORM_APP WHERE APP_ID = 'assistant'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_PLATFORM_USER
           WHERE OWNER_APP_ID = 'assistant'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_PERMISSION WHERE APP_ID = 'assistant'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_ROLE WHERE APP_ID = 'assistant'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_DOMAIN WHERE APP_ID = 'assistant'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_MEMBER WHERE APP_ID = 'assistant'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_MEMBER_ROLE WHERE APP_ID = 'assistant'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_MEMBER_ROLE_SCOPE
           WHERE APP_ID = 'assistant'
          UNION ALL
          SELECT COUNT(*) FROM KBOT_APP_API_CLIENT WHERE APP_ID = 'assistant'
      );

    IF l_remaining_count <> 0 THEN
        RAISE_APPLICATION_ERROR(-20001, 'assistant 授权数据物理删除不完整');
    END IF;

    IF l_target_count > 0 OR l_owned_user_count > 0 OR l_app_member_count > 0 THEN
        INSERT INTO KBOT_ACCESS_AUDIT_EVENT (
            EVENT_ID, ACTOR_ID, APP_ID, ACTION, RESOURCE_TYPE,
            RESOURCE_ID, SUMMARY_JSON, AFFECTED_MEMBER_COUNT, CREATED_AT
        ) VALUES (
            RAWTOHEX(SYS_GUID()), 'migration:retired-assistant-purge',
            'assistant', 'RETIRED_APP_AUTHORIZATION_PURGED',
            'PLATFORM_APP', 'assistant',
            '{"physical_delete":true,"owned_user_count":' ||
            l_owned_user_count || '}',
            l_app_member_count, SYSTIMESTAMP
        );
    END IF;

    COMMIT;
    DBMS_OUTPUT.PUT_LINE(
        'assistant 退役数据已物理删除：App 自有用户=' ||
        l_owned_user_count || '，App 成员=' || l_app_member_count
    );
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
/
