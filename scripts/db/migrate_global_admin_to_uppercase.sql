-- 一次性将旧的全局管理员账号 admin 迁移为 ADMIN。
-- 在 SQL Developer 中使用 Run Script（F5）直接执行；不要求输入参数。
-- 脚本复制父记录和外键子记录后删除旧账号，失败时整体回滚。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    l_lower_count      PLS_INTEGER;
    l_upper_count      PLS_INTEGER;
    l_credential_count PLS_INTEGER;
    l_role_count       PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_lower_count
      FROM KBOT_PLATFORM_USER
     WHERE USER_ID = 'admin';

    SELECT COUNT(*)
      INTO l_upper_count
      FROM KBOT_PLATFORM_USER
     WHERE USER_ID = 'ADMIN';

    IF l_lower_count = 0 AND l_upper_count = 1 THEN
        dbms_output.put_line('ADMIN 已存在，无需迁移。');
        RETURN;
    END IF;

    IF l_lower_count <> 1 THEN
        raise_application_error(
            -20001,
            '未找到唯一的小写 admin 用户，拒绝迁移。'
        );
    END IF;

    IF l_upper_count <> 0 THEN
        raise_application_error(
            -20002,
            '大写 ADMIN 已存在，拒绝覆盖现有用户。'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_credential_count
      FROM KBOT_PLATFORM_USER_CREDENTIAL
     WHERE USER_ID = 'admin';

    IF l_credential_count <> 1 THEN
        raise_application_error(
            -20003,
            '小写 admin 的登录凭据不完整，拒绝迁移。'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_role_count
      FROM KBOT_APP_MEMBER_ROLE
     WHERE USER_ID = 'admin';

    INSERT INTO KBOT_PLATFORM_USER (
        USER_ID, DISPLAY_NAME, MAX_SECURITY_LEVEL,
        STATUS, CREATED_AT, UPDATED_AT
    )
    SELECT
        'ADMIN', DISPLAY_NAME, 3, STATUS, CREATED_AT, SYSTIMESTAMP
      FROM KBOT_PLATFORM_USER
     WHERE USER_ID = 'admin';

    INSERT INTO KBOT_PLATFORM_USER_CREDENTIAL (
        USER_ID, PASSWORD_HASH, MUST_CHANGE_PASSWORD,
        PASSWORD_UPDATED_AT, CREATED_AT, UPDATED_AT
    )
    SELECT
        'ADMIN', PASSWORD_HASH, MUST_CHANGE_PASSWORD,
        PASSWORD_UPDATED_AT, CREATED_AT, SYSTIMESTAMP
      FROM KBOT_PLATFORM_USER_CREDENTIAL
     WHERE USER_ID = 'admin';

    INSERT INTO KBOT_APP_MEMBER_ROLE (
        APP_ID, DOMAIN_ID, USER_ID, ROLE_CODE,
        STATUS, CREATED_BY, CREATED_AT
    )
    SELECT
        APP_ID, DOMAIN_ID, 'ADMIN', ROLE_CODE,
        STATUS, CREATED_BY, CREATED_AT
      FROM KBOT_APP_MEMBER_ROLE
     WHERE USER_ID = 'admin';

    DELETE FROM KBOT_APP_MEMBER_ROLE
     WHERE USER_ID = 'admin';

    DELETE FROM KBOT_PLATFORM_USER_CREDENTIAL
     WHERE USER_ID = 'admin';

    DELETE FROM KBOT_PLATFORM_USER
     WHERE USER_ID = 'admin';

    COMMIT;
    dbms_output.put_line('全局管理员已迁移为 ADMIN。');
    dbms_output.put_line('保留角色授权：' || l_role_count || ' 行。');
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        dbms_output.put_line('迁移失败，事务已回滚：' || SQLERRM);
        RAISE;
END;
/

SELECT USER_ID, DISPLAY_NAME, STATUS
  FROM KBOT_PLATFORM_USER
 WHERE USER_ID IN ('admin', 'ADMIN')
 ORDER BY USER_ID;

SELECT USER_ID, DOMAIN_ID, APP_ID, ROLE_CODE, STATUS
  FROM KBOT_APP_MEMBER_ROLE
 WHERE USER_ID IN ('admin', 'ADMIN')
 ORDER BY USER_ID, DOMAIN_ID, APP_ID, ROLE_CODE;
