-- 一次性回退 bootstrap_km_initial_admin.sql 创建的固定 KM 资源。
-- 在 SQL Developer 中使用 Run Script（F5）执行；不使用输入变量或外部脚本。
-- 本脚本按固定名称定位 km_portal Domain 及其唯一的 assets Collection。
-- 如果 Domain 已产生来源、Asset、任务、Agent、KC 入库数据或其他业务引用，
-- 脚本会整体回滚并拒绝删除。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    l_domain_id NUMBER(38);
    l_collection_id RAW(16);
    l_count PLS_INTEGER;
BEGIN
    BEGIN
        SELECT DOMAIN_ID
          INTO l_domain_id
          FROM KBOT_PLATFORM_DOMAIN
         WHERE NAME = 'km_portal';
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            raise_application_error(
                -20101,
                '未找到 km_portal Domain，无需回退。'
            );
    END;

    SELECT COUNT(*)
      INTO l_count
      FROM KBOT_KC_COLLECTION
     WHERE DOMAIN_ID = l_domain_id
       AND DISPLAY_NAME = 'assets';

    IF l_count > 1 THEN
        raise_application_error(
            -20103,
            'km_portal 中存在多个 assets Collection，拒绝删除。'
        );
    ELSIF l_count = 1 THEN
        SELECT COLLECTION_ID
          INTO l_collection_id
          FROM KBOT_KC_COLLECTION
         WHERE DOMAIN_ID = l_domain_id
           AND DISPLAY_NAME = 'assets';
    ELSE
        l_collection_id := NULL;
        dbms_output.put_line('未找到 assets Collection，将继续清理其余初始化资源。');
    END IF;

    SELECT
        (SELECT COUNT(*) FROM KBOT_KM_SOURCE WHERE DOMAIN_ID = l_domain_id)
        + (SELECT COUNT(*) FROM KBOT_KM_ASSET WHERE DOMAIN_ID = l_domain_id)
        + (SELECT COUNT(*) FROM KBOT_KM_JOB WHERE DOMAIN_ID = l_domain_id)
        + (SELECT COUNT(*) FROM KBOT_KM_AGENT WHERE DOMAIN_ID = l_domain_id)
        + (SELECT COUNT(*) FROM KBOT_KM_AGENT_GRANT WHERE DOMAIN_ID = l_domain_id)
      INTO l_count
      FROM DUAL;

    IF l_count > 0 THEN
        raise_application_error(
            -20104,
            'km_portal 已产生 KM 来源、Asset、任务或 Agent 数据，拒绝回退。'
        );
    END IF;

    IF l_collection_id IS NOT NULL THEN
        SELECT
            (SELECT COUNT(*) FROM KBOT_KC_COLLECTION_BINDING
              WHERE COLLECTION_ID = l_collection_id)
            + (SELECT COUNT(*) FROM KBOT_KC_INGESTION_RECEIPT
                WHERE COLLECTION_ID = l_collection_id)
            + (SELECT COUNT(*) FROM KBOT_KC_BUNDLE
                WHERE COLLECTION_ID = l_collection_id)
            + (SELECT COUNT(*) FROM KBOT_KC_INGESTION_JOB
                WHERE COLLECTION_ID = l_collection_id)
          INTO l_count
          FROM DUAL;

        IF l_count > 0 THEN
            raise_application_error(
                -20105,
                'assets Collection 已产生绑定或入库数据，拒绝回退。'
            );
        END IF;
    END IF;

    SELECT COUNT(*)
      INTO l_count
      FROM KBOT_APP_MEMBER_ROLE
     WHERE USER_ID = 'kmadmin'
       AND APP_ID <> 'km_asset';

    IF l_count > 0 THEN
        raise_application_error(
            -20106,
            'kmadmin 已获得非 KM App 授权，拒绝删除该用户。'
        );
    END IF;

    IF l_collection_id IS NOT NULL THEN
        DELETE FROM KBOT_KC_COLLECTION
         WHERE COLLECTION_ID = l_collection_id;
        dbms_output.put_line('已删除 km_portal/assets Collection。');
    END IF;

    DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE
     WHERE APP_ID = 'km_asset'
       AND USER_ID = 'kmadmin';

    DELETE FROM KBOT_APP_MEMBER_ROLE
     WHERE APP_ID = 'km_asset'
       AND USER_ID = 'kmadmin';
    dbms_output.put_line('已删除 kmadmin 的全部 KM App 授权：' || SQL%ROWCOUNT || ' 行。');

    DELETE FROM KBOT_APP_MEMBER
     WHERE APP_ID = 'km_asset'
       AND USER_ID = 'kmadmin';

    DELETE FROM KBOT_APP_DOMAIN
     WHERE APP_ID = 'km_asset'
       AND DOMAIN_ID = l_domain_id;

    DELETE FROM KBOT_PLATFORM_USER_CREDENTIAL
     WHERE USER_ID = 'kmadmin';
    dbms_output.put_line('已删除 kmadmin 登录凭据：' || SQL%ROWCOUNT || ' 行。');

    DELETE FROM KBOT_PLATFORM_USER
     WHERE USER_ID = 'kmadmin';
    dbms_output.put_line('已删除 kmadmin 用户：' || SQL%ROWCOUNT || ' 行。');

    DELETE FROM KBOT_PLATFORM_DOMAIN
     WHERE DOMAIN_ID = l_domain_id
       AND NAME = 'km_portal';
    IF SQL%ROWCOUNT <> 1 THEN
        raise_application_error(
            -20108,
            'km_portal Domain 删除数量异常，回退已取消。'
        );
    END IF;

    COMMIT;
    dbms_output.put_line('KM 初始化资源回退完成。');
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        dbms_output.put_line('KM 初始化资源回退失败，所有删除均已回滚：' || SQLERRM);
        RAISE;
END;
/
