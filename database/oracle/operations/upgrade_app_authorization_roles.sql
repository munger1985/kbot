-- 将既有业务 App 无损升级为统一三层系统角色。
-- 使用 SQL Developer 的 Run Script（F5）执行；任何预检或迁移校验失败都会回滚 DML。
-- Oracle DDL 会隐式提交，因此审计表创建与后续角色 DML 是两个明确阶段。

WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;
SET SERVEROUTPUT ON;

DECLARE
    l_count NUMBER;
BEGIN
    SELECT COUNT(*) INTO l_count
      FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_ACCESS_AUDIT_EVENT';
    IF l_count = 0 THEN
        EXECUTE IMMEDIATE q'[
            CREATE TABLE KBOT_ACCESS_AUDIT_EVENT (
                EVENT_ID VARCHAR2(36 CHAR) PRIMARY KEY,
                ACTOR_ID VARCHAR2(256 CHAR) NOT NULL,
                APP_ID VARCHAR2(64 CHAR) NOT NULL,
                ACTION VARCHAR2(64 CHAR) NOT NULL,
                RESOURCE_TYPE VARCHAR2(64 CHAR) NOT NULL,
                RESOURCE_ID VARCHAR2(256 CHAR) NOT NULL,
                SUMMARY_JSON CLOB NOT NULL,
                AFFECTED_MEMBER_COUNT NUMBER(19) DEFAULT 0 NOT NULL,
                CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
            )
        ]';
        EXECUTE IMMEDIATE
            'CREATE INDEX IX_ACCESS_AUDIT_APP_TIME ' ||
            'ON KBOT_ACCESS_AUDIT_EVENT (APP_ID, CREATED_AT)';
        EXECUTE IMMEDIATE
            'CREATE INDEX IX_ACCESS_AUDIT_ACTOR_TIME ' ||
            'ON KBOT_ACCESS_AUDIT_EVENT (ACTOR_ID, CREATED_AT)';
    END IF;
END;
/

DECLARE
    l_count NUMBER;
    l_migrated_bindings NUMBER := 0;
    l_stripped_role_count NUMBER := 0;
    l_affected_custom_members NUMBER := 0;
BEGIN
    SELECT COUNT(*) INTO l_count
      FROM KBOT_APP_ROLE role_row
     WHERE role_row.APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND role_row.ROLE_CODE IN ('app_user', 'app_manager', 'app_admin')
       AND role_row.IS_SYSTEM <> 'Y'
       AND (
           role_row.STATUS <> 'ACTIVE'
           OR role_row.SCOPE_POLICY <> CASE role_row.ROLE_CODE
               WHEN 'app_user' THEN 'SELECTABLE'
               ELSE 'ALL_APP_DOMAINS'
           END
           OR EXISTS (
               SELECT 1
                 FROM KBOT_APP_ROLE_PERMISSION actual_permission
                WHERE actual_permission.APP_ID = role_row.APP_ID
                  AND actual_permission.ROLE_CODE = role_row.ROLE_CODE
                  AND NOT (
                      role_row.ROLE_CODE = 'app_admin'
                      OR (
                          role_row.ROLE_CODE = 'app_manager'
                          AND actual_permission.PERMISSION_CODE NOT IN (
                              role_row.APP_ID || ':member_manage',
                              role_row.APP_ID || ':role_manage'
                          )
                      )
                      OR (
                          role_row.ROLE_CODE = 'app_user'
                          AND actual_permission.PERMISSION_CODE IN (
                              'knowledge_retrieval:use',
                              'knowledge_retrieval:knowledge_chat',
                              'knowledge_retrieval:x_search',
                              'media_studio:use',
                              'media_studio:image_generate',
                              'media_studio:media_read',
                              'media_studio:run_read',
                              'aiops:use',
                              'aiops:work_item_handle'
                          )
                      )
                  )
           )
           OR EXISTS (
               SELECT 1
                 FROM KBOT_PERMISSION expected_permission
                WHERE expected_permission.APP_ID = role_row.APP_ID
                  AND (
                      role_row.ROLE_CODE = 'app_admin'
                      OR (
                          role_row.ROLE_CODE = 'app_manager'
                          AND expected_permission.PERMISSION_CODE NOT IN (
                              role_row.APP_ID || ':member_manage',
                              role_row.APP_ID || ':role_manage'
                          )
                      )
                      OR (
                          role_row.ROLE_CODE = 'app_user'
                          AND expected_permission.PERMISSION_CODE IN (
                              'knowledge_retrieval:use',
                              'knowledge_retrieval:knowledge_chat',
                              'knowledge_retrieval:x_search',
                              'media_studio:use',
                              'media_studio:image_generate',
                              'media_studio:media_read',
                              'media_studio:run_read',
                              'aiops:use',
                              'aiops:work_item_handle'
                          )
                      )
                  )
                  AND NOT EXISTS (
                      SELECT 1
                        FROM KBOT_APP_ROLE_PERMISSION actual_permission
                       WHERE actual_permission.APP_ID = role_row.APP_ID
                         AND actual_permission.ROLE_CODE = role_row.ROLE_CODE
                         AND actual_permission.PERMISSION_CODE =
                             expected_permission.PERMISSION_CODE
                  )
           )
       );
    IF l_count > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20001,
            '同名自定义角色与统一系统角色的权限或 Scope 不一致，请先人工审计冲突'
        );
    END IF;

    SELECT COUNT(*) INTO l_count
      FROM KBOT_APP_ROLE role_row
     WHERE role_row.APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND role_row.ROLE_CODE NOT IN (
               'user', 'app_user', 'app_manager', 'app_admin'
           )
       AND EXISTS (
           SELECT 1
             FROM KBOT_APP_ROLE_PERMISSION reserved_permission
            WHERE reserved_permission.APP_ID = role_row.APP_ID
              AND reserved_permission.ROLE_CODE = role_row.ROLE_CODE
              AND reserved_permission.PERMISSION_CODE IN (
                  role_row.APP_ID || ':member_manage',
                  role_row.APP_ID || ':role_manage'
              )
       )
       AND NOT EXISTS (
           SELECT 1
             FROM KBOT_APP_ROLE_PERMISSION use_permission
            WHERE use_permission.APP_ID = role_row.APP_ID
              AND use_permission.ROLE_CODE = role_row.ROLE_CODE
              AND use_permission.PERMISSION_CODE = role_row.APP_ID || ':use'
       );
    IF l_count > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20002,
            '自定义角色清理保留管理权限后将缺少 App use 权限，请先人工审计冲突'
        );
    END IF;

    SELECT COUNT(*) INTO l_count
      FROM KBOT_APP_MEMBER_ROLE binding
     WHERE NOT EXISTS (
               SELECT 1
                 FROM KBOT_APP_ROLE role_row
                WHERE role_row.APP_ID = binding.APP_ID
                  AND role_row.ROLE_CODE = binding.ROLE_CODE
           );
    IF l_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20003, '检测到孤儿成员角色绑定');
    END IF;

    SELECT COUNT(*) INTO l_count
      FROM KBOT_APP_MEMBER_ROLE_SCOPE scope_row
     WHERE NOT EXISTS (
               SELECT 1
                 FROM KBOT_APP_MEMBER_ROLE binding
                WHERE binding.APP_ID = scope_row.APP_ID
                  AND binding.USER_ID = scope_row.USER_ID
                  AND binding.ROLE_CODE = scope_row.ROLE_CODE
           )
        OR NOT EXISTS (
               SELECT 1
                 FROM KBOT_APP_DOMAIN app_domain
                WHERE app_domain.APP_ID = scope_row.APP_ID
                  AND app_domain.DOMAIN_ID = scope_row.DOMAIN_ID
           );
    IF l_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20004, '检测到孤儿 Domain 范围');
    END IF;

    MERGE INTO KBOT_PERMISSION target
    USING (
        SELECT 'media_studio:member_manage' PERMISSION_CODE,
               'media_studio' APP_ID, '管理多媒体创作工作台成员' DISPLAY_NAME
          FROM DUAL
        UNION ALL
        SELECT 'media_studio:role_manage', 'media_studio',
               '管理多媒体创作工作台角色' FROM DUAL
    ) source
       ON (target.PERMISSION_CODE = source.PERMISSION_CODE)
    WHEN MATCHED THEN UPDATE SET
        target.APP_ID = source.APP_ID,
        target.DISPLAY_NAME = source.DISPLAY_NAME
    WHEN NOT MATCHED THEN INSERT (
        PERMISSION_CODE, APP_ID, DISPLAY_NAME
    ) VALUES (
        source.PERMISSION_CODE, source.APP_ID, source.DISPLAY_NAME
    );

    UPDATE KBOT_PERMISSION
       SET DISPLAY_NAME = '管理巡检计划与报告模板'
     WHERE PERMISSION_CODE = 'aiops:plan_manage';

    MERGE INTO KBOT_APP_ROLE target
    USING (
        SELECT 'knowledge_retrieval' APP_ID, 'app_user' ROLE_CODE,
               'App 用户' DISPLAY_NAME, 'SELECTABLE' SCOPE_POLICY FROM DUAL
        UNION ALL SELECT 'knowledge_retrieval', 'app_manager',
               'App 业务管理员', 'ALL_APP_DOMAINS' FROM DUAL
        UNION ALL SELECT 'knowledge_retrieval', 'app_admin',
               'App 管理员', 'ALL_APP_DOMAINS' FROM DUAL
        UNION ALL SELECT 'media_studio', 'app_user',
               'App 用户', 'SELECTABLE' FROM DUAL
        UNION ALL SELECT 'media_studio', 'app_manager',
               'App 业务管理员', 'ALL_APP_DOMAINS' FROM DUAL
        UNION ALL SELECT 'media_studio', 'app_admin',
               'App 管理员', 'ALL_APP_DOMAINS' FROM DUAL
        UNION ALL SELECT 'aiops', 'app_user',
               'App 用户', 'SELECTABLE' FROM DUAL
        UNION ALL SELECT 'aiops', 'app_manager',
               'App 业务管理员', 'ALL_APP_DOMAINS' FROM DUAL
        UNION ALL SELECT 'aiops', 'app_admin',
               'App 管理员', 'ALL_APP_DOMAINS' FROM DUAL
    ) source
       ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE)
    WHEN MATCHED THEN UPDATE SET
        target.DISPLAY_NAME = source.DISPLAY_NAME,
        target.IS_SYSTEM = 'Y',
        target.SCOPE_POLICY = source.SCOPE_POLICY,
        target.STATUS = 'ACTIVE',
        target.ROW_VERSION = target.ROW_VERSION + 1
        WHERE target.DISPLAY_NAME <> source.DISPLAY_NAME
           OR target.IS_SYSTEM <> 'Y'
           OR target.SCOPE_POLICY <> source.SCOPE_POLICY
           OR target.STATUS <> 'ACTIVE'
    WHEN NOT MATCHED THEN INSERT (
        APP_ID, ROLE_CODE, DISPLAY_NAME, IS_SYSTEM,
        SCOPE_POLICY, STATUS, ROW_VERSION
    ) VALUES (
        source.APP_ID, source.ROLE_CODE, source.DISPLAY_NAME, 'Y',
        source.SCOPE_POLICY, 'ACTIVE', 1
    );

    SELECT COUNT(*) INTO l_stripped_role_count
      FROM KBOT_APP_ROLE role_row
     WHERE role_row.APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND role_row.ROLE_CODE NOT IN (
               'user', 'app_user', 'app_manager', 'app_admin'
           )
       AND EXISTS (
           SELECT 1
             FROM KBOT_APP_ROLE_PERMISSION reserved_permission
            WHERE reserved_permission.APP_ID = role_row.APP_ID
              AND reserved_permission.ROLE_CODE = role_row.ROLE_CODE
              AND reserved_permission.PERMISSION_CODE IN (
                  role_row.APP_ID || ':member_manage',
                  role_row.APP_ID || ':role_manage'
              )
       );

    SELECT COUNT(*) INTO l_affected_custom_members
      FROM (
          SELECT DISTINCT binding.APP_ID, binding.USER_ID
            FROM KBOT_APP_MEMBER_ROLE binding
           WHERE binding.STATUS = 'ACTIVE'
             AND binding.APP_ID IN (
                     'knowledge_retrieval',
                     'media_studio', 'aiops'
                 )
             AND binding.ROLE_CODE NOT IN (
                     'user', 'app_user', 'app_manager', 'app_admin'
                 )
             AND EXISTS (
                 SELECT 1
                   FROM KBOT_APP_ROLE_PERMISSION reserved_permission
                  WHERE reserved_permission.APP_ID = binding.APP_ID
                    AND reserved_permission.ROLE_CODE = binding.ROLE_CODE
                    AND reserved_permission.PERMISSION_CODE IN (
                        binding.APP_ID || ':member_manage',
                        binding.APP_ID || ':role_manage'
                    )
             )
      );

    MERGE INTO KBOT_APP_MEMBER_ROLE target
    USING (
        SELECT binding.APP_ID, binding.USER_ID,
               MIN(binding.CREATED_BY) CREATED_BY
          FROM KBOT_APP_MEMBER_ROLE binding
         WHERE binding.STATUS = 'ACTIVE'
           AND binding.APP_ID IN (
                   'knowledge_retrieval',
                   'media_studio', 'aiops'
               )
           AND binding.ROLE_CODE NOT IN (
                   'user', 'app_user', 'app_manager', 'app_admin'
               )
           AND EXISTS (
               SELECT 1
                 FROM KBOT_APP_ROLE_PERMISSION reserved_permission
                WHERE reserved_permission.APP_ID = binding.APP_ID
                  AND reserved_permission.ROLE_CODE = binding.ROLE_CODE
                  AND reserved_permission.PERMISSION_CODE IN (
                      binding.APP_ID || ':member_manage',
                      binding.APP_ID || ':role_manage'
                  )
           )
         GROUP BY binding.APP_ID, binding.USER_ID
    ) source
       ON (target.APP_ID = source.APP_ID
           AND target.USER_ID = source.USER_ID
           AND target.ROLE_CODE = 'app_admin')
    WHEN MATCHED THEN UPDATE SET
        target.SCOPE_MODE = 'ALL_APP_DOMAINS',
        target.STATUS = 'ACTIVE'
        WHERE target.SCOPE_MODE <> 'ALL_APP_DOMAINS'
           OR target.STATUS <> 'ACTIVE'
    WHEN NOT MATCHED THEN INSERT (
        APP_ID, USER_ID, ROLE_CODE, SCOPE_MODE,
        STATUS, CREATED_BY, CREATED_AT
    ) VALUES (
        source.APP_ID, source.USER_ID, 'app_admin', 'ALL_APP_DOMAINS',
        'ACTIVE', source.CREATED_BY, SYSTIMESTAMP
    );

    SELECT COUNT(*) INTO l_count
      FROM KBOT_APP_MEMBER_ROLE binding
     WHERE binding.STATUS = 'ACTIVE'
       AND binding.APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND binding.ROLE_CODE NOT IN (
               'user', 'app_user', 'app_manager', 'app_admin'
           )
       AND EXISTS (
           SELECT 1
             FROM KBOT_APP_ROLE_PERMISSION reserved_permission
            WHERE reserved_permission.APP_ID = binding.APP_ID
              AND reserved_permission.ROLE_CODE = binding.ROLE_CODE
              AND reserved_permission.PERMISSION_CODE IN (
                  binding.APP_ID || ':member_manage',
                  binding.APP_ID || ':role_manage'
              )
       )
       AND NOT EXISTS (
           SELECT 1
             FROM KBOT_APP_MEMBER_ROLE admin_binding
            WHERE admin_binding.APP_ID = binding.APP_ID
              AND admin_binding.USER_ID = binding.USER_ID
              AND admin_binding.ROLE_CODE = 'app_admin'
              AND admin_binding.SCOPE_MODE = 'ALL_APP_DOMAINS'
              AND admin_binding.STATUS = 'ACTIVE'
       );
    IF l_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20008, '旧管理员成员迁移至 app_admin 失败');
    END IF;

    INSERT INTO KBOT_ACCESS_AUDIT_EVENT (
        EVENT_ID, ACTOR_ID, APP_ID, ACTION, RESOURCE_TYPE,
        RESOURCE_ID, SUMMARY_JSON, AFFECTED_MEMBER_COUNT, CREATED_AT
    )
    SELECT RAWTOHEX(SYS_GUID()),
           'migration:app-authorization-v2', role_row.APP_ID,
           'CUSTOM_ROLE_ADMIN_MEMBERS_MIGRATED',
           'APP_ROLE', role_row.ROLE_CODE,
           '{"member_migration":"app_admin","scope_mode":"ALL_APP_DOMAINS",' ||
           '"reserved_permissions_removed":true}',
           (
               SELECT COUNT(DISTINCT binding.USER_ID)
                 FROM KBOT_APP_MEMBER_ROLE binding
                WHERE binding.APP_ID = role_row.APP_ID
                  AND binding.ROLE_CODE = role_row.ROLE_CODE
                  AND binding.STATUS = 'ACTIVE'
           ),
           SYSTIMESTAMP
      FROM KBOT_APP_ROLE role_row
     WHERE role_row.APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND role_row.ROLE_CODE NOT IN (
               'user', 'app_user', 'app_manager', 'app_admin'
           )
       AND EXISTS (
           SELECT 1
             FROM KBOT_APP_ROLE_PERMISSION reserved_permission
            WHERE reserved_permission.APP_ID = role_row.APP_ID
              AND reserved_permission.ROLE_CODE = role_row.ROLE_CODE
              AND reserved_permission.PERMISSION_CODE IN (
                  role_row.APP_ID || ':member_manage',
                  role_row.APP_ID || ':role_manage'
              )
       );

    DELETE FROM KBOT_APP_ROLE_PERMISSION role_permission
     WHERE role_permission.APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND role_permission.ROLE_CODE NOT IN (
               'user', 'app_user', 'app_manager', 'app_admin'
           )
       AND role_permission.PERMISSION_CODE IN (
               role_permission.APP_ID || ':member_manage',
               role_permission.APP_ID || ':role_manage'
           );

    MERGE INTO KBOT_APP_MEMBER_ROLE_SCOPE target
    USING (
        SELECT binding.APP_ID, binding.USER_ID, binding.ROLE_CODE,
               app_domain.DOMAIN_ID
          FROM KBOT_APP_MEMBER_ROLE binding
          JOIN KBOT_APP_DOMAIN app_domain
            ON app_domain.APP_ID = binding.APP_ID
           AND app_domain.STATUS = 'ACTIVE'
         WHERE binding.ROLE_CODE = 'app_user'
           AND binding.SCOPE_MODE = 'ALL_APP_DOMAINS'
    ) source
       ON (target.APP_ID = source.APP_ID
           AND target.USER_ID = source.USER_ID
           AND target.ROLE_CODE = source.ROLE_CODE
           AND target.DOMAIN_ID = source.DOMAIN_ID)
    WHEN NOT MATCHED THEN INSERT (
        APP_ID, USER_ID, ROLE_CODE, DOMAIN_ID
    ) VALUES (
        source.APP_ID, source.USER_ID, source.ROLE_CODE, source.DOMAIN_ID
    );

    MERGE INTO KBOT_APP_MEMBER_ROLE target
    USING (
        SELECT old_binding.APP_ID, old_binding.USER_ID,
               old_binding.STATUS, old_binding.CREATED_BY
          FROM KBOT_APP_MEMBER_ROLE old_binding
         WHERE old_binding.ROLE_CODE = 'user'
           AND old_binding.APP_ID IN (
                   'knowledge_retrieval', 'media_studio', 'aiops'
               )
    ) source
       ON (target.APP_ID = source.APP_ID
           AND target.USER_ID = source.USER_ID
           AND target.ROLE_CODE = 'app_user')
    WHEN MATCHED THEN UPDATE SET
        target.SCOPE_MODE = 'SELECTED_DOMAINS',
        target.STATUS = CASE
            WHEN target.STATUS = 'ACTIVE' OR source.STATUS = 'ACTIVE'
            THEN 'ACTIVE' ELSE 'DISABLED' END
    WHEN NOT MATCHED THEN INSERT (
        APP_ID, USER_ID, ROLE_CODE, SCOPE_MODE,
        STATUS, CREATED_BY, CREATED_AT
    ) VALUES (
        source.APP_ID, source.USER_ID, 'app_user', 'SELECTED_DOMAINS',
        source.STATUS, source.CREATED_BY, SYSTIMESTAMP
    );
    l_migrated_bindings := SQL%ROWCOUNT;

    MERGE INTO KBOT_APP_MEMBER_ROLE_SCOPE target
    USING (
        SELECT binding.APP_ID, binding.USER_ID, 'app_user' ROLE_CODE,
               app_domain.DOMAIN_ID
          FROM KBOT_APP_MEMBER_ROLE binding
          JOIN KBOT_APP_DOMAIN app_domain
            ON app_domain.APP_ID = binding.APP_ID
           AND app_domain.STATUS = 'ACTIVE'
         WHERE binding.ROLE_CODE = 'app_user'
           AND binding.SCOPE_MODE = 'ALL_APP_DOMAINS'
        UNION
        SELECT old_binding.APP_ID, old_binding.USER_ID, 'app_user',
               app_domain.DOMAIN_ID
          FROM KBOT_APP_MEMBER_ROLE old_binding
          JOIN KBOT_APP_DOMAIN app_domain
            ON app_domain.APP_ID = old_binding.APP_ID
           AND app_domain.STATUS = 'ACTIVE'
         WHERE old_binding.ROLE_CODE = 'user'
           AND old_binding.SCOPE_MODE = 'ALL_APP_DOMAINS'
        UNION
        SELECT old_scope.APP_ID, old_scope.USER_ID, 'app_user',
               old_scope.DOMAIN_ID
          FROM KBOT_APP_MEMBER_ROLE_SCOPE old_scope
         WHERE old_scope.ROLE_CODE = 'user'
    ) source
       ON (target.APP_ID = source.APP_ID
           AND target.USER_ID = source.USER_ID
           AND target.ROLE_CODE = source.ROLE_CODE
           AND target.DOMAIN_ID = source.DOMAIN_ID)
    WHEN NOT MATCHED THEN INSERT (
        APP_ID, USER_ID, ROLE_CODE, DOMAIN_ID
    ) VALUES (
        source.APP_ID, source.USER_ID, source.ROLE_CODE, source.DOMAIN_ID
    );

    UPDATE KBOT_APP_MEMBER_ROLE
       SET SCOPE_MODE = 'SELECTED_DOMAINS'
     WHERE ROLE_CODE = 'app_user'
       AND APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           );

    DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE
     WHERE ROLE_CODE = 'user'
       AND APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           );
    DELETE FROM KBOT_APP_MEMBER_ROLE
     WHERE ROLE_CODE = 'user'
       AND APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           );
    DELETE FROM KBOT_APP_ROLE_PERMISSION
     WHERE ROLE_CODE = 'user'
       AND APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           );
    DELETE FROM KBOT_APP_ROLE
     WHERE ROLE_CODE = 'user'
       AND APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           );

    UPDATE KBOT_APP_ROLE
       SET IS_SYSTEM = 'N', ROW_VERSION = ROW_VERSION + 1
     WHERE APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND ROLE_CODE NOT IN ('app_user', 'app_manager', 'app_admin')
       AND IS_SYSTEM <> 'N';

    DELETE FROM KBOT_APP_ROLE_PERMISSION
     WHERE APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND ROLE_CODE IN ('app_user', 'app_manager', 'app_admin');

    INSERT INTO KBOT_APP_ROLE_PERMISSION (APP_ID, ROLE_CODE, PERMISSION_CODE)
    SELECT permission.APP_ID, 'app_admin', permission.PERMISSION_CODE
      FROM KBOT_PERMISSION permission
     WHERE permission.APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           );

    INSERT INTO KBOT_APP_ROLE_PERMISSION (APP_ID, ROLE_CODE, PERMISSION_CODE)
    SELECT permission.APP_ID, 'app_manager', permission.PERMISSION_CODE
      FROM KBOT_PERMISSION permission
     WHERE permission.APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND permission.PERMISSION_CODE NOT IN (
               permission.APP_ID || ':member_manage',
               permission.APP_ID || ':role_manage'
           );

    INSERT INTO KBOT_APP_ROLE_PERMISSION (APP_ID, ROLE_CODE, PERMISSION_CODE)
    SELECT permission.APP_ID, 'app_user', permission.PERMISSION_CODE
      FROM KBOT_PERMISSION permission
     WHERE permission.PERMISSION_CODE IN (
               'knowledge_retrieval:use',
               'knowledge_retrieval:knowledge_chat',
               'knowledge_retrieval:x_search',
               'media_studio:use',
               'media_studio:image_generate',
               'media_studio:media_read',
               'media_studio:run_read',
               'aiops:use',
               'aiops:work_item_handle'
           );

    UPDATE KBOT_APP_MEMBER_ROLE
       SET SCOPE_MODE = 'ALL_APP_DOMAINS'
     WHERE APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND ROLE_CODE IN ('app_manager', 'app_admin');
    DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE
     WHERE APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND ROLE_CODE IN ('app_manager', 'app_admin');

    SELECT COUNT(*) INTO l_count
      FROM KBOT_APP_ROLE
     WHERE APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND IS_SYSTEM = 'Y'
       AND STATUS = 'ACTIVE';
    IF l_count <> 12 THEN
        RAISE_APPLICATION_ERROR(-20005, '统一系统角色数量校验失败');
    END IF;

    SELECT COUNT(*) INTO l_count
      FROM KBOT_APP_ROLE role_row
     WHERE role_row.APP_ID IN (
               'knowledge_retrieval', 'media_studio', 'aiops'
           )
       AND role_row.IS_SYSTEM = 'Y'
       AND role_row.ROLE_CODE NOT IN ('app_user', 'app_manager', 'app_admin');
    IF l_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20006, '仍存在非规范系统角色');
    END IF;

    SELECT COUNT(*) INTO l_count
      FROM KBOT_APP_MEMBER_ROLE binding
     WHERE binding.ROLE_CODE = 'app_user'
       AND binding.SCOPE_MODE <> 'SELECTED_DOMAINS';
    IF l_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20007, 'app_user Domain 范围迁移失败');
    END IF;

    INSERT INTO KBOT_ACCESS_AUDIT_EVENT (
        EVENT_ID, ACTOR_ID, APP_ID, ACTION, RESOURCE_TYPE,
        RESOURCE_ID, SUMMARY_JSON, AFFECTED_MEMBER_COUNT, CREATED_AT
    )
    SELECT
        RAWTOHEX(SYS_GUID()), 'migration:app-authorization-v2', 'platform',
        'APP_ROLE_MODEL_MIGRATED', 'APP_AUTHORIZATION', 'v2',
        '{"system_roles":["app_user","app_manager","app_admin"]}',
        l_migrated_bindings, SYSTIMESTAMP
      FROM DUAL
     WHERE NOT EXISTS (
         SELECT 1 FROM KBOT_ACCESS_AUDIT_EVENT
          WHERE ACTION = 'APP_ROLE_MODEL_MIGRATED'
            AND RESOURCE_TYPE = 'APP_AUTHORIZATION'
            AND RESOURCE_ID = 'v2'
     );

    COMMIT;
    DBMS_OUTPUT.PUT_LINE(
        'App 权限角色升级完成，迁移 user 绑定数=' || l_migrated_bindings ||
        '，清理保留管理权限的自定义角色数=' || l_stripped_role_count ||
        '，迁移至 app_admin 的 ACTIVE 成员数=' ||
        l_affected_custom_members
    );
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
/
