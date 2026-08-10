-- KBot 4.0 AIOps 与知识检索 App 默认角色、权限及映射。
-- 只写 KBOT_PERMISSION、KBOT_APP_ROLE、KBOT_APP_ROLE_PERMISSION，
-- 不创建用户，不分配成员角色。

SET SERVEROUTPUT ON SIZE UNLIMITED
SET DEFINE OFF
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

PROMPT [1/3] 补齐权限目录

MERGE INTO KBOT_PERMISSION target
USING (
    SELECT 'knowledge_retrieval:use' code, 'knowledge_retrieval' app_id,
           '使用知识检索' display_name FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:upload', 'knowledge_retrieval',
           '上传知识文件' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:review', 'knowledge_retrieval',
           '审核知识文件' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:member_manage', 'knowledge_retrieval',
           '管理应用成员' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:knowledge_manage', 'knowledge_retrieval',
           '管理知识库' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:data_manage', 'knowledge_retrieval',
           '管理问数资源' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:agent_manage', 'knowledge_retrieval',
           '管理知识检索 Agent' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:operations_manage', 'knowledge_retrieval',
           '管理知识检索运行' FROM DUAL
    UNION ALL SELECT 'aiops:use', 'aiops', '使用 AIOps' FROM DUAL
    UNION ALL SELECT 'aiops:domain_manage', 'aiops',
           '管理 AIOps Domain 配置' FROM DUAL
    UNION ALL SELECT 'aiops:member_manage', 'aiops',
           '管理 AIOps 成员' FROM DUAL
    UNION ALL SELECT 'aiops:operations_manage', 'aiops',
           '管理 AIOps 运行' FROM DUAL
    UNION ALL SELECT 'aiops:target_manage', 'aiops', '管理诊断目标' FROM DUAL
    UNION ALL SELECT 'aiops:monitor_source_manage', 'aiops',
           '管理监控源' FROM DUAL
    UNION ALL SELECT 'aiops:policy_manage', 'aiops', '管理诊断策略' FROM DUAL
    UNION ALL SELECT 'aiops:plan_manage', 'aiops', '管理变更计划' FROM DUAL
    UNION ALL SELECT 'aiops:agent_manage', 'aiops',
           '管理 AIOps Agent' FROM DUAL
    UNION ALL SELECT 'aiops:proposal:approve', 'aiops',
           '审批执行提案' FROM DUAL
) source
ON (target.PERMISSION_CODE = source.code)
WHEN MATCHED THEN UPDATE SET
    target.APP_ID = source.app_id,
    target.DISPLAY_NAME = source.display_name
WHEN NOT MATCHED THEN INSERT (
    PERMISSION_CODE, APP_ID, DISPLAY_NAME
) VALUES (
    source.code, source.app_id, source.display_name
);

PROMPT [2/3] 补齐默认角色

MERGE INTO KBOT_APP_ROLE target
USING (
    SELECT 'knowledge_retrieval' app_id, 'user' role_code,
           '用户' display_name, 'ACTIVE' status FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval', 'contributor',
           '贡献者', 'ACTIVE' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval', 'reviewer',
           '审核人', 'ACTIVE' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval', 'manager',
           '管理员', 'ACTIVE' FROM DUAL
    UNION ALL SELECT 'aiops', 'operator',
           '运维操作员', 'ACTIVE' FROM DUAL
    UNION ALL SELECT 'aiops', 'approver',
           '审批人', 'ACTIVE' FROM DUAL
    UNION ALL SELECT 'aiops', 'manager',
           '管理员', 'ACTIVE' FROM DUAL
) source
ON (
    target.APP_ID = source.app_id
    AND target.ROLE_CODE = source.role_code
)
WHEN MATCHED THEN UPDATE SET
    target.DISPLAY_NAME = source.display_name,
    target.STATUS = source.status
WHEN NOT MATCHED THEN INSERT (
    APP_ID, ROLE_CODE, DISPLAY_NAME, STATUS
) VALUES (
    source.app_id, source.role_code, source.display_name, source.status
);

PROMPT [3/3] 补齐角色权限映射

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT role.APP_ID, role.ROLE_CODE, permission.PERMISSION_CODE
      FROM KBOT_APP_ROLE role
      JOIN KBOT_PERMISSION permission
        ON permission.APP_ID = role.APP_ID
     WHERE role.APP_ID = 'knowledge_retrieval'
       AND (
           role.ROLE_CODE = 'manager'
           OR (role.ROLE_CODE = 'user'
               AND permission.PERMISSION_CODE = 'knowledge_retrieval:use')
           OR (role.ROLE_CODE = 'contributor'
               AND permission.PERMISSION_CODE IN (
                   'knowledge_retrieval:use',
                   'knowledge_retrieval:upload'
               ))
           OR (role.ROLE_CODE = 'reviewer'
               AND permission.PERMISSION_CODE IN (
                   'knowledge_retrieval:use',
                   'knowledge_retrieval:upload',
                   'knowledge_retrieval:review'
               ))
       )
    UNION ALL
    SELECT role.APP_ID, role.ROLE_CODE, permission.PERMISSION_CODE
      FROM KBOT_APP_ROLE role
      JOIN KBOT_PERMISSION permission
        ON permission.APP_ID = role.APP_ID
     WHERE role.APP_ID = 'aiops'
       AND (
           role.ROLE_CODE = 'manager'
           OR (role.ROLE_CODE = 'operator'
               AND permission.PERMISSION_CODE IN (
                   'aiops:use',
                   'aiops:operations_manage',
                   'aiops:target_manage',
                   'aiops:monitor_source_manage',
                   'aiops:policy_manage',
                   'aiops:plan_manage'
               ))
           OR (role.ROLE_CODE = 'approver'
               AND permission.PERMISSION_CODE IN (
                   'aiops:use',
                   'aiops:operations_manage',
                   'aiops:proposal:approve'
               ))
       )
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.ROLE_CODE = source.ROLE_CODE
    AND target.PERMISSION_CODE = source.PERMISSION_CODE
)
WHEN NOT MATCHED THEN INSERT (
    APP_ID, ROLE_CODE, PERMISSION_CODE
) VALUES (
    source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE
);

COMMIT;

DECLARE
    l_permission_count PLS_INTEGER;
    l_role_count PLS_INTEGER;
    l_mapping_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*) INTO l_permission_count
      FROM KBOT_PERMISSION
     WHERE APP_ID IN ('knowledge_retrieval', 'aiops');
    SELECT COUNT(*) INTO l_role_count
      FROM KBOT_APP_ROLE
     WHERE APP_ID IN ('knowledge_retrieval', 'aiops');
    SELECT COUNT(*) INTO l_mapping_count
      FROM KBOT_APP_ROLE_PERMISSION
     WHERE APP_ID IN ('knowledge_retrieval', 'aiops');

    dbms_output.put_line(
        '默认权限/角色补齐完成：permissions=' || l_permission_count
        || ', roles=' || l_role_count
        || ', mappings=' || l_mapping_count
    );
END;
/

PROMPT 未创建用户，未写入 KBOT_APP_MEMBER_ROLE。
