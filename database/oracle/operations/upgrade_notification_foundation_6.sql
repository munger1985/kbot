-- 通知基础修复：补齐管理权限和审批队列空指派；不修改业务记录或重置用户。
MERGE INTO KBOT_PERMISSION target
USING (SELECT 'platform:notification_manage' code FROM DUAL) source
ON (target.PERMISSION_CODE = source.code)
WHEN NOT MATCHED THEN INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
VALUES (source.code, 'platform', '管理通知隔离事件与重试');

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (SELECT 'platform' app_id, 'platform_admin' role_code,
              'platform:notification_manage' permission_code FROM DUAL) source
ON (target.APP_ID = source.app_id AND target.ROLE_CODE = source.role_code
    AND target.PERMISSION_CODE = source.permission_code)
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
VALUES (source.app_id, source.role_code, source.permission_code);

-- 审批队列不指派给申请人；非变更人工输入仍由应用层要求确切处理人。
BEGIN
    EXECUTE IMMEDIATE 'ALTER TABLE KBOT_OPS_HITL MODIFY (ASSIGNEE_USER_ID NULL)';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -1451 THEN RAISE; END IF;
END;
/
-- 运行终态由应用层合同校验；取消和过期不能伪装为执行失败。
BEGIN
    EXECUTE IMMEDIATE 'ALTER TABLE KBOT_BACKGROUND_OPERATION DROP CONSTRAINT CK_NOTIFY_OPER_STATUS';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -2443 THEN RAISE; END IF;
END;
/

COMMIT;
