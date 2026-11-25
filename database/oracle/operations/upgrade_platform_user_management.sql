-- 将普通 App 自建账号收归平台管理；用户 ID、凭据、飞书绑定和角色范围不变。
-- 停止旧版用户管理写入并备份后，由部署人员在 KBot 业务 Schema 执行；不含敏感值。
-- 任一步失败即回滚；可重复执行。
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    unsafe_count NUMBER;
BEGIN
    -- 来源变为平台后不能让既有的越界绑定获得原本没有的访问权限。
    SELECT COUNT(*) INTO unsafe_count
    FROM KBOT_PLATFORM_USER u
    WHERE u.ACCOUNT_ORIGIN = 'APP'
      AND u.IS_PROTECTED = 'N'
      AND (
          EXISTS (SELECT 1 FROM KBOT_PLATFORM_USER_ROLE r WHERE r.USER_ID = u.USER_ID)
          OR EXISTS (
              SELECT 1 FROM KBOT_APP_MEMBER m
              WHERE m.USER_ID = u.USER_ID AND (m.APP_ID <> u.OWNER_APP_ID OR m.IS_INITIAL_ADMIN = 'Y')
          )
      );
    IF unsafe_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20091, '普通 App 账号存在范围异常的授权，请先人工核对；没有迁移任何账号。');
    END IF;
END;
/

UPDATE KBOT_APP_MEMBER m
SET MEMBER_SOURCE = 'PLATFORM_GRANT',
    -- 旧版停用普通账号时同步停用其所属 App；新版仅由账号状态控制暂停。
    STATUS = CASE WHEN m.STATUS = 'DISABLED' AND EXISTS (
        SELECT 1 FROM KBOT_PLATFORM_USER u
        WHERE u.USER_ID = m.USER_ID AND u.STATUS = 'DISABLED'
    ) THEN 'ACTIVE' ELSE m.STATUS END,
    UPDATED_AT = SYSTIMESTAMP
WHERE m.MEMBER_SOURCE = 'APP_CREATED'
  AND m.IS_INITIAL_ADMIN = 'N'
  AND EXISTS (
      SELECT 1 FROM KBOT_PLATFORM_USER u
      WHERE u.USER_ID = m.USER_ID
        AND u.ACCOUNT_ORIGIN = 'APP'
        AND u.OWNER_APP_ID = m.APP_ID
        AND u.IS_PROTECTED = 'N'
  );

UPDATE KBOT_PLATFORM_USER
SET ACCOUNT_ORIGIN = 'PLATFORM',
    OWNER_APP_ID = NULL,
    UPDATED_AT = SYSTIMESTAMP
WHERE ACCOUNT_ORIGIN = 'APP'
  AND IS_PROTECTED = 'N';

COMMIT;
