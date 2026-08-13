-- 为已部署的 KM Asset 来源增加后台自动同步开关。
-- 在 SQL Developer 中使用 Run Script（F5）执行，可重复执行。
-- 所有现有来源默认关闭，升级或项目重启后不会自动创建同步任务。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    l_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_KM_SOURCE'
       AND COLUMN_NAME = 'AUTO_SYNC_ENABLED';

    IF l_count = 0 THEN
        EXECUTE IMMEDIATE q'[
            ALTER TABLE KBOT_KM_SOURCE ADD (
                AUTO_SYNC_ENABLED NUMBER(1) DEFAULT 0 NOT NULL
            )
        ]';
        dbms_output.put_line('已新增 KBOT_KM_SOURCE.AUTO_SYNC_ENABLED，现有来源默认关闭。');
    ELSE
        EXECUTE IMMEDIATE q'[
            UPDATE KBOT_KM_SOURCE
               SET AUTO_SYNC_ENABLED = 0
             WHERE AUTO_SYNC_ENABLED IS NULL
        ]';
        dbms_output.put_line('AUTO_SYNC_ENABLED 已存在，保留现有开关状态。');
    END IF;

    SELECT COUNT(*)
      INTO l_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_KM_SOURCE'
       AND CONSTRAINT_NAME = 'CK_KM_SOURCE_AUTO_SYNC';

    IF l_count = 0 THEN
        EXECUTE IMMEDIATE q'[
            ALTER TABLE KBOT_KM_SOURCE
            ADD CONSTRAINT CK_KM_SOURCE_AUTO_SYNC
            CHECK (AUTO_SYNC_ENABLED IN (0, 1))
        ]';
        dbms_output.put_line('已新增后台自动同步开关约束。');
    END IF;

    COMMIT;
    dbms_output.put_line('KM 来源后台自动同步开关升级完成。');
END;
/
