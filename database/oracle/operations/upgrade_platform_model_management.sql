-- 将全局 AI 模型管理移交平台；不调整多媒体绑定权限，不把旧授权继承为平台权限。
MERGE INTO KBOT_PERMISSION target
USING (SELECT 'platform:model_manage' CODE FROM DUAL) source
ON (target.PERMISSION_CODE = source.CODE)
WHEN NOT MATCHED THEN INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
VALUES (source.CODE, 'platform', '管理全局 AI 模型');

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (SELECT 'platform' APP_ID, 'platform_admin' ROLE_CODE,
              'platform:model_manage' PERMISSION_CODE FROM DUAL) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE
    AND target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

DELETE FROM KBOT_APP_ROLE_PERMISSION
WHERE APP_ID = 'media_studio' AND PERMISSION_CODE = 'media_studio:model_manage';

DELETE FROM KBOT_PERMISSION
WHERE APP_ID = 'media_studio' AND PERMISSION_CODE = 'media_studio:model_manage';

COMMIT;
