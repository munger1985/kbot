SET DEFINE OFF
SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    source_date_expression VARCHAR2(32767) := q'~
        COALESCE(
            NULLIF(TRIM(JSON_VALUE(
                NORMALIZED_METADATA_JSON,
                '$.publish_time' RETURNING VARCHAR2(128)
                NULL ON EMPTY NULL ON ERROR
            )), ''),
            NULLIF(TRIM(JSON_VALUE(
                NORMALIZED_METADATA_JSON,
                '$.last_update_time' RETURNING VARCHAR2(128)
                NULL ON EMPTY NULL ON ERROR
            )), ''),
            NULLIF(TRIM(JSON_VALUE(
                NORMALIZED_METADATA_JSON,
                '$.create_time' RETURNING VARCHAR2(128)
                NULL ON EMPTY NULL ON ERROR
            )), ''),
            NULLIF(TRIM(JSON_VALUE(
                NORMALIZED_METADATA_JSON,
                '$.publish_date' RETURNING VARCHAR2(128)
                NULL ON EMPTY NULL ON ERROR
            )), ''),
            NULLIF(TRIM(JSON_VALUE(
                NORMALIZED_METADATA_JSON,
                '$.asset_date' RETURNING VARCHAR2(128)
                NULL ON EMPTY NULL ON ERROR
            )), '')
        )
    ~';
    changed_rows PLS_INTEGER;
    object_count PLS_INTEGER;
BEGIN
    EXECUTE IMMEDIATE
        'UPDATE KBOT_KM_ASSET SET PUBLISH_DATE = '
        || source_date_expression
        || ' WHERE '
        || source_date_expression
        || ' IS NOT NULL AND ('
        || 'PUBLISH_DATE IS NULL OR TRIM(PUBLISH_DATE) <> '
        || source_date_expression
        || ')';
    changed_rows := SQL%ROWCOUNT;

    SELECT COUNT(*) INTO object_count
      FROM USER_INDEXES
     WHERE INDEX_NAME = 'IX_KM_ASSET_DATE';
    IF object_count > 0 THEN
        EXECUTE IMMEDIATE 'DROP INDEX IX_KM_ASSET_DATE';
    END IF;

    SELECT COUNT(*) INTO object_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_KM_ASSET'
       AND COLUMN_NAME = 'ASSET_DATE_VALUE';
    IF object_count > 0 THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_KM_ASSET DROP COLUMN ASSET_DATE_VALUE';
    END IF;

    EXECUTE IMMEDIATE q'~
        ALTER TABLE KBOT_KM_ASSET ADD (
            ASSET_DATE_VALUE GENERATED ALWAYS AS (
                COALESCE(
                    TO_DATE(
                        SUBSTR(TRIM(PUBLISH_DATE), 1, 10)
                        DEFAULT NULL ON CONVERSION ERROR,
                        'FXYYYY-MM-DD'
                    ),
                    TO_DATE(
                        SUBSTR(TRIM(LAST_UPDATE_TIME), 1, 10)
                        DEFAULT NULL ON CONVERSION ERROR,
                        'FXYYYY-MM-DD'
                    )
                )
            ) VIRTUAL
        )
    ~';
    EXECUTE IMMEDIATE
        'CREATE INDEX IX_KM_ASSET_DATE '
        || 'ON KBOT_KM_ASSET (DOMAIN_ID, ASSET_DATE_VALUE)';

    EXECUTE IMMEDIATE 'ALTER VIEW KBOT_V_KM_ASSET_CURRENT COMPILE';
    EXECUTE IMMEDIATE 'ALTER VIEW KBOT_V_KM_ASSET_SEARCHABLE COMPILE';
    COMMIT;
    DBMS_OUTPUT.PUT_LINE(
        'KM Asset 源日期修复完成，回填行数=' || changed_rows
    );
END;
/
