-- 从当前 Schema 的 KBOT_AI_MODEL 生成可复制到其他环境执行的 INSERT。
-- 在 SQL Developer 中使用 Run Script（F5）执行，并从 DBMS Output 复制结果。
-- 输出包含模型凭据和 OCI 私钥，只能在受控环境中使用，不得提交生成结果。
-- 本脚本只读，不修改当前 Schema；执行失败时无需数据库回滚。

SET SERVEROUTPUT ON SIZE UNLIMITED
SET LINESIZE 32767
SET LONG 1000000
SET LONGCHUNKSIZE 32767
SET DEFINE OFF
SET FEEDBACK OFF

DECLARE
    insert_sql VARCHAR2(32767);
    params_sql VARCHAR2(32767);

    FUNCTION quote_text(input_value VARCHAR2)
        RETURN VARCHAR2
    IS
    BEGIN
        IF input_value IS NULL THEN
            RETURN 'NULL';
        END IF;

        RETURN ''''
            || REPLACE(input_value, '''', '''''')
            || '''';
    END quote_text;

    FUNCTION quote_clob(input_value CLOB)
        RETURN VARCHAR2
    IS
        text_value VARCHAR2(30000);
    BEGIN
        IF input_value IS NULL THEN
            RETURN 'NULL';
        END IF;

        IF DBMS_LOB.GETLENGTH(input_value) > 30000 THEN
            RAISE_APPLICATION_ERROR(
                -20001,
                'API_KEY或MODEL_PARAMS超过30000字符'
            );
        END IF;

        text_value := DBMS_LOB.SUBSTR(input_value, 30000, 1);

        -- 防止 CLOB 中的真实换行破坏生成的 JSON 或 SQL。
        text_value := REPLACE(text_value, CHR(13), '\r');
        text_value := REPLACE(text_value, CHR(10), '\n');

        RETURN quote_text(text_value);
    END quote_clob;

BEGIN
    DBMS_OUTPUT.PUT_LINE('SET DEFINE OFF');
    DBMS_OUTPUT.PUT_LINE(
        'WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK'
    );
    DBMS_OUTPUT.PUT_LINE(
        'ALTER SESSION SET TIME_ZONE = ''+00:00'';'
    );
    DBMS_OUTPUT.PUT_LINE('');

    FOR model_row IN (
        SELECT
            RAWTOHEX(MODEL_ID) AS MODEL_ID_HEX,
            SERVED_MODEL_NAME,
            DISPLAY_NAME,
            PROVIDER_MODEL_NAME,
            CATEGORY,
            PROVIDER,
            API_ENDPOINT,
            API_KEY,
            STATUS,
            JSON_SERIALIZE(
                MODEL_PARAMS RETURNING CLOB
            ) AS MODEL_PARAMS_TEXT,
            DESCS
        FROM KBOT_AI_MODEL
        ORDER BY MODEL_ID
    ) LOOP
        IF model_row.MODEL_PARAMS_TEXT IS NULL THEN
            params_sql := 'NULL';
        ELSE
            params_sql :=
                'JSON('
                || quote_clob(model_row.MODEL_PARAMS_TEXT)
                || ')';
        END IF;

        insert_sql :=
            'INSERT INTO KBOT_AI_MODEL ('
            || 'MODEL_ID, '
            || 'SERVED_MODEL_NAME, '
            || 'DISPLAY_NAME, '
            || 'PROVIDER_MODEL_NAME, '
            || 'CATEGORY, '
            || 'PROVIDER, '
            || 'API_ENDPOINT, '
            || 'API_KEY, '
            || 'STATUS, '
            || 'MODEL_PARAMS, '
            || 'DESCS'
            || ') VALUES ('
            || 'HEXTORAW('
            || quote_text(model_row.MODEL_ID_HEX)
            || '), '
            || quote_text(model_row.SERVED_MODEL_NAME)
            || ', '
            || quote_text(model_row.DISPLAY_NAME)
            || ', '
            || quote_text(model_row.PROVIDER_MODEL_NAME)
            || ', '
            || TO_CHAR(model_row.CATEGORY)
            || ', '
            || quote_text(model_row.PROVIDER)
            || ', '
            || quote_text(model_row.API_ENDPOINT)
            || ', '
            || quote_clob(model_row.API_KEY)
            || ', '
            || TO_CHAR(model_row.STATUS)
            || ', '
            || params_sql
            || ', '
            || quote_text(model_row.DESCS)
            || ');';

        DBMS_OUTPUT.PUT_LINE(insert_sql);
    END LOOP;

    DBMS_OUTPUT.PUT_LINE('');
    DBMS_OUTPUT.PUT_LINE('COMMIT;');
END;
/
