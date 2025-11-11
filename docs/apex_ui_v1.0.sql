--==============================================================================
--1.KBOT_MD_UI_UTL
--==============================================================================
create or replace package "KBOT_MD_UI_UTL" 
as

--==============================================================================
-- Record operation logs
--==============================================================================
PROCEDURE ins_action_record(
    iv_func_name      in  varchar2,
    iv_action         in  varchar2
);

--==============================================================================
-- Get data dictionary values（display name）
--==============================================================================
function get_dic_dname (
    iv_dic_type      in  varchar2,
    iv_val           in     varchar2)
return varchar2;

--==============================================================================
-- File size byte conversion（kb、mb、gb、tb）
--==============================================================================
function format_file_size(
    in_bytes IN NUMBER
)
return varchar2;

--==============================================================================
-- Get default value from the dictionary（display，return）
--==============================================================================
function get_dic_default_value(
    iv_dic_type      in  varchar2,
    iv_type          in  varchar2
)
return varchar2;

--==============================================================================
-- Get default value from the dictionary（display，return）
--==============================================================================
function get_dic_sql_string(
    iv_dic_type      in  varchar2
)
return varchar2;

end "KBOT_MD_UI_UTL";
/

create or replace package body "KBOT_MD_UI_UTL" 
as
--==============================================================================
-- Record operation logs
--==============================================================================
PROCEDURE ins_action_record(
    iv_func_name      IN  VARCHAR2,
    iv_action         IN  VARCHAR2
)
IS

BEGIN
    INSERT INTO kbot_md_action_record(
        app_id,
        func_name,
        action,
        created_by,
        created_time,
        updated_by,
        updated_time)
    VALUES(
        V('APP_ID'),
        iv_func_name,
        iv_action,
        V('APP_USER'),
        current_date,
        V('APP_USER'),
        current_date  
    );    
   COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        NULL;
END ins_action_record;

--==============================================================================
--Get data dictionary values
--==============================================================================
FUNCTION get_dic_dname (
    iv_dic_type      IN  VARCHAR2,
    iv_val           IN  VARCHAR2)
RETURN VARCHAR2
IS
    lv_lang_code  VARCHAR2(100) := apex_util.get_session_lang;
    lv_dname      VARCHAR2(256); 
BEGIN
    SELECT CASE 
           WHEN lv_lang_code = lang_code
           THEN  display_tran_value
           ELSE  display_name
           END CASE
    INTO   lv_dname
    FROM   kbot_md_data_dic
    WHERE  NAME = iv_dic_type
    AND    return_value = iv_val
    AND    app_id = V('APP_ID')
    AND    status = '1';
     
    RETURN lv_dname;
    
EXCEPTION
    WHEN OTHERS THEN
        RETURN iv_val;
END get_dic_dname;


--==============================================================================
-- File size byte conversion（kb、mb、gb、tb）
--==============================================================================
function format_file_size(
    in_bytes IN NUMBER
) return varchar2
is
    TYPE t_units IS TABLE OF VARCHAR2(10);
    lv_unit_array t_units := t_units('Bytes', 'KB', 'MB', 'GB', 'TB');
    ln_i NUMBER;
    lv_result VARCHAR2(400);
begin
    IF in_bytes = 0 THEN
        RETURN '0 Bytes';
    END IF;
    -- Calculate logarithms to determine unit index
    ln_i := FLOOR(LOG(1024, in_bytes));
    
    -- Ensure that the index does not exceed the range of the unit array
    IF ln_i < 0 THEN
        ln_i := 0;
    ELSIF ln_i > lv_unit_array.COUNT - 1 THEN
        ln_i := lv_unit_array.COUNT - 1;
    END IF;
    
    -- Calculate formatted values
    lv_result := ROUND(in_bytes / POWER(1024, ln_i), 2);
    
    -- Return result string
    RETURN TO_CHAR(lv_result, '9999990.99') || ' ' || lv_unit_array(ln_i + 1);
    
EXCEPTION
    WHEN OTHERS THEN
        RETURN 'Invalid input';
end format_file_size;

--==============================================================================
-- Get default value from the dictionary（display，return）
--==============================================================================
FUNCTION get_dic_default_value(
    iv_dic_type      IN  VARCHAR2,
    iv_type          in  varchar2
)
RETURN VARCHAR2
IS
    lv_value       VARCHAR2(256);
    lv_d_name      VARCHAR2(256);
    lv_lang_code   VARCHAR2(100) := apex_util.get_session_lang;
BEGIN
    SELECT return_value,
        CASE 
           WHEN lv_lang_code = lang_code THEN  display_tran_value
           ELSE  display_name
        END CASE
    INTO   lv_value,
           lv_d_name
    FROM   kbot_md_data_dic 
    WHERE  NAME = iv_dic_type
    AND    status = '1'
    AND    is_default = '1'
    AND app_id = V('APP_ID');
    
    if upper(iv_type) = 'D' THEN
        RETURN lv_d_name;
    ELSE
        RETURN lv_value;
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END get_dic_default_value;

--==============================================================================
-- Get default value from the dictionary（display，return）
--==============================================================================
FUNCTION get_dic_sql_string(
    iv_dic_type      IN  VARCHAR2
)
RETURN VARCHAR2
IS
    lv_sql  VARCHAR2(32766);
BEGIN
    lv_sql := 'SELECT  
               CASE 
                    WHEN apex_util.get_session_lang = lang_code THEN  display_tran_value
                    ELSE  display_name
                    END CASE AS D,
                    return_value AS r
               FROM kbot_md_data_dic 
               WHERE name = '''||iv_dic_type||'''
               AND   status = ''1''
               AND   app_id = V(''APP_ID'')
               ORDER BY return_value
               ';
    
RETURN lv_sql;

END;


end "KBOT_MD_UI_UTL";
/


--==============================================================================
--2.KBOT_DOMAIN_POLICY
--==============================================================================
create or replace function "KBOT_DOMAIN_POLICY" (   
    schema_var IN VARCHAR2,
    table_var IN VARCHAR2
)
RETURN VARCHAR2
AS
    lv_available_view_agent VARCHAR2(4000);
    l_super                 VARCHAR2(100);
    l_agents                VARCHAR2(4000);
    v_role                  VARCHAR2(100);
    predicate               VARCHAR2(4000);
    v_user_name             VARCHAR2(50) := SYS_CONTEXT('USERENV', 'SESSION_USER');
    v_login                 VARCHAR2(50);
BEGIN
    v_login := V('APP_USER'); 

    IF v_login IS NULL OR v_login = UPPER('KBOTUI_DEV') OR v_user_name IN ('SYS')THEN
        predicate := '1 = 1';
    ELSE 
        SELECT JSON_VALUE(ar.KBOT_SUB, '$."KBOT".domain')
              ,ar.KBOT
              ,ar.SUPER
        INTO   lv_available_view_agent
              ,v_role
              ,l_super
        FROM AI_PLATFORM_USERS au
        JOIN AI_PLATFORM_PRIVILEGES_MU ar 
        ON   au.id = ar.USER_ID
        WHERE UPPER(au.USER_NAME) = UPPER(v_login);

        IF l_super = 'Y' OR v_role = 'KB_ADMIN' THEN
            predicate := '1 = 1';
        -- ELSIF v_role = 'KB_VIEWER' THEN
        --     predicate := '1 = 2';
        ELSE
            predicate := 'DOMAIN_ID IN ( 
                ' || lv_available_view_agent || '
            )';
            
        END IF;
    END IF;

    RETURN predicate;
EXCEPTION
    WHEN OTHERS THEN
        RETURN '1 =2';
END;
/
--==============================================================================
--3.KBOT_AGENT_POLICY
--==============================================================================
create or replace function "KBOT_AGENT_POLICY" (   
    schema_var IN VARCHAR2,
    table_var IN VARCHAR2
)
RETURN VARCHAR2
AS
    lv_available_view_agent VARCHAR2(4000);
    l_super                 VARCHAR2(100);
    l_agents                VARCHAR2(4000);
    v_role                  VARCHAR2(100);
    predicate               VARCHAR2(4000);
    v_user_name             VARCHAR2(50) := SYS_CONTEXT('USERENV', 'SESSION_USER');
    v_login                 VARCHAR2(50);
BEGIN
    v_login := V('APP_USER'); 

    IF v_login IS NULL OR v_login = UPPER('KBOTUI_DEV') OR v_user_name IN ('SYS')THEN
        predicate := '1 = 1';
    ELSE 
        SELECT JSON_VALUE(ar.KBOT_SUB, '$."KBOT".agent')
              ,ar.KBOT
              ,ar.SUPER
        INTO   lv_available_view_agent
              ,v_role
              ,l_super
        FROM AI_PLATFORM_USERS au
        JOIN AI_PLATFORM_PRIVILEGES_MU ar 
        ON   au.id = ar.USER_ID
        WHERE UPPER(au.USER_NAME) = UPPER(v_login);

        IF l_super = 'Y' OR v_role IN ('KB_ADMIN','KB_MANAGEMENT')  THEN
            predicate := '1 = 1';  
        ELSE
            predicate := 'AGENT_ID IN ( 
                ' || lv_available_view_agent || '
            )';

        END IF;
    END IF;

    RETURN predicate;
EXCEPTION
    WHEN OTHERS THEN
        RETURN '1 =2';
END;
/

--==============================================================================
--4.KBOT_CHAT_HISTORY_POLICY
--==============================================================================
create or replace function "KBOT_CHAT_HISTORY_POLICY" (   
    schema_var IN VARCHAR2,
    table_var IN VARCHAR2
)
RETURN VARCHAR2
AS
    lv_available_view_agent VARCHAR2(4000);
    l_super                 VARCHAR2(100);
    l_agents                VARCHAR2(4000);
    v_role                  VARCHAR2(100);
    predicate               VARCHAR2(4000);
    v_user_name             VARCHAR2(50) := SYS_CONTEXT('USERENV', 'SESSION_USER');
    v_login                 VARCHAR2(50);
BEGIN
    v_login := V('APP_USER'); 

    IF v_login IS NULL OR v_login = UPPER('KBOTUI_DEV') OR v_user_name IN ('SYS')THEN
        predicate := '1 = 1';
    ELSE 
        SELECT ar.KBOT
              ,ar.SUPER
        INTO   v_role
              ,l_super
        FROM AI_PLATFORM_USERS au
        JOIN AI_PLATFORM_PRIVILEGES_MU ar 
        ON   au.id = ar.USER_ID
        WHERE UPPER(au.USER_NAME) = UPPER(v_login);

        IF l_super = 'Y' OR v_role = 'KB_ADMIN'  THEN
            predicate := '1 = 1';  
        ELSE
            predicate := 'UPPER(CREATED_BY) = '''|| v_login ||'''';
        END IF;
    END IF;

    RETURN predicate;
EXCEPTION
    WHEN OTHERS THEN
        RETURN '1 =2';
END;
/

--==============================================================================
--5.配置VPD策略，注意，需要修改object_schema名称。
--需要把KBOTUI_DEV修改成实际的schema。
--==============================================================================
--下面的语句需要dba用户执行
--GRANT EXECUTE ON DBMS_RLS TO KBOTUI_DEV;
--下面的脚步用实际的schema执行即可
BEGIN
    DBMS_RLS.ADD_POLICY(
        object_schema => 'KBOTUI_DEV',
        object_name => 'KBOT_MD_DOMAIN',
        policy_name => 'KBOT_DOMAIN_POLICY',
        policy_function => 'KBOT_DOMAIN_POLICY',
        statement_types => 'SELECT'
    );
END;
/
BEGIN
    DBMS_RLS.ADD_POLICY(
        object_schema => 'KBOTUI_DEV',
        object_name => 'KBOT_MD_AGENT',
        policy_name => 'KBOT_AGENT_POLICY',
        policy_function => 'KBOT_AGENT_POLICY',
        statement_types => 'SELECT'
    );
END;
/
BEGIN
    DBMS_RLS.ADD_POLICY(
        object_schema => 'KBOTUI_DEV',
        object_name => 'KBOT_MD_CHAT_HISTORY',
        policy_name => 'KBOT_CHAT_HISTORY_POLICY',
        policy_function => 'KBOT_CHAT_HISTORY_POLICY',
        statement_types => 'SELECT'
    );
END;
/

--==============================================================================
--6.apex ui数据字典，执行完之后，需要更新app_id
--update KBOT_MD_DATA_DIC set app_id = ?;
--==============================================================================
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (2,112,'ENABLE_FLAG','Enabled','1','zh-cn','启用',1,null,'ADMIN',to_date('2025-07-01 16:49:05','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-02 19:16:17','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (3,112,'ENABLE_FLAG','Disabled','0','zh-cn','禁用',1,null,'ADMIN',to_date('2025-07-01 16:49:23','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-02 19:16:00','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (4,112,'KB_CATEGORY','Kbot','1','zh-cn','知识库',1,'包括：文搜文、文搜图（图片抽取文本、或者转成文本，然后文本向量化检索）','ADMIN',to_date('2025-07-01 16:53:59','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-08 17:01:31','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (5,112,'KB_CATEGORY','Image Search','2','zh-cn','图片检索',0,null,'ADMIN',to_date('2025-07-01 16:54:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-11-07 10:44:28','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (6,112,'KB_CATEGORY','Generate Report','3','zh-cn','生成报告',0,null,'ADMIN',to_date('2025-07-01 16:55:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-11-07 10:44:31','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (10,112,'FILE_STATUS','Approved','3','zh-cn','已审批',1,null,'ADMIN',to_date('2025-07-01 16:59:10','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-07 17:41:53','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (11,112,'FILE_STATUS','Rejected','4','zh-cn','审批失败',1,null,'ADMIN',to_date('2025-07-01 16:59:37','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-07 17:41:59','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (89,112,'SIMILARITY_FLAG','On','1','zh-cn','开',1,null,'ADMIN',to_date('2025-07-07 11:20:50','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-07 11:21:57','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (90,112,'SIMILARITY_FLAG','Off','0','zh-cn','关',1,null,'ADMIN',to_date('2025-07-07 11:22:47','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-07 11:26:12','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (93,112,'TOOL_TYPE','Knowledge Base','1','zh-cn','知识库',1,null,'ADMIN',to_date('2025-07-08 12:00:47','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-18 21:33:59','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (94,112,'TOOL_TYPE','Function Call','2','zh-cn','功能调用',0,null,'ADMIN',to_date('2025-07-08 12:01:33','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-11-07 10:45:24','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (221,112,'CHUNK_PARSER_TYPE','ROW','5','zh-cn','按行来拆分',1,null,'ADMIN',to_date('2025-08-27 18:01:34','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-27 18:18:09','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (7,112,'KB_CATEGORY','Translate','4','zh-cn','翻译',0,null,'ADMIN',to_date('2025-07-01 16:56:45','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-11-07 10:44:34','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (8,112,'KB_CATEGORY','Summary','5','zh-cn','摘要',0,null,'ADMIN',to_date('2025-07-01 16:57:12','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-11-07 10:44:37','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (9,112,'FILE_STATUS','Uploaded','1','zh-cn','已上传',1,null,'ADMIN',to_date('2025-07-01 16:58:05','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-04 16:18:44','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (12,112,'FILE_STATUS','Parsing','5','zh-cn','解析中',1,null,'ADMIN',to_date('2025-07-01 17:00:26','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-07 17:42:12','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (14,112,'FILE_STATUS','ParseFailed','7','zh-cn','解析失败',1,null,'ADMIN',to_date('2025-07-01 17:01:44','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-07 17:42:34','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (321,112,'CHUNK_TYPE','table','3','zh-cn','表格',1,null,'ADMIN',to_date('2025-10-16 15:44:21','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-10-16 15:44:52','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (322,112,'CHUNK_TYPE','summary','4','zh-cn','摘要',1,null,'ADMIN',to_date('2025-10-16 15:44:46','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-10-16 15:44:46','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (18,112,'CHUNK_TYPE','Txt','1','zh-cn','文本',1,null,'ADMIN',to_date('2025-07-01 17:12:10','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-23 22:19:31','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (19,112,'CHUNK_TYPE','Img','2','zh-cn','图片',1,null,'ADMIN',to_date('2025-07-01 17:12:28','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-01 17:12:28','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (20,112,'PROMPT_CATEGORY','System Prompt','1','zh-cn','系统提示词',1,null,'ADMIN',to_date('2025-07-01 17:12:57','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-18 21:33:34','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (21,112,'PROMPT_CATEGORY','Prompt Template','2','zh-cn','提示词模版',1,null,'ADMIN',to_date('2025-07-01 17:13:20','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-18 21:33:22','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (22,112,'PROMPT_CATEGORY','Agent Prompt','3','zh-cn','Agent提示词',1,null,'ADMIN',to_date('2025-07-01 17:14:33','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-18 21:33:15','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (23,112,'MODEL_CATEGORY','LLM','1','zh-cn','大语言模型',1,null,'ADMIN',to_date('2025-07-01 17:15:00','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-01 17:15:32','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (24,112,'MODEL_CATEGORY','Text Embedding','2','zh-cn','文本嵌入模型',1,null,'ADMIN',to_date('2025-07-01 17:16:00','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-18 21:32:09','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (25,112,'MODEL_CATEGORY','Reranker','4','zh-cn','重排模型',1,null,'ADMIN',to_date('2025-07-01 17:16:50','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-08 10:26:12','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (26,112,'MODEL_CATEGORY','VLM','5','zh-cn','视觉大模型',1,null,'ADMIN',to_date('2025-07-01 17:17:44','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-08 10:26:15','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (27,112,'SYS_PARAM_TYPE','Service URL','1','zh-cn','服务URL',1,null,'ADMIN',to_date('2025-07-01 17:19:05','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-28 16:30:54','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (28,112,'SYS_PARAM_TYPE','System Logo','2','zh-cn','系统Logo',1,null,'ADMIN',to_date('2025-07-01 17:19:20','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-24 14:49:11','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (29,112,'SYS_PARAM_TYPE','System Name','3','zh-cn','系统名称',1,null,'ADMIN',to_date('2025-07-01 17:19:38','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-24 14:44:28','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (30,112,'SYS_PARAM_TYPE','Feedback Text Embedding','4','zh-cn','反馈文本Embedding',0,null,'ADMIN',to_date('2025-07-01 17:20:04','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-11-07 10:45:35','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (31,112,'SYS_PARAM_TYPE','Feedback Similarity Threshold','5','zh-cn','反馈相似度阈值',0,null,'ADMIN',to_date('2025-07-01 17:20:47','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-11-07 10:45:38','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (33,112,'DB_TYPE','Oracle','1','zh-cn','Oracle',1,null,'ADMIN',to_date('2025-07-01 17:23:37','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-22 17:05:55','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (34,112,'DB_TYPE','ADB','2','zh-cn','ADB',0,null,'ADMIN',to_date('2025-07-01 17:24:15','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-26 10:14:50','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (35,112,'DB_TYPE','Heatwave','3','zh-cn','Heatwave',0,null,'ADMIN',to_date('2025-07-01 17:24:42','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-12 12:22:11','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (37,112,'AGENT_STATUS','Enabled','1','zh-cn','启用',1,null,'ADMIN',to_date('2025-07-01 17:26:04','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-21 17:21:32','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (38,112,'AGENT_STATUS','Disabled','0','zh-cn','禁用',1,null,'ADMIN',to_date('2025-07-01 17:26:21','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-21 17:21:37','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (39,112,'AGENT_STATUS','Archived','2','zh-cn','归档',1,null,'ADMIN',to_date('2025-07-01 17:26:37','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-21 17:21:42','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (40,112,'SEARCH_TYPE','Vector Search','1','zh-cn','向量检索',1,null,'ADMIN',to_date('2025-07-01 17:26:58','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-12 20:22:06','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (41,112,'SEARCH_TYPE','Full Text Search','2','zh-cn','全文检索',1,null,'ADMIN',to_date('2025-07-01 17:27:38','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-12 20:22:02','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (42,112,'SEARCH_TYPE','Summary Search','3','zh-cn','摘要检索',1,null,'ADMIN',to_date('2025-07-01 17:28:00','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-19 16:39:48','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (43,112,'SEARCH_TYPE','Graph Search','4','zh-cn','Graph检索',0,null,'ADMIN',to_date('2025-07-01 17:28:25','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-12 12:24:30','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (63,112,'PROCESS_PRIORITY_TYPE','Low','1','zh-cn','低',1,null,'ADMIN',to_date('2025-07-03 15:53:16','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-10 15:04:57','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (92,112,'MODEL_CATEGORY','Image Embedding','3','zh-cn','图片嵌入模型',0,null,'ADMIN',to_date('2025-07-08 10:26:02','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-11-07 10:43:05','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (101,112,'RERANKER_FLAG','Enabled','1','zh-cn','启用',1,null,'ADMIN',to_date('2025-07-15 15:57:59','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-24 10:27:50','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (102,112,'RERANKER_FLAG','Disabled','0','zh-cn','禁用',1,null,'ADMIN',to_date('2025-07-15 15:58:47','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-24 10:27:54','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (61,112,'OVERWRITE_TYPE','Yes','1','zh-cn','是',1,null,'ADMIN',to_date('2025-07-03 14:55:31','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-03 14:55:31','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (62,112,'OVERWRITE_TYPE','No','0','zh-cn','否',1,null,'ADMIN',to_date('2025-07-03 14:55:49','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-03 15:48:42','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (64,112,'PROCESS_PRIORITY_TYPE','Medium','2','zh-cn','中',1,null,'ADMIN',to_date('2025-07-03 16:10:17','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-10 15:04:49','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (65,112,'PROCESS_PRIORITY_TYPE','High','3','zh-cn','高',1,null,'ADMIN',to_date('2025-07-03 16:10:48','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-10 15:04:53','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (66,112,'SUMMARY_TYPE','No','0','zh-cn','否',1,'是否开启Summary','ADMIN',to_date('2025-07-03 16:13:07','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-03 17:15:50','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (67,112,'SUMMARY_TYPE','Yes','1','zh-cn','是',1,null,'ADMIN',to_date('2025-07-03 16:13:34','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-03 16:13:34','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (68,112,'SECURITY_LEVEL_TYPE','Low','1','zh-cn','低',1,null,'ADMIN',to_date('2025-07-03 16:15:36','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-14 11:29:31','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (69,112,'SECURITY_LEVEL_TYPE','Medium','2','zh-cn','中',1,null,'ADMIN',to_date('2025-07-03 16:16:21','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-14 11:29:35','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (70,112,'SECURITY_LEVEL_TYPE','High','3','zh-cn','高',1,null,'ADMIN',to_date('2025-07-03 16:16:50','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-14 11:29:39','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (87,112,'CHUNK_PARSER_TYPE','Page','3','zh-cn','按页分块',1,null,'ADMIN',to_date('2025-07-07 10:33:11','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-07 17:20:21','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (96,112,'IS_IMG2TXT','Yes','1','zh-cn','是',1,null,'ADMIN',to_date('2025-07-10 16:39:10','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-10 16:39:10','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (97,112,'IS_IMG2TXT','No','0','zh-cn','否',1,null,'ADMIN',to_date('2025-07-10 16:39:34','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-10 16:40:01','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (98,112,'IS_TABLE_HEAD_FILL','Yes','1','zh-cn','是',1,null,'ADMIN',to_date('2025-07-10 16:39:56','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-10 16:39:56','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (99,112,'IS_TABLE_HEAD_FILL','No','0','zh-cn','否',1,null,'ADMIN',to_date('2025-07-10 16:40:34','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-10 16:40:34','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (121,112,'MODEL_PROVIDER','Local','local','zh-cn','本地模型',1,null,'ADMIN',to_date('2025-07-17 14:18:55','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-17 14:18:55','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (122,112,'MODEL_PROVIDER','OpenAI','openai','zh-cn','OpenAI',1,null,'ADMIN',to_date('2025-07-17 14:19:39','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-18 06:52:04','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (123,112,'MODEL_PROVIDER','Azure','azure','zh-cn','Azure',0,null,'ADMIN',to_date('2025-07-17 14:20:30','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-10-31 12:53:14','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (124,112,'MODEL_PROVIDER','Cohere','cohere','zh-cn','Cohere',0,null,'ADMIN',to_date('2025-07-17 14:20:47','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-10-31 12:53:10','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (323,112,'DB_TYPE','ElasticSearch','4','zh-cn','ElasticSearch',1,null,'ADMIN',to_date('2025-10-17 11:26:45','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-10-17 11:26:45','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (81,112,'KB_STATUS','Archived','2','zh-cn','归档',1,null,'ADMIN',to_date('2025-07-07 10:15:04','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-21 17:18:01','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (83,112,'KB_STATUS','Enabled','1','zh-cn','启用',1,null,'ADMIN',to_date('2025-07-07 10:16:42','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-21 17:17:43','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (84,112,'KB_STATUS','Disabled','0','zh-cn','禁用',1,null,'ADMIN',to_date('2025-07-07 10:17:10','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-21 17:17:17','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (85,112,'CHUNK_PARSER_TYPE','Fixed Size','1','zh-cn','固定大小分块',1,null,'ADMIN',to_date('2025-07-07 10:32:13','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-07 17:20:39','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (86,112,'CHUNK_PARSER_TYPE','Doc Structure','2','zh-cn','基于文档结构分块',1,null,'ADMIN',to_date('2025-07-07 10:32:43','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 17:37:29','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (88,112,'CHUNK_PARSER_TYPE','Semantic','4','zh-cn','语义分块',0,null,'ADMIN',to_date('2025-07-07 10:33:33','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-12 12:21:51','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (13,112,'FILE_STATUS','Parsed','6','zh-cn','解析成功',1,null,'ADMIN',to_date('2025-07-01 17:01:22','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-07 17:42:17','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (15,112,'FILE_STATUS','Archived','8','zh-cn','归档',1,null,'ADMIN',to_date('2025-07-01 17:05:40','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-18 16:13:15','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (91,112,'FILE_STATUS','Pending Approve','2','zh-cn','待审批',1,null,'ADMIN',to_date('2025-07-07 17:41:28','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-07-18 20:58:05','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (201,112,'MODEL_PROVIDER','OCIGenAI','oci','zh-cn','OCIGenAI',1,null,'ADMIN',to_date('2025-08-20 17:19:44','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-08-20 21:37:01','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (241,112,'ACCESSOR_TYPE','kbot user','1','zh-cn','kbot用户',1,null,'ADMIN',to_date('2025-09-04 11:00:33','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-04 11:00:33','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (242,112,'ACCESSOR_TYPE','kbot service','2','zh-cn','kbot服务',1,null,'ADMIN',to_date('2025-09-04 11:01:07','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-04 11:01:07','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (261,112,'FILE_CATEGORY','text','1','zh-cn','文本',1,null,'ADMIN',to_date('2025-09-12 15:21:41','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:44:41','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (262,112,'FILE_CATEGORY','image','2','zh-cn','图片',1,null,'ADMIN',to_date('2025-09-12 15:22:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:44:50','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (263,112,'FILE_CATEGORY','audio','3','zh-cn','音频',1,null,'ADMIN',to_date('2025-09-12 15:23:04','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-10-31 12:54:27','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (264,112,'FILE_CATEGORY','video','4','zh-cn','视频',1,null,'ADMIN',to_date('2025-09-12 15:23:28','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-10-31 12:54:31','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (265,112,'FILE_CATEGORY_TEXT','pdf','.pdf','zh-cn','pdf',1,null,'ADMIN',to_date('2025-09-12 15:29:20','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:45:18','YYYY-MM-DD HH24:MI:SS'),1);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (266,112,'FILE_CATEGORY_TEXT','docx','.docx','zh-cn','docx',1,null,'ADMIN',to_date('2025-09-12 15:30:05','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:45:07','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (267,112,'FILE_CATEGORY_TEXT','doc','.doc','zh-cn','doc',1,null,'ADMIN',to_date('2025-09-12 15:46:39','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:46:39','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (268,112,'FILE_CATEGORY_TEXT','txt','.txt','zh-cn','txt',1,null,'ADMIN',to_date('2025-09-12 15:47:09','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:09','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (269,112,'FILE_CATEGORY_TEXT','Markdown','.md','zh-cn','Markdown',1,null,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (270,112,'FILE_CATEGORY_TEXT','pptx','.pptx','zh-cn','pptx',1,null,'ADMIN',to_date('2025-09-12 15:47:58','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:58','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (271,112,'FILE_CATEGORY_TEXT','ppt','.ppt','zh-cn','ppt',1,null,'ADMIN',to_date('2025-09-12 15:48:10','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:48:15','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (272,112,'FILE_CATEGORY_TEXT','xlsx','.xlsx','zh-cn','xlsx',1,null,'ADMIN',to_date('2025-09-12 15:49:12','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:49:12','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (273,112,'FILE_CATEGORY_TEXT','xls','.xls','zh-cn','xls',1,null,'ADMIN',to_date('2025-09-12 15:50:29','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:50:29','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (274,112,'FILE_CATEGORY_TEXT','html','.html','zh-cn','html',1,null,'ADMIN',to_date('2025-09-12 15:51:56','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:51:56','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (275,112,'FILE_CATEGORY_IMAGE','png','.png','zh-cn','png',1,null,'ADMIN',to_date('2025-09-12 15:52:22','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:52:22','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (276,112,'FILE_CATEGORY_IMAGE','jpg','.jpg','zh-cn','jpg',1,null,'ADMIN',to_date('2025-09-12 15:52:51','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:52:51','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (277,112,'FILE_CATEGORY_IMAGE','jpeg','.jpeg','zh-cn','jpge',1,null,'ADMIN',to_date('2025-09-12 15:53:09','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:53:09','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (278,112,'FILE_CATEGORY_IMAGE','bmp','.bmp','zh-cn','bmp',1,null,'ADMIN',to_date('2025-09-12 15:53:31','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:53:31','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (279,112,'FILE_CATEGORY_IMAGE','gif','.gif','zh-cn','gif',1,null,'ADMIN',to_date('2025-09-12 15:53:43','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:53:43','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (280,112,'FILE_CATEGORY_AUDIO','mp3','.mp3','zh-cn','mp3',1,null,'ADMIN',to_date('2025-09-12 15:56:11','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:56:11','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (281,112,'FILE_CATEGORY_AUDIO','wav','.wav','zh-cn','wav',1,null,'ADMIN',to_date('2025-09-12 15:56:24','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:56:24','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (282,112,'FILE_CATEGORY_AUDIO','aac','.aac','zh-cn','aac',1,null,'ADMIN',to_date('2025-09-12 15:56:37','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:56:37','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (283,112,'FILE_CATEGORY_AUDIO','flac','.flac','zh-cn','flac',1,null,'ADMIN',to_date('2025-09-12 15:56:55','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:56:55','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (284,112,'FILE_CATEGORY_AUDIO','opus','.opus','zh-cn','opus',1,null,'ADMIN',to_date('2025-09-12 15:57:09','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:57:09','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (285,112,'FILE_CATEGORY_VIDEO','mp4','.mp4','zh-cn','mp4',1,null,'ADMIN',to_date('2025-09-12 15:57:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:57:32','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (286,112,'FILE_CATEGORY_VIDEO','avi','.avi','zh-cn','avi',1,null,'ADMIN',to_date('2025-09-12 15:57:55','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:57:55','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (287,112,'FILE_CATEGORY_VIDEO','mkv','.mkv','zh-cn','mkv',1,null,'ADMIN',to_date('2025-09-12 15:58:09','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:58:09','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (288,112,'FILE_CATEGORY_VIDEO','mov','.mov','zh-cn','mov',1,null,'ADMIN',to_date('2025-09-12 15:58:24','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:58:24','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (289,112,'FILE_CATEGORY_VIDEO','webm','.webm','zh-cn','webm',1,null,'ADMIN',to_date('2025-09-12 15:59:35','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:59:35','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (290,112,'FILE_CATEGORY_VIDEO','flv','.flv','zh-cn','flv',1,null,'ADMIN',to_date('2025-09-12 15:59:49','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:59:49','YYYY-MM-DD HH24:MI:SS'),0);
Insert into KBOT_MD_DATA_DIC (DIC_ID,APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (301,112,'SYS_PARAM_TYPE','Office Online Preview','6','zh-cn','office在线预览',1,null,'ADMIN',to_date('2025-09-22 14:21:18','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-11-07 10:47:41','YYYY-MM-DD HH24:MI:SS'),0);

--==============================================================================
--7.apex ui KBOT_MD_PROMPT
--update KBOT_MD_PROMPT set app_id = ?;
--==============================================================================
Insert into KBOT_MD_PROMPT (APP_ID,DOMAIN_ID,NAME,PROMPT_CATEGORY,TEMPLATE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,PROMPT_UNIQUE_NAME) values (112,null,'Common_CN',2,'你是一个乐于助人、尊重他人、诚实的AI助手。 请参考下面的上下文内容，回答后面的问题。如果您不知道答案，就回答说不知道，不要试图编造答案，也不要漏掉任何相關内容。 上下文： {context} 回答的问题：{question}',1,null,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'TEMPLATE/Common_CN');
Insert into KBOT_MD_PROMPT (APP_ID,DOMAIN_ID,NAME,PROMPT_CATEGORY,TEMPLATE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,PROMPT_UNIQUE_NAME) values (112,161,'image2text',1,'Analyze the image carefully and identify its key visual elements and important details. Detect and read any prominent text within the image, determine the main language of that text, and use that same language to provide a clear, accurate, and detailed description of the image, focusing on its essential content',1,null,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'SYSTEM/image2text');
Insert into KBOT_MD_PROMPT (APP_ID,DOMAIN_ID,NAME,PROMPT_CATEGORY,TEMPLATE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,PROMPT_UNIQUE_NAME) values (112,161,'summary',1,'请对以下文本进行总结，提炼出核心内容和关键信息。要求摘要简洁、准确、连贯。待总结文本：\n{chunk}\n',1,null,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'SYSTEM/summary');

--==============================================================================
--8.apex ui KBOT_MD_API_SECURITY
--update KBOT_MD_API_SECURITY set app_id = ?;
--==============================================================================
Insert into KBOT_MD_API_SECURITY (SECURITY_ID,APP_ID,ACCESSOR,ACCESSOR_TYPE,HASHED_SECRET,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (61,112,'KBot_Embedding_Service',2,'$2b$12$Sp6d3hJt.7T5ZHZHSIL/i.e0i1VLDT4S8okTSTQFaTgTeIPUXCB8O',1,null,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_MD_API_SECURITY (SECURITY_ID,APP_ID,ACCESSOR,ACCESSOR_TYPE,HASHED_SECRET,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (62,112,'KBot_VLM_Service',2,'$2b$12$0TdCNxsf2o8W6oWGD5eV3.8aRvpd/HIY.ad/I8iAFTdG7O8izoaXG',1,null,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_MD_API_SECURITY (SECURITY_ID,APP_ID,ACCESSOR,ACCESSOR_TYPE,HASHED_SECRET,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (43,112,'KBot_Reranker_Service',2,'$2b$12$LdCT/F5qHRqrVy8Hqg8/ZOhv6jOmwIAzzrkXkDyY3nZnW83kvG.ru',1,'3123123123213','ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_MD_API_SECURITY (SECURITY_ID,APP_ID,ACCESSOR,ACCESSOR_TYPE,HASHED_SECRET,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (23,112,'KBot_UI',1,'$2b$12$jNTp9SXqV4jtScnmOvCwse.s1J9OCVhAYKNfIn1YBT4AYC7n.EZei',1,'kbot ui默认用，请勿修改密码','ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_MD_API_SECURITY (SECURITY_ID,APP_ID,ACCESSOR,ACCESSOR_TYPE,HASHED_SECRET,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (44,112,'KBot_LLM_Service',2,'$2b$12$6xbxa7qS1ydaWfCjuDehf.RlolpkvvGYvO.RxICZlSEq9gHKN1XUy',1,'123','ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));


--==============================================================================
--9.apex ui KBOT_SYS_PARSER_CONF
--update KBOT_SYS_PARSER_CONF set app_id = ?;
--==============================================================================
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (1,112,1,'.docx',1,'{"split_strategy":1,"chunk_size":1000,"chunk_overlap":100}',0,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (2,112,1,'.docx',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (3,112,2,'.bmp',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (4,112,3,'.flac',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (5,112,1,'.doc',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (6,112,1,'.doc',1,'{"split_strategy":1,"chunk_size":1000,"chunk_overlap":100}',0,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (7,112,1,'.pdf',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (8,112,1,'.pdf',1,'{"split_strategy":1,"chunk_size":1000,"chunk_overlap":100}',0,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (9,112,1,'.txt',1,'{"split_strategy":1,"chunk_size":1000,"chunk_overlap":100}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (10,112,1,'.docx',2,'{"split_strategy":2}',0,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (11,112,1,'.doc',2,'{"split_strategy":2}',0,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (12,112,1,'.pdf',2,'{"split_strategy":2}',0,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (13,112,1,'.md',2,'{"split_strategy":2}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (14,112,1,'.pptx',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (15,112,1,'.ppt',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (16,112,1,'.xlsx',5,'{"split_strategy":5}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (17,112,1,'.xls',5,'{"split_strategy":5}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (18,112,1,'.html',1,'{"split_strategy":1,"chunk_size":1000,"chunk_overlap":100}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (19,112,2,'.jpg',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (20,112,2,'.png',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (22,112,2,'.jpeg',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (23,112,2,'.gif',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (24,112,3,'.aac',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (25,112,3,'.mp3',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (26,112,3,'.opus',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (27,112,3,'.wav',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (28,112,4,'.avi',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (29,112,4,'.flv',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (30,112,4,'.mkv',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (31,112,4,'.mov',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (32,112,4,'.mp4',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));
Insert into KBOT_SYS_PARSER_CONF (CONF_ID,APP_ID,FILE_CATEGORY,FILE_EXT,CHUNK_PARSER,CHUNK_PARSER_PARAM,IS_DEFAULT,STATUS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (33,112,4,'.webm',3,'{"split_strategy":3}',1,1,'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));


--==============================================================================
--8.apex ui Realse note
--update KBOT_MD_RELEASE_NOTE set app_id = ?;
--==============================================================================
Insert into KBOT_MD_RELEASE_NOTE (RELEASE_ID,APP_ID,RELEASE_TITLE,RELEASE_CONTENT,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME) values (21,112,'Kbot_v3.1_20251017','1.	支持知识库级别，按照不同文本类型配置默认chunk策略。同时，某一个文件粒度也可以修改chunk策略。
2.	后台REST API调用增加了安全认证功能
3.	增加Markdown格式解析，按照Markdown文件结构进行解析。
4.	支持Excel/csv格式解析，把excel/csv格式数据转成Json进行embedding。也可以支持里面的图片处理
5.	增加Pdf/docx/pptx/excel/txt/jpg/bmp/wav/mp3/mp4等格式数据独立的预览功能。
6.	增加了对文本chunk生成摘要，同时增加了摘要检索引擎。
7.	增加了Chunk编辑/Disable/Enable功能
8.	支持对图片格式数据构建RAG，把图片转成文本=》向量化，文搜图功能。
9.	支持Qwen Embedding和Reranker模型。
10.	增加了模型联通性校验接口。
11. 支持Elastic Search作为向量存储。
12.	一些bug修复以及后台程序优化。','ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'),'ADMIN',to_date('2025-09-12 15:47:32','YYYY-MM-DD HH24:MI:SS'));

