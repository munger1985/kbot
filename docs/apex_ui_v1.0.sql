
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

    IF v_login IS NULL OR v_login = UPPER('kbotui_dev') OR v_user_name IN ('SYS')THEN
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

    IF v_login IS NULL OR v_login = UPPER('kbotui_dev') OR v_user_name IN ('SYS')THEN
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

    IF v_login IS NULL OR v_login = UPPER('kbotui_dev') OR v_user_name IN ('SYS')THEN
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


--==============================================================================
--5.配置VPD策略，注意，需要修改object_schema名称。
--需要把KBOTUI_DEV修改成实际的schema。
--==============================================================================
--下面的语句需要dba用户执行
GRANT EXECUTE ON DBMS_RLS TO KBOTUI_DEV;
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
BEGIN
    DBMS_RLS.ADD_POLICY(
        object_schema => 'KBOTUI_DEV',
        object_name => 'KBOT_MD_AGENT',
        policy_name => 'KBOT_AGENT_POLICY',
        policy_function => 'KBOT_AGENT_POLICY',
        statement_types => 'SELECT'
    );
END;
BEGIN
    DBMS_RLS.ADD_POLICY(
        object_schema => 'KBOTUI_DEV',
        object_name => 'KBOT_MD_CHAT_HISTORY',
        policy_name => 'KBOT_CHAT_HISTORY_POLICY',
        policy_function => 'KBOT_CHAT_HISTORY_POLICY',
        statement_types => 'SELECT'
    );
END;

--==============================================================================
--5.apex ui数据字典，执行完之后，需要更新app_id
--update KBOT_MD_DATA_DIC set app_id = ?;
--==============================================================================
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'ENABLE_FLAG','Enabled','1','zh-cn','启用',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('02-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'ENABLE_FLAG','Disabled','0','zh-cn','禁用',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('02-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'KB_CATEGORY','Kbot','1','zh-cn','知识库',1,'包括：文搜文、文搜图（图片抽取文本、或者转成文本，然后文本向量化检索）','ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('08-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'KB_CATEGORY','Image Search','2','zh-cn','图片检索',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'KB_CATEGORY','Generate Report','3','zh-cn','生成报告',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'FILE_STATUS','Approved','3','zh-cn','已审批',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'FILE_STATUS','Rejected','4','zh-cn','审批失败',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SIMILARITY_FLAG','On','1','zh-cn','开',1,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SIMILARITY_FLAG','Off','0','zh-cn','关',1,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'TOOL_TYPE','Knowledge Base','1','zh-cn','知识库',1,null,'ADMIN',to_date('08-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'TOOL_TYPE','Function Call','2','zh-cn','功能调用',1,null,'ADMIN',to_date('08-7月 -25','DD-MON-RR'),'ADMIN',to_date('08-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'KB_CATEGORY','Translate','4','zh-cn','翻译',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('01-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'KB_CATEGORY','Summary','5','zh-cn','摘要',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('01-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'FILE_STATUS','Uploaded','1','zh-cn','已上传',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('04-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'FILE_STATUS','Parsing','5','zh-cn','解析中',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'FILE_STATUS','ParseFailed','7','zh-cn','解析失败',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'CHUNK_TYPE','Txt','1','zh-cn','文本',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('23-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'CHUNK_TYPE','Img','2','zh-cn','图片',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('01-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'PROMPT_CATEGORY','System Prompt','1','zh-cn','系统提示词',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'PROMPT_CATEGORY','Prompt Template','2','zh-cn','提示词模版',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'PROMPT_CATEGORY','Agent Prompt','3','zh-cn','Agent提示词',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_CATEGORY','LLM','1','zh-cn','大语言模型',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('01-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_CATEGORY','Text Embedding','2','zh-cn','文本嵌入模型',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_CATEGORY','Reranker','4','zh-cn','重排模型',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('08-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_CATEGORY','VLM','5','zh-cn','视觉大模型',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('08-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SYS_PARAM_TYPE','Service URL','1','zh-cn','服务URL',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('28-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SYS_PARAM_TYPE','System Logo','2','zh-cn','系统Logo',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('24-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SYS_PARAM_TYPE','System Name','3','zh-cn','系统名称',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('24-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SYS_PARAM_TYPE','Feedback Text Embedding','4','zh-cn','反馈文本Embedding',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('24-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SYS_PARAM_TYPE','Feedback Similarity Threshold','5','zh-cn','反馈相似度阈值',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('24-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'DB_TYPE','Oracle','1','zh-cn','Oracle',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('22-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'DB_TYPE','ADB','2','zh-cn','ADB',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('01-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'DB_TYPE','Heatwave','3','zh-cn','Heatwave',0,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('12-8月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'AGENT_STATUS','Enabled','1','zh-cn','启用',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('21-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'AGENT_STATUS','Disabled','0','zh-cn','禁用',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('21-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'AGENT_STATUS','Archived','2','zh-cn','归档',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('21-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SEARCH_TYPE','Vector Search','1','zh-cn','向量检索',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('12-8月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SEARCH_TYPE','Full Text Search','2','zh-cn','全文检索',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('12-8月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SEARCH_TYPE','Summary Search','3','zh-cn','摘要检索',0,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('12-8月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SEARCH_TYPE','Graph Search','4','zh-cn','Graph检索',0,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('12-8月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'PROCESS_PRIORITY_TYPE','Low','1','zh-cn','低',1,null,'ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('10-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_CATEGORY','Image Embedding','3','zh-cn','图片嵌入模型',1,null,'ADMIN',to_date('08-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'RERANKER_FLAG','Enabled','1','zh-cn','启用',1,null,'ADMIN',to_date('15-7月 -25','DD-MON-RR'),'ADMIN',to_date('24-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'RERANKER_FLAG','Disabled','0','zh-cn','禁用',1,null,'ADMIN',to_date('15-7月 -25','DD-MON-RR'),'ADMIN',to_date('24-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'OVERWRITE_TYPE','Yes','1','zh-cn','是',1,null,'ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('03-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'OVERWRITE_TYPE','No','0','zh-cn','否',1,null,'ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('03-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'PROCESS_PRIORITY_TYPE','Medium','2','zh-cn','中',1,null,'ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('10-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'PROCESS_PRIORITY_TYPE','High','3','zh-cn','高',1,null,'ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('10-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SUMMARY_TYPE','No','0','zh-cn','否',1,'是否开启Summary','ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('03-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SUMMARY_TYPE','Yes','1','zh-cn','是',1,null,'ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('03-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SECURITY_LEVEL_TYPE','Low','1','zh-cn','低',1,null,'ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('14-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SECURITY_LEVEL_TYPE','Medium','2','zh-cn','中',1,null,'ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('14-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'SECURITY_LEVEL_TYPE','High','3','zh-cn','高',1,null,'ADMIN',to_date('03-7月 -25','DD-MON-RR'),'ADMIN',to_date('14-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'CHUNK_PARSER_TYPE','Page','3','zh-cn','按页分块',1,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-8月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_CATEGORY','OCR','6','zh-cn','OCR',1,null,'ADMIN',to_date('08-7月 -25','DD-MON-RR'),'ADMIN',to_date('08-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'IS_IMG2TXT','Yes','1','zh-cn','是',1,null,'ADMIN',to_date('10-7月 -25','DD-MON-RR'),'ADMIN',to_date('10-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'IS_IMG2TXT','No','0','zh-cn','否',1,null,'ADMIN',to_date('10-7月 -25','DD-MON-RR'),'ADMIN',to_date('10-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'IS_TABLE_HEAD_FILL','Yes','1','zh-cn','是',1,null,'ADMIN',to_date('10-7月 -25','DD-MON-RR'),'ADMIN',to_date('10-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'IS_TABLE_HEAD_FILL','No','0','zh-cn','否',1,null,'ADMIN',to_date('10-7月 -25','DD-MON-RR'),'ADMIN',to_date('10-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_PROVIDER','Local','local','zh-cn','本地模型',1,null,'ADMIN',to_date('17-7月 -25','DD-MON-RR'),'ADMIN',to_date('17-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_PROVIDER','OpenAI','openai','zh-cn','OpenAI',1,null,'ADMIN',to_date('17-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_PROVIDER','Azure','azure','zh-cn','Azure',1,null,'ADMIN',to_date('17-7月 -25','DD-MON-RR'),'ADMIN',to_date('17-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'MODEL_PROVIDER','Cohere','cohere','zh-cn','Cohere',1,null,'ADMIN',to_date('17-7月 -25','DD-MON-RR'),'ADMIN',to_date('17-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'APIKEY_SCOPE','Domain','2','zh-cn','业务域',1,null,'ADMIN',to_date('28-7月 -25','DD-MON-RR'),'ADMIN',to_date('28-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'APIKEY_USAGE_TYPE','Internal','1','zh-cn','内部',1,null,'ADMIN',to_date('28-7月 -25','DD-MON-RR'),'ADMIN',to_date('28-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'APIKEY_SCOPE','System','1','zh-cn','系统',1,null,'ADMIN',to_date('28-7月 -25','DD-MON-RR'),'ADMIN',to_date('28-7月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'APIKEY_USAGE_TYPE','External','2','zh-cn','外部',1,null,'ADMIN',to_date('28-7月 -25','DD-MON-RR'),'ADMIN',to_date('28-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'KB_STATUS','Archived','2','zh-cn','归档',1,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('21-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'KB_STATUS','Enabled','1','zh-cn','启用',1,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('21-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'KB_STATUS','Disabled','0','zh-cn','禁用',1,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('21-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'CHUNK_PARSER_TYPE','Fixed Size','1','zh-cn','固定大小分块',1,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-8月 -25','DD-MON-RR'),1);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'CHUNK_PARSER_TYPE','Paragraph','2','zh-cn','基于文档段落分块',0,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('13-8月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'CHUNK_PARSER_TYPE','Semantic','4','zh-cn','语义分块',0,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('12-8月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'FILE_STATUS','Parsed','6','zh-cn','解析成功',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'FILE_STATUS','Archived','9','zh-cn','归档',1,null,'ADMIN',to_date('01-7月 -25','DD-MON-RR'),'ADMIN',to_date('07-7月 -25','DD-MON-RR'),0);
Insert into  KBOT_MD_DATA_DIC (APP_ID,NAME,DISPLAY_NAME,RETURN_VALUE,LANG_CODE,DISPLAY_TRAN_VALUE,STATUS,DESCS,CREATED_BY,CREATED_TIME,UPDATED_BY,UPDATED_TIME,IS_DEFAULT) values (112,'FILE_STATUS','Pending Approve','2','zh-cn','待审批',1,null,'ADMIN',to_date('07-7月 -25','DD-MON-RR'),'ADMIN',to_date('18-7月 -25','DD-MON-RR'),0);

