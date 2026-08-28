-- AIOps Schema 14 -> 15 原位升级脚本。
--
-- 保留 Target、Agent、监控源、Webhook 凭据、会话及诊断数据，完成
-- “逻辑 Target + 多 Target Agent + 单 Target 会话”结构升级。
--
-- 执行要求：
-- 1. 仅允许在 KBOT_V_OPS_SCHEMA_VERSION=AIOPS/14/aiops-oracle-v4 时执行一次；
-- 2. 执行前停止 AIOps API、Worker 和 Scheduler，并完成 Schema 备份；
-- 3. 使用 KBot Schema Owner 在 SQL Developer 中以 Run Script（F5）执行；
-- 4. 本脚本不修改监控源、托管凭据、Target Source Binding 的业务数据或密钥。

SET DEFINE OFF
SET SERVEROUTPUT ON SIZE UNLIMITED
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

PROMPT [1/9] 校验当前 Schema 版本与可迁移数据

DECLARE
    v_component        VARCHAR2(32 CHAR);
    v_schema_version   NUMBER;
    v_contract_version VARCHAR2(64 CHAR);
    v_unresolved       NUMBER;
BEGIN
    SELECT COMPONENT, SCHEMA_VERSION, CONTRACT_VERSION
      INTO v_component, v_schema_version, v_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF v_component <> 'AIOPS'
       OR v_schema_version <> 14
       OR v_contract_version <> 'aiops-oracle-v4' THEN
        RAISE_APPLICATION_ERROR(
            -20001,
            '仅允许从 AIOPS/14/aiops-oracle-v4 升级，当前为 '
            || v_component || '/' || v_schema_version || '/'
            || v_contract_version
        );
    END IF;

    -- 当前 Agent 版本必须能够由旧 TARGET_ID、旧 Agent Binding，或唯一的
    -- Source Binding 推导出至少一个逻辑 Target。
    SELECT COUNT(*)
      INTO v_unresolved
      FROM KBOT_OPS_AGENT a
      JOIN KBOT_OPS_AGENT_VERSION av
        ON av.AGENT_VERSION_ID = a.CURRENT_VERSION_ID
     WHERE av.TARGET_ID IS NULL
       AND NOT EXISTS (
            SELECT 1
              FROM KBOT_OPS_TARGET_BINDING tb
             WHERE tb.AGENT_ID = a.AGENT_ID
               AND tb.STATUS = 'ACTIVE'
       )
       AND 1 <> (
            SELECT COUNT(DISTINCT tsb.TARGET_ID)
              FROM KBOT_OPS_AGENT_VERSION_SOURCE avs
              JOIN KBOT_OPS_TARGET_SOURCE_BINDING tsb
                ON tsb.DIAGNOSTIC_SOURCE_ID = avs.DIAGNOSTIC_SOURCE_ID
               AND tsb.STATUS = 'ACTIVE'
             WHERE avs.AGENT_VERSION_ID = av.AGENT_VERSION_ID
       );

    IF v_unresolved > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20002,
            '存在 ' || v_unresolved
            || ' 个当前 Agent 版本无法唯一推导 Target；请先补齐旧 Agent Target 或 Target Source Binding'
        );
    END IF;

    -- 会话必须能够在升级前唯一确定 Target。来源 Run/Situation/Report、Turn
    -- 证据和旧 Agent 版本依次作为事实来源；最后才使用唯一绑定候选。
    WITH resolved_conversation AS (
        SELECT c.CONVERSATION_ID,
               COALESCE(
                   (SELECT r.TARGET_ID
                      FROM KBOT_OPS_RUN r
                     WHERE r.OPS_RUN_ID = c.SOURCE_RUN_ID),
                   (SELECT s.TARGET_ID
                      FROM KBOT_OPS_SITUATION s
                     WHERE s.SITUATION_ID = c.SOURCE_SITUATION_ID),
                   (SELECT rp.TARGET_ID
                      FROM KBOT_OPS_REPORT rp
                     WHERE rp.REPORT_ID = c.SOURCE_REPORT_ID),
                   (SELECT HEXTORAW(MIN(RAWTOHEX(ct.RESOLVED_TARGET_ID)))
                      FROM KBOT_OPS_CONVERSATION_TURN ct
                     WHERE ct.CONVERSATION_ID = c.CONVERSATION_ID
                       AND ct.RESOLVED_TARGET_ID IS NOT NULL
                    HAVING COUNT(DISTINCT ct.RESOLVED_TARGET_ID) = 1),
                   av.TARGET_ID,
                   (SELECT HEXTORAW(MIN(RAWTOHEX(tb.TARGET_ID)))
                      FROM KBOT_OPS_TARGET_BINDING tb
                     WHERE tb.AGENT_ID = c.AGENT_ID
                       AND tb.STATUS = 'ACTIVE'
                    HAVING COUNT(DISTINCT tb.TARGET_ID) = 1),
                   (SELECT HEXTORAW(MIN(RAWTOHEX(tsb.TARGET_ID)))
                      FROM KBOT_OPS_AGENT_VERSION_SOURCE avs
                      JOIN KBOT_OPS_TARGET_SOURCE_BINDING tsb
                        ON tsb.DIAGNOSTIC_SOURCE_ID = avs.DIAGNOSTIC_SOURCE_ID
                       AND tsb.STATUS = 'ACTIVE'
                     WHERE avs.AGENT_VERSION_ID = c.AGENT_VERSION_ID
                    HAVING COUNT(DISTINCT tsb.TARGET_ID) = 1)
               ) AS TARGET_ID
          FROM KBOT_OPS_CONVERSATION c
          JOIN KBOT_OPS_AGENT_VERSION av
            ON av.AGENT_VERSION_ID = c.AGENT_VERSION_ID
    )
    SELECT COUNT(*)
      INTO v_unresolved
      FROM resolved_conversation
     WHERE TARGET_ID IS NULL;

    IF v_unresolved > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20003,
            '存在 ' || v_unresolved
            || ' 条历史会话无法唯一推导 Target；升级未开始，请先人工补齐关联'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE('升级前置校验通过');
END;
/

PROMPT [2/9] 增加 Target 连接与受控变更开关

ALTER TABLE KBOT_OPS_TARGET ADD (
    READONLY_CONNECTION_ENABLED NUMBER(1) DEFAULT 0 NOT NULL,
    CONTROLLED_CHANGE_ENABLED NUMBER(1) DEFAULT 0 NOT NULL
);

UPDATE KBOT_OPS_TARGET t
   SET READONLY_CONNECTION_ENABLED = CASE
           WHEN t.ENDPOINT_JSON IS NOT NULL
            AND t.DIAGNOSTIC_CREDENTIAL_ID IS NOT NULL THEN 1
           ELSE 0
       END,
       CONTROLLED_CHANGE_ENABLED = CASE
           WHEN t.ENDPOINT_JSON IS NOT NULL
            AND t.DIAGNOSTIC_CREDENTIAL_ID IS NOT NULL
            AND t.EXECUTION_CREDENTIAL_ID IS NOT NULL
            AND (
                EXISTS (
                    SELECT 1
                      FROM KBOT_OPS_AGENT_VERSION av
                      JOIN KBOT_OPS_POLICY p
                        ON p.POLICY_ID = av.POLICY_ID
                     WHERE av.TARGET_ID = t.TARGET_ID
                       AND JSON_VALUE(
                               p.RULES_JSON,
                               '$.allow_agent_execution'
                               RETURNING VARCHAR2(5)
                           ) = 'true'
                )
                OR EXISTS (
                    SELECT 1
                      FROM KBOT_OPS_TARGET_BINDING tb
                     WHERE tb.TARGET_ID = t.TARGET_ID
                       AND tb.ALLOW_MUTATION = 1
                       AND tb.STATUS = 'ACTIVE'
                )
            ) THEN 1
           ELSE 0
       END;

ALTER TABLE KBOT_OPS_TARGET ADD CONSTRAINT CK_OPS_TARGET_ACCESS CHECK (
    READONLY_CONNECTION_ENABLED IN (0, 1)
    AND CONTROLLED_CHANGE_ENABLED IN (0, 1)
    AND (CONTROLLED_CHANGE_ENABLED = 0 OR READONLY_CONNECTION_ENABLED = 1)
);

PROMPT [3/9] 创建 Agent 版本与逻辑 Target 多对多关系

CREATE TABLE KBOT_OPS_AGENT_VERSION_TARGET (
    AGENT_VERSION_ID RAW(16) NOT NULL,
    TARGET_ID RAW(16) NOT NULL,
    CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT PK_OPS_AGENT_VERSION_TARGET
        PRIMARY KEY (AGENT_VERSION_ID, TARGET_ID),
    CONSTRAINT FK_OPS_AGENT_VER_TARGET_VER FOREIGN KEY (AGENT_VERSION_ID)
        REFERENCES KBOT_OPS_AGENT_VERSION (AGENT_VERSION_ID) ON DELETE CASCADE,
    CONSTRAINT FK_OPS_AGENT_VER_TARGET_TARGET FOREIGN KEY (TARGET_ID)
        REFERENCES KBOT_OPS_TARGET (TARGET_ID)
);

CREATE INDEX IX_OPS_AGENT_VER_TARGET_TARGET
    ON KBOT_OPS_AGENT_VERSION_TARGET (TARGET_ID);

PROMPT [4/9] 回填 Agent 版本 Target 集合

-- 首选旧版本直接绑定的 Target，保持原有行为不变。
INSERT INTO KBOT_OPS_AGENT_VERSION_TARGET (
    AGENT_VERSION_ID, TARGET_ID, CREATED_AT
)
SELECT AGENT_VERSION_ID, TARGET_ID, CURRENT_TIMESTAMP
  FROM KBOT_OPS_AGENT_VERSION
 WHERE TARGET_ID IS NOT NULL;

-- 兼容早期由 KBOT_OPS_TARGET_BINDING 表达的明确 Agent-Target 关系。
INSERT INTO KBOT_OPS_AGENT_VERSION_TARGET (
    AGENT_VERSION_ID, TARGET_ID, CREATED_AT
)
SELECT av.AGENT_VERSION_ID, tb.TARGET_ID, CURRENT_TIMESTAMP
  FROM KBOT_OPS_AGENT_VERSION av
  JOIN KBOT_OPS_TARGET_BINDING tb
    ON tb.AGENT_ID = av.AGENT_ID
   AND tb.STATUS = 'ACTIVE'
 WHERE NOT EXISTS (
       SELECT 1
         FROM KBOT_OPS_AGENT_VERSION_TARGET avt
        WHERE avt.AGENT_VERSION_ID = av.AGENT_VERSION_ID
          AND avt.TARGET_ID = tb.TARGET_ID
 );

-- 仅当一个旧 Agent 版本的全部启用 Source Binding 只指向一个 Target 时才推导，
-- 不把共享 Prometheus 中的多个数据库错误地全部绑定给 Agent。
INSERT INTO KBOT_OPS_AGENT_VERSION_TARGET (
    AGENT_VERSION_ID, TARGET_ID, CREATED_AT
)
SELECT avs.AGENT_VERSION_ID,
       HEXTORAW(MIN(RAWTOHEX(tsb.TARGET_ID))),
       CURRENT_TIMESTAMP
  FROM KBOT_OPS_AGENT_VERSION_SOURCE avs
  JOIN KBOT_OPS_TARGET_SOURCE_BINDING tsb
    ON tsb.DIAGNOSTIC_SOURCE_ID = avs.DIAGNOSTIC_SOURCE_ID
   AND tsb.STATUS = 'ACTIVE'
 WHERE NOT EXISTS (
       SELECT 1
         FROM KBOT_OPS_AGENT_VERSION_TARGET avt
        WHERE avt.AGENT_VERSION_ID = avs.AGENT_VERSION_ID
 )
 GROUP BY avs.AGENT_VERSION_ID
HAVING COUNT(DISTINCT tsb.TARGET_ID) = 1;

PROMPT [5/9] 为历史会话回填冻结 Target

ALTER TABLE KBOT_OPS_CONVERSATION ADD (TARGET_ID RAW(16));

UPDATE KBOT_OPS_CONVERSATION c
   SET TARGET_ID = COALESCE(
       (SELECT r.TARGET_ID
          FROM KBOT_OPS_RUN r
         WHERE r.OPS_RUN_ID = c.SOURCE_RUN_ID),
       (SELECT s.TARGET_ID
          FROM KBOT_OPS_SITUATION s
         WHERE s.SITUATION_ID = c.SOURCE_SITUATION_ID),
       (SELECT rp.TARGET_ID
          FROM KBOT_OPS_REPORT rp
         WHERE rp.REPORT_ID = c.SOURCE_REPORT_ID),
       (SELECT HEXTORAW(MIN(RAWTOHEX(ct.RESOLVED_TARGET_ID)))
          FROM KBOT_OPS_CONVERSATION_TURN ct
         WHERE ct.CONVERSATION_ID = c.CONVERSATION_ID
           AND ct.RESOLVED_TARGET_ID IS NOT NULL
        HAVING COUNT(DISTINCT ct.RESOLVED_TARGET_ID) = 1),
       (SELECT av.TARGET_ID
          FROM KBOT_OPS_AGENT_VERSION av
         WHERE av.AGENT_VERSION_ID = c.AGENT_VERSION_ID),
       (SELECT HEXTORAW(MIN(RAWTOHEX(avt.TARGET_ID)))
          FROM KBOT_OPS_AGENT_VERSION_TARGET avt
         WHERE avt.AGENT_VERSION_ID = c.AGENT_VERSION_ID
        HAVING COUNT(DISTINCT avt.TARGET_ID) = 1)
   );

DECLARE
    v_unresolved NUMBER;
BEGIN
    SELECT COUNT(*)
      INTO v_unresolved
      FROM KBOT_OPS_CONVERSATION
     WHERE TARGET_ID IS NULL;
    IF v_unresolved > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20004,
            '回填后仍有 ' || v_unresolved || ' 条会话缺少 Target'
        );
    END IF;
END;
/

ALTER TABLE KBOT_OPS_CONVERSATION MODIFY (TARGET_ID NOT NULL);
ALTER TABLE KBOT_OPS_CONVERSATION ADD CONSTRAINT FK_OPS_CONV_TARGET
    FOREIGN KEY (TARGET_ID) REFERENCES KBOT_OPS_TARGET (TARGET_ID);
CREATE INDEX IX_OPS_CONV_TARGET ON KBOT_OPS_CONVERSATION (TARGET_ID);

-- 确保由历史会话确定的 Target 也属于对应的冻结 Agent 版本。
INSERT INTO KBOT_OPS_AGENT_VERSION_TARGET (
    AGENT_VERSION_ID, TARGET_ID, CREATED_AT
)
SELECT DISTINCT c.AGENT_VERSION_ID, c.TARGET_ID, CURRENT_TIMESTAMP
  FROM KBOT_OPS_CONVERSATION c
 WHERE NOT EXISTS (
       SELECT 1
         FROM KBOT_OPS_AGENT_VERSION_TARGET avt
        WHERE avt.AGENT_VERSION_ID = c.AGENT_VERSION_ID
          AND avt.TARGET_ID = c.TARGET_ID
 );

PROMPT [6/9] 删除 Agent 版本旧单 Target 列

ALTER TABLE KBOT_OPS_AGENT_VERSION DROP CONSTRAINT FK_OPS_AGENT_VERSION_TARGET;
DROP INDEX IX_OPS_AGENT_VERSION_TARGET;
ALTER TABLE KBOT_OPS_AGENT_VERSION DROP COLUMN TARGET_ID;

PROMPT [7/9] 更新 Schema 版本投影

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    15 AS SCHEMA_VERSION,
    'aiops-oracle-v5' AS CONTRACT_VERSION
FROM DUAL;

PROMPT [8/9] 提交升级

COMMIT;

PROMPT [9/9] 输出升级结果

SELECT COMPONENT, SCHEMA_VERSION, CONTRACT_VERSION
  FROM KBOT_V_OPS_SCHEMA_VERSION;

SELECT
    (SELECT COUNT(*) FROM KBOT_OPS_TARGET) AS TARGET_COUNT,
    (SELECT COUNT(*) FROM KBOT_OPS_AGENT) AS AGENT_COUNT,
    (SELECT COUNT(*) FROM KBOT_OPS_AGENT_VERSION_TARGET)
        AS AGENT_TARGET_COUNT,
    (SELECT COUNT(*) FROM KBOT_OPS_DIAGNOSTIC_SOURCE) AS SOURCE_COUNT,
    (SELECT COUNT(*) FROM KBOT_OPS_CONVERSATION) AS CONVERSATION_COUNT,
    (SELECT COUNT(*) FROM KBOT_MANAGED_CREDENTIAL) AS CREDENTIAL_COUNT
FROM DUAL;

PROMPT AIOps Schema 14 -> 15 原位升级完成；监控源及 Webhook 凭据数据未重建。

WHENEVER SQLERROR CONTINUE
