-- ==========================================
-- 1. 新增表：kbot_md_parser_conf
-- ==========================================
CREATE TABLE kbot_md_parser_conf (
    parser_conf_id NUMBER(38, 0) GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    domain_id NUMBER(38, 0),
    engine VARCHAR2(10),
    parser_params CLOB CONSTRAINT chk_parser_params_json CHECK (parser_params IS JSON),
    created_by VARCHAR2(256),
    created_time DATE DEFAULT CURRENT_DATE,
    updated_by VARCHAR2(256),
    updated_time DATE DEFAULT CURRENT_DATE
);

COMMENT ON TABLE kbot_md_parser_conf IS '解析器配置表';
COMMENT ON COLUMN kbot_md_parser_conf.parser_conf_id IS '自增ID，主键';
COMMENT ON COLUMN kbot_md_parser_conf.domain_id IS 'Domain ID, referencing domain table';
COMMENT ON COLUMN kbot_md_parser_conf.engine IS 'Parser engine：Enum：ParserEngine';
COMMENT ON COLUMN kbot_md_parser_conf.parser_params IS 'Parser parameters';
COMMENT ON COLUMN kbot_md_parser_conf.created_by IS 'Creator user';
COMMENT ON COLUMN kbot_md_parser_conf.created_time IS 'Creation time';
COMMENT ON COLUMN kbot_md_parser_conf.updated_by IS 'Updater user';
COMMENT ON COLUMN kbot_md_parser_conf.updated_time IS 'Update time';

-- ==========================================
-- 2. 表变更：kbot_md_kb_files
-- ==========================================
ALTER TABLE kbot_md_kb_files ADD (batch VARCHAR2(256));
COMMENT ON COLUMN kbot_md_kb_files.batch IS '批次名称';

ALTER TABLE kbot_md_kb_files DROP COLUMN batch_id;

-- ==========================================
-- 3. 表变更：kbot_md_kb
-- ==========================================
-- 删除旧模型字段
ALTER TABLE kbot_md_kb DROP COLUMN txt_embed_model_id;
ALTER TABLE kbot_md_kb DROP COLUMN img_embed_model_id;
ALTER TABLE kbot_md_kb DROP COLUMN img2txt_model_id;
ALTER TABLE kbot_md_kb DROP COLUMN llm_model_id;

-- 添加新配置字段
ALTER TABLE kbot_md_kb ADD (
    engine VARCHAR2(50),
    models CLOB CONSTRAINT chk_kb_models_json CHECK (models IS JSON),
    dbconf CLOB CONSTRAINT chk_kb_dbconf_json CHECK (dbconf IS JSON)
);

COMMENT ON COLUMN kbot_md_kb.engine IS '知识库解析引擎类型';
COMMENT ON COLUMN kbot_md_kb.models IS '知识库关联的模型配置参数';
COMMENT ON COLUMN kbot_md_kb.dbconf IS '知识库关联的数据库配置参数';

-- ==========================================
-- 4. 表变更：kbot_md_user_profile
-- ==========================================
TRUNCATE TABLE kbot_md_user_profile;
ALTER TABLE kbot_md_user_profile ADD (
    entity_relations CLOB CONSTRAINT chk_up_entity_rel_json CHECK (entity_relations IS JSON),
    correction_history CLOB CONSTRAINT chk_up_corr_hist_json CHECK (correction_history IS JSON)
);

COMMENT ON COLUMN kbot_md_user_profile.entity_relations IS '轻量级实体关联，如产线-负责人';
COMMENT ON COLUMN kbot_md_user_profile.correction_history IS '用户订正过的错误事实或偏好';

-- ==========================================
-- 5. 表变更：kbot_md_conv_context
-- ==========================================
TRUNCATE TABLE kbot_md_conv_context;
ALTER TABLE kbot_md_conv_context ADD (
    current_plan CLOB CONSTRAINT chk_ctx_plan_json CHECK (current_plan IS JSON),
    step_outputs CLOB CONSTRAINT chk_ctx_outputs_json CHECK (step_outputs IS JSON),
    last_relevance_score NUMBER(2,1),
    active_topic VARCHAR2(512)
);

COMMENT ON COLUMN kbot_md_conv_context.current_plan IS 'TaskPlanner 生成的当前待执行步骤';
COMMENT ON COLUMN kbot_md_conv_context.step_outputs IS '上一个执行步骤的输出';
COMMENT ON COLUMN kbot_md_conv_context.last_relevance_score IS '上一个执行步骤的相关性评分';
COMMENT ON COLUMN kbot_md_conv_context.active_topic IS '当前活跃话题标签';

-- ==========================================
-- 6. 表变更：kbot_md_memory_entry
-- ==========================================
TRUNCATE TABLE kbot_md_memory_entry;
-- 新增字段
ALTER TABLE kbot_md_memory_entry ADD (
    user_id VARCHAR2(256) NOT NULL,
    thought CLOB,
    current_plan CLOB CONSTRAINT chk_mem_plan_json CHECK (current_plan IS JSON),
    reasoning_path CLOB CONSTRAINT chk_mem_reason_json CHECK (reasoning_path IS JSON),
    turn_type VARCHAR2(64),
    blocks CLOB CONSTRAINT chk_mem_blocks_json CHECK (blocks IS JSON)
);

COMMENT ON COLUMN kbot_md_memory_entry.user_id IS '用户ID';
COMMENT ON COLUMN kbot_md_memory_entry.thought IS 'LLM 在改写阶段的思考过程';
COMMENT ON COLUMN kbot_md_memory_entry.current_plan IS 'TaskPlanner 生成的当前待执行步骤';
COMMENT ON COLUMN kbot_md_memory_entry.reasoning_path IS '推理路径';
COMMENT ON COLUMN kbot_md_memory_entry.turn_type IS '轮次类型';
COMMENT ON COLUMN kbot_md_memory_entry.blocks IS '流式响应块';

-- 删除旧字段
ALTER TABLE kbot_md_memory_entry DROP COLUMN retrieved_chunks;

-- ==========================================
-- 7. 表变更：kbot_md_agent
-- ==========================================
-- 添加统一配置字段
ALTER TABLE kbot_md_agent ADD (
    models CLOB CONSTRAINT chk_agent_models_json CHECK (models IS JSON)
);
COMMENT ON COLUMN kbot_md_agent.models IS 'AI模型统一配置（整合原LLM、Embedding、Reranker配置）';

-- 删除分散的旧字段
ALTER TABLE kbot_md_agent DROP COLUMN llm_id;
ALTER TABLE kbot_md_agent DROP COLUMN llm_params;
ALTER TABLE kbot_md_agent DROP COLUMN embedding_model_id;
ALTER TABLE kbot_md_agent DROP COLUMN feedback_similarity_flag;
ALTER TABLE kbot_md_agent DROP COLUMN synonym_similarity_flag;
ALTER TABLE kbot_md_agent DROP COLUMN reranker_model_id;
ALTER TABLE kbot_md_agent DROP COLUMN reranker_topk;
ALTER TABLE kbot_md_agent DROP COLUMN reranker_score_threshold;

-- ==========================================
-- 8. 删除表：kbot_md_kb_batch
-- ==========================================
DROP TABLE kbot_md_kb_batch PURGE;