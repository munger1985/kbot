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




-- ========================================================
-- 1. 顶点表 (KBOT_GRAPH_KNOWLEDGE_VERTICES) 
-- ========================================================
CREATE TABLE KBOT_GRAPH_KNOWLEDGE_VERTICES (
    kb_id          NUMBER(38,0) NOT NULL,
    vertex_id      VARCHAR2(64) NOT NULL,
    vertex_name    VARCHAR2(255) NOT NULL,
    vertex_type    VARCHAR2(64) NOT NULL,
    description    CLOB,
    attributes     JSON,
    name_vector    VECTOR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP,
    CONSTRAINT pk_kbot_graph_knowledge_vertices PRIMARY KEY (kb_id, vertex_id)
);
-- 为实体名称和类型建立索引，加速传统检索
CREATE INDEX idx_vertex_name ON KBOT_GRAPH_KNOWLEDGE_VERTICES(kb_id, vertex_name);


COMMENT ON TABLE KBOT_GRAPH_KNOWLEDGE_VERTICES IS '知识图谱顶点表（实体表）：存储从非结构化文档中抽取的业务实体、术语或概念，并支持向量消歧。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_VERTICES.kb_id IS '知识库ID，引用知识库表。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_VERTICES.vertex_id IS '顶点（实体）的唯一标识。推荐使用名称+类型的规范化哈希值（如MD5），以天然实现初级消歧。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_VERTICES.vertex_name IS '实体或概念的实际名称（如"RTX 5080"、"晶圆厚度"）。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_VERTICES.vertex_type IS '实体的业务大类分类（如：技术、设备、指标、组织、概念等）。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_VERTICES.description IS 'LLM 对该实体在上下文中提炼的简要文本定义或描述。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_VERTICES.attributes IS 'JSON 格式的动态扩展属性，用于存储 LLM 针对特定实体类型抽取的个性化非固定字段（如设备的型号、指标的单位等）。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_VERTICES.name_vector IS '基于实体名称/描述生成的原生向量嵌入（Vector Embeddings），用于支持向量相似度检索以及高级实体对齐与合并（消歧）。';


-- ========================================================
-- 2. 边表 (KBOT_GRAPH_KNOWLEDGE_EDGES) 
-- ========================================================
CREATE TABLE KBOT_GRAPH_KNOWLEDGE_EDGES (
    kb_id          NUMBER(38,0) NOT NULL,
    edge_id        VARCHAR2(64) NOT NULL,
    source_id      VARCHAR2(64) NOT NULL,
    target_id      VARCHAR2(64) NOT NULL,
    relation_type  VARCHAR2(128) NOT NULL,
    weight         NUMBER(10) DEFAULT 1,
    attributes     JSON,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP,
    CONSTRAINT pk_graph_knowledge_edges PRIMARY KEY (kb_id, edge_id),
    CONSTRAINT fk_edge_source FOREIGN KEY (kb_id, source_id) REFERENCES KBOT_GRAPH_KNOWLEDGE_VERTICES(kb_id, vertex_id) ON DELETE CASCADE,
    CONSTRAINT fk_edge_target FOREIGN KEY (kb_id, target_id) REFERENCES KBOT_GRAPH_KNOWLEDGE_VERTICES(kb_id, vertex_id) ON DELETE CASCADE
);

CREATE INDEX idx_edge_source_target ON KBOT_GRAPH_KNOWLEDGE_EDGES(source_id, target_id);

COMMENT ON TABLE KBOT_GRAPH_KNOWLEDGE_EDGES IS '知识图谱边表（关系表）：存储实体与实体之间的有向关系，通过权重记录关系的业务强弱和提及频次。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_EDGES.kb_id IS '知识库ID，引用知识库表。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_EDGES.edge_id IS '边的唯一标识。推荐由源节点ID、目标节点ID和关系类型进行组合哈希生成。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_EDGES.source_id IS '关系的源顶点ID（起点），对应 KNOWLEDGE_VERTICES 的主键。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_EDGES.target_id IS '关系的目标顶点ID（终点），对应 KNOWLEDGE_VERTICES 的主键。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_EDGES.relation_type IS '关系的语义类型（如：属于、导致、测量、引发、集成于等动宾短语或逻辑关系）。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_EDGES.weight IS '关系权重值。若在多个切片中重复抽取到完全相同的边，则此数值递增，数值越大表示该关系在知识库中越核心或被提及越多。';
COMMENT ON COLUMN KBOT_GRAPH_KNOWLEDGE_EDGES.attributes IS 'JSON 格式的动态扩展属性，存储该条边（关系）特有的附加信息。';



-- ========================================================
-- 3. 关系-切片映射表 (KBOT_GRAPH_EDGE_CHUNK_MAP) 
-- ========================================================
CREATE TABLE KBOT_GRAPH_EDGE_CHUNK_MAP (
    kb_id          NUMBER(38,0) NOT NULL,
    edge_id        VARCHAR2(64) NOT NULL,
    chunk_id       VARCHAR2(64) NOT NULL,
    file_id        VARCHAR2(64),
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP,
    CONSTRAINT pk_graph_edge_chunk_map PRIMARY KEY (kb_id, edge_id, chunk_id),
    CONSTRAINT fk_map_edge FOREIGN KEY (kb_id, edge_id) REFERENCES KBOT_GRAPH_KNOWLEDGE_EDGES(kb_id, edge_id) ON DELETE CASCADE
);

CREATE INDEX idx_map_chunk ON KBOT_GRAPH_EDGE_CHUNK_MAP(chunk_id);

COMMENT ON TABLE KBOT_GRAPH_EDGE_CHUNK_MAP IS '图关系与文档切片映射表（中间关联表）：记录每条关系是由哪些具体的文档切片抽取得来的，是图搜索 Skill 溯源到原文 Chunk 的核心依赖。';
COMMENT ON COLUMN KBOT_GRAPH_EDGE_CHUNK_MAP.kb_id IS '知识库ID，引用知识库表。';
COMMENT ON COLUMN KBOT_GRAPH_EDGE_CHUNK_MAP.edge_id IS '边（关系）的唯一标识，对应 KBOT_GRAPH_KNOWLEDGE_EDGES 的主键。';
COMMENT ON COLUMN KBOT_GRAPH_EDGE_CHUNK_MAP.chunk_id IS '提取出该条关系的原始文档切片ID。对应 Vector DB 或系统其他业务表中的 Chunk ID，用于顺藤摸瓜检索原文。';
COMMENT ON COLUMN KBOT_GRAPH_EDGE_CHUNK_MAP.file_id IS '冗余存储的文档唯一标识，方便在用户删除或更新整个文档时进行级联数据清理。';



-- create graph
CREATE PROPERTY GRAPH kbot_knowledge_rag_graph
    VERTEX TABLES (
        KBOT_GRAPH_KNOWLEDGE_VERTICES
            KEY (vertex_id)
            LABEL vertex
            PROPERTIES (kb_id, vertex_id, vertex_name, vertex_type, description, attributes)
    )
    EDGE TABLES (
        KBOT_GRAPH_KNOWLEDGE_EDGES
            KEY (edge_id)
            SOURCE KEY (source_id) REFERENCES KBOT_GRAPH_KNOWLEDGE_VERTICES(vertex_id)
            DESTINATION KEY (target_id) REFERENCES KBOT_GRAPH_KNOWLEDGE_VERTICES(vertex_id)
            LABEL connects_to
            PROPERTIES (kb_id, edge_id, relation_type, weight, attributes)
    );

-- ==========================================
-- 运维Agent (AIOps) 移植 — 新增表
-- ==========================================

-- KBOT_OPS_DB_INSTANCE: 数据库实例资产配置表 (CMDB)
CREATE TABLE KBOT_OPS_DB_INSTANCE (
    instance_id    VARCHAR2(36) PRIMARY KEY,
    instance_name  VARCHAR2(128) NOT NULL,
    environment    VARCHAR2(16) DEFAULT 'dev',
    security_level NUMBER(1) DEFAULT 3,
    db_type        VARCHAR2(32) NOT NULL,
    version_code   NUMBER DEFAULT 0,
    charset        VARCHAR2(32) DEFAULT 'utf8mb4',
    host           VARCHAR2(256) NOT NULL,
    port           NUMBER(5) NOT NULL,
    service_name   VARCHAR2(128),
    database_name  VARCHAR2(128),
    dsn            VARCHAR2(512),
    db_role        VARCHAR2(32) DEFAULT 'primary',
    monitor_type   VARCHAR2(32) DEFAULT 'prometheus',
    prometheus_instance_label VARCHAR2(256),
    cluster_id     VARCHAR2(36),
    ops_user       VARCHAR2(64) NOT NULL,
    encrypted_password VARCHAR2(512) NOT NULL,
    secret_vault_key VARCHAR2(256),
    status         VARCHAR2(16) DEFAULT 'active',
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_ops_instance_host_port UNIQUE (host, port),
    CONSTRAINT ck_ops_db_env CHECK (environment IN ('prod', 'stg', 'dev')),
    CONSTRAINT ck_ops_db_role CHECK (db_role IN ('primary', 'standby', 'cluster_node')),
    CONSTRAINT ck_ops_db_status CHECK (status IN ('active', 'maintenance', 'offline'))
);

COMMENT ON TABLE KBOT_OPS_DB_INSTANCE IS '智能运维线 - 物理与云端数据库实例资产配置表 (CMDB 核心表)';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.instance_id IS '运维全域唯一实例ID (UUID v7 格式)';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.instance_name IS '实例直观可读名称';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.environment IS '环境分级隔离策略: prod, stg, dev';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.security_level IS '安全控制等级, 数值越高触发变更自愈时的审批流越严格';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.db_type IS '数据库引擎内核类型: oracle, postgresql, mysql';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.version_code IS '精准内核版本数字代码, 例如: 26000000';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.host IS '物理机/虚拟机 IP 地址或高可用对外域名';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.port IS '数据库内核监听服务的物理端口号';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.db_role IS '物理角色: primary(主), standby(备), cluster_node';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.monitor_type IS '监控数据源类型: prometheus, zabbix, none';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.prometheus_instance_label IS '对应 Prometheus 采集时的 instance 标签值';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.ops_user IS '大模型运维通道专属的高权/监控账号';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.encrypted_password IS '经过强加密后的运维账号物理密码密文';
COMMENT ON COLUMN KBOT_OPS_DB_INSTANCE.status IS '生命周期状态: active, maintenance, offline';

-- KBOT_OPS_AGENT_CONF: 智能体与运维资产多对多绑定配置表
CREATE TABLE KBOT_OPS_AGENT_CONF (
    id                   NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    agent_id             NUMBER NOT NULL,
    instance_id          VARCHAR2(36) NOT NULL,
    is_mutation_allowed  NUMBER(1) DEFAULT 0,
    require_approval     NUMBER(1) DEFAULT 1,
    max_daily_execution  NUMBER DEFAULT 10,
    created_by           VARCHAR2(64),
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by           VARCHAR2(64),
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_agent_ops UNIQUE (agent_id, instance_id),
    CONSTRAINT fk_agent_ops_agent FOREIGN KEY (agent_id) REFERENCES kbot_md_agent(agent_id),
    CONSTRAINT fk_agent_ops_instance FOREIGN KEY (instance_id) REFERENCES KBOT_OPS_DB_INSTANCE(instance_id)
);

COMMENT ON TABLE KBOT_OPS_AGENT_CONF IS '智能运维线 - 智能体与运维物理资产多对多动态绑定配置表';
COMMENT ON COLUMN KBOT_OPS_AGENT_CONF.id IS '关系配置自增 ID, 主键';
COMMENT ON COLUMN KBOT_OPS_AGENT_CONF.agent_id IS '关联的智能体 Agent ID (int, 对应 kbot_md_agent.agent_id)';
COMMENT ON COLUMN KBOT_OPS_AGENT_CONF.instance_id IS '关联的运维实例 ID (对应 KBOT_OPS_DB_INSTANCE.instance_id)';
COMMENT ON COLUMN KBOT_OPS_AGENT_CONF.is_mutation_allowed IS '是否允许执行变更动作(如 Kill/DDL)';
COMMENT ON COLUMN KBOT_OPS_AGENT_CONF.require_approval IS '是否必须触发人工审批门禁';
COMMENT ON COLUMN KBOT_OPS_AGENT_CONF.max_daily_execution IS '单日高危变更自愈动作上限频次';

