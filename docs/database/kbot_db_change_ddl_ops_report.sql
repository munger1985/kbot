-- kbot3 AIOps 执行报告表 (Oracle 23ai)
-- 功能: 存储每次自愈操作的完整审计报告
-- 执行: sqlplus <user>/<password>@<db> @this_file.sql

CREATE TABLE ops_execution_report (
    id                VARCHAR2(64) DEFAULT SYS_GUID() NOT NULL,
    entry_id          VARCHAR2(64)  NOT NULL,
    session_id        VARCHAR2(64)  NOT NULL,
    user_id           VARCHAR2(64),
    agent_id          VARCHAR2(64)  NOT NULL,
    instance_id       VARCHAR2(64)  NOT NULL,
    instance_name     VARCHAR2(256) NOT NULL,
    db_type           VARCHAR2(32)  NOT NULL,
    environment       VARCHAR2(32)  DEFAULT 'prod' NOT NULL,
    trigger_type      VARCHAR2(32)  DEFAULT 'manual' NOT NULL,

    -- 诊断信息
    original_question CLOB          DEFAULT EMPTY_CLOB(),
    diagnosis_summary CLOB          DEFAULT EMPTY_CLOB(),

    -- JSON 快照字段 (Oracle 23ai JSON 类型, 支持 JSON 查询和搜索索引)
    actions_executed     JSON DEFAULT '[]' NOT NULL,
    pre_snapshot         JSON,
    post_snapshot        JSON,
    health_check_result  JSON,
    rollback_info        JSON,

    -- 判定
    verification_status  VARCHAR2(32)  DEFAULT 'skipped' NOT NULL,

    -- 报告正文 (CLOB)
    report_content       CLOB          DEFAULT EMPTY_CLOB(),
    recommendations      CLOB          DEFAULT EMPTY_CLOB(),

    -- 元数据
    total_duration_seconds NUMBER      DEFAULT 0 NOT NULL,
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,

    -- 约束
    CONSTRAINT pk_ops_execution_report PRIMARY KEY (id),
    CONSTRAINT fk_ops_report_entry FOREIGN KEY (entry_id)
        REFERENCES kbot_md_memory_entry(entry_id)
        ON DELETE CASCADE,
    CONSTRAINT chk_ops_report_status CHECK (
        verification_status IN ('verified', 'degraded', 'failed', 'skipped')
    ),
    CONSTRAINT chk_ops_report_env CHECK (
        environment IN ('prod', 'staging', 'dev')
    )
);

-- 索引
CREATE INDEX idx_ops_rep_entry    ON ops_execution_report(entry_id);
CREATE INDEX idx_ops_rep_instance ON ops_execution_report(instance_id);
CREATE INDEX idx_ops_rep_created  ON ops_execution_report(created_at DESC);
CREATE INDEX idx_ops_rep_status   ON ops_execution_report(verification_status);

-- Oracle 23ai JSON 搜索索引 (加速 JSON 字段内查询)
CREATE SEARCH INDEX idx_ops_rep_json
    ON ops_execution_report(actions_executed)
    FOR JSON;

-- 注释
COMMENT ON TABLE ops_execution_report IS 'AIOps 自愈执行报告';
COMMENT ON COLUMN ops_execution_report.verification_status IS 'verified / degraded / failed / skipped';
COMMENT ON COLUMN ops_execution_report.report_content IS 'Markdown 格式完整报告';
COMMENT ON COLUMN ops_execution_report.recommendations IS 'LLM 生成的后续优化建议';
COMMENT ON COLUMN ops_execution_report.actions_executed IS '执行的变更动作列表 [{sql, impact, risk_level}]';
