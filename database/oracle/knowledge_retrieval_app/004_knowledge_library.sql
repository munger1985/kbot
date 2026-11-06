-- 知识检索 App 资料组织；不创建公共 KC 集合或跨 App 共享授权。
CREATE TABLE KBOT_KR_KNOWLEDGE_LIBRARY (
    library_id RAW(16) NOT NULL,
    domain_id NUMERIC(38, 0) NOT NULL,
    display_name VARCHAR2(256 CHAR) NOT NULL,
    description CLOB,
    status VARCHAR2(32 CHAR) NOT NULL,
    grants_json JSON NOT NULL,
    configuration_json JSON NOT NULL,
    created_by VARCHAR2(256 CHAR) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    row_version INTEGER NOT NULL,
    PRIMARY KEY (library_id)
);
CREATE INDEX IX_KR_LIBRARY_DOMAIN ON KBOT_KR_KNOWLEDGE_LIBRARY (domain_id, status);
