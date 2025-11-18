
drop table kbot_md_chat_session purge;
CREATE TABLE kbot_md_chat_session (
  qa_id NUMBER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1) PRIMARY KEY,
  session_id VARCHAR2(256) NOT NULL,
  agent_id INT NOT NULL,
  question CLOB,
  answer CLOB,
  qa_embedding VECTOR,
  feedback NUMBER(1) DEFAULT 0,
  username VARCHAR2(64),
  request_time TIMESTAMP,
  response_time TIMESTAMP
);

COMMENT ON TABLE kbot_md_chat_session IS '聊天问答数据表';
COMMENT ON COLUMN kbot_md_chat_session.qa_id IS '自增主键';
COMMENT ON COLUMN kbot_md_chat_session.session_id IS '关联的会话ID';
COMMENT ON COLUMN kbot_md_chat_session.agent_id IS '关联的智能体ID';
COMMENT ON COLUMN kbot_md_chat_session.question IS '用户问题';
COMMENT ON COLUMN kbot_md_chat_session.answer IS 'AI回答';
COMMENT ON COLUMN kbot_md_chat_session.qa_embedding IS '问答向量';
COMMENT ON COLUMN kbot_md_chat_session.feedback IS '反馈评价';
COMMENT ON COLUMN kbot_md_chat_session.username IS '提问者';
COMMENT ON COLUMN kbot_md_chat_session.request_time IS '请求时间';
COMMENT ON COLUMN kbot_md_chat_session.response_time IS '响应时间';

drop table kbot_md_chat_references purge;
CREATE TABLE kbot_md_chat_references (
  ref_id NUMBER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1) PRIMARY KEY,
  qa_id NUMBER NOT NULL,
  chunk_type NUMBER(1),
  chunk_file_path VARCHAR2(256),
  page_num NUMBER(5),
  chunk_content CLOB,
  download_link VARCHAR2(512),
  preview_link VARCHAR2(512),
  similarity_score NUMBER(5,3),
  reranker_score NUMBER(5,3)
  );
  COMMENT ON TABLE kbot_md_chat_references IS '聊天引用数据表';
  COMMENT ON COLUMN kbot_md_chat_references.ref_id IS '自增主键';
  COMMENT ON COLUMN kbot_md_chat_references.qa_id IS '关联的问答ID';
  COMMENT ON COLUMN kbot_md_chat_references.chunk_type IS '引用类型';
  COMMENT ON COLUMN kbot_md_chat_references.chunk_file_path IS '引用文件路径';
  COMMENT ON COLUMN kbot_md_chat_references.page_num IS '页码';
  COMMENT ON COLUMN kbot_md_chat_references.chunk_content IS '引用内容';
  COMMENT ON COLUMN kbot_md_chat_references.download_link IS '下载链接';
  COMMENT ON COLUMN kbot_md_chat_references.preview_link IS '预览链接';
  COMMENT ON COLUMN kbot_md_chat_references.similarity_score IS '相似度分数';
  COMMENT ON COLUMN kbot_md_chat_references.reranker_score IS '重排分数';