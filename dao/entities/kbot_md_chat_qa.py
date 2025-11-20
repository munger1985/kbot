from sqlalchemy import String, Numeric, CLOB, TIMESTAMP, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.oracle import NUMBER
from .base import Base


class KbotMdChatQa(Base):
    """聊天问答数据表"""
    
    qa_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="自增主键")
    session_id: Mapped[str] = mapped_column(String(256), nullable=False, comment="关联的会话ID")
    agent_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="关联的智能体ID")
    question: Mapped[str | None] = mapped_column(CLOB, comment="用户问题")
    answer: Mapped[str | None] = mapped_column(CLOB, comment="AI回答")
    qa_embedding: Mapped[str | None] = mapped_column(Text, comment="问答向量")
    feedback: Mapped[int | None] = mapped_column(NUMBER(1), comment="反馈评价")
    username: Mapped[str | None] = mapped_column(String(64), comment="提问者")
    request_time: Mapped[TIMESTAMP | None] = mapped_column(TIMESTAMP, comment="请求时间")
    response_time: Mapped[TIMESTAMP | None] = mapped_column(TIMESTAMP, comment="响应时间")

    def __repr__(self):
        return f"KbotMdChatQa(qa_id={self.qa_id!r}, session_id={self.session_id!r}, agent_id={self.agent_id!r})"


class KbotMdChatReferences(Base):
    """聊天引用数据表"""
    
    ref_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="自增主键")
    qa_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="关联的问答ID")
    chunk_type: Mapped[int] = mapped_column(NUMBER(1), comment="引用类型")
    chunk_file_path: Mapped[str | None] = mapped_column(String(256), comment="引用文件路径")
    page_num: Mapped[int | None] = mapped_column(NUMBER(5), comment="页码")
    chunk_content: Mapped[str | None] = mapped_column(CLOB, comment="引用内容")
    download_link: Mapped[str | None] = mapped_column(String(512), comment="下载链接")
    preview_link: Mapped[str | None] = mapped_column(String(512), comment="预览链接")
    similarity_score: Mapped[float | None] = mapped_column(NUMBER(5, 3), comment="相似度分数")
    reranker_score: Mapped[float | None] = mapped_column(NUMBER(5, 3), comment="重排分数")

    def __repr__(self):
        return f"KbotBizChatReferences(ref_id={self.ref_id!r}, qa_id={self.qa_id!r}, chunk_type={self.chunk_type!r})"
