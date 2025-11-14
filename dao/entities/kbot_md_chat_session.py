from sqlalchemy import String, Numeric, CLOB, TIMESTAMP, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.oracle import NUMBER
from .base import Base
from typing import Optional, Any
from datetime import datetime

class KbotMdChatSession(Base):
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
        return f"KbotBizChatSession(qa_id={self.qa_id!r}, session_id={self.session_id!r}, agent_id={self.agent_id!r})"


class KbotMdChatReferences(Base):
    """聊天引用数据表"""
    
    ref_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="自增主键")
    qa_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="关联的问答ID")
    chunk_type: Mapped[int | None] = mapped_column(NUMBER(1), comment="引用类型")
    chunk_file_path: Mapped[str | None] = mapped_column(String(256), comment="引用文件路径")
    file_ext: Mapped[str | None] = mapped_column(String(10), comment="文件扩展名")
    page_num: Mapped[int | None] = mapped_column(NUMBER(5), comment="页码")
    chunk_content: Mapped[str | None] = mapped_column(CLOB, comment="引用内容")
    download_link: Mapped[str | None] = mapped_column(String(512), comment="下载链接")
    preview_link: Mapped[str | None] = mapped_column(String(512), comment="预览链接")
    similarity_score: Mapped[float | None] = mapped_column(NUMBER(5, 3), comment="相似度分数")
    reranker_score: Mapped[float | None] = mapped_column(NUMBER(5, 3), comment="重排分数")

    def __repr__(self):
        return f"KbotBizChatReferences(ref_id={self.ref_id!r}, qa_id={self.qa_id!r}, chunk_type={self.chunk_type!r})"


# class KbotMdChatSession(Base):
#     """聊天问答数据表"""
    
#     qa_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="自增主键")
#     session_id: Mapped[str] = mapped_column(String(256), nullable=False, comment="关联的会话ID")
#     agent_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="关联的智能体ID")
#     question: Mapped[Optional[str]] = mapped_column(CLOB, comment="用户问题")
#     answer: Mapped[Optional[str]] = mapped_column(CLOB, comment="AI回答")
#     qa_embedding: Mapped[Optional[str]] = mapped_column(Text, comment="问答向量")
#     feedback: Mapped[Optional[int]] = mapped_column(NUMBER(1), comment="反馈评价")
#     username: Mapped[Optional[str]] = mapped_column(String(64), comment="提问者")
#     request_time: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, comment="请求时间")
#     response_time: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, comment="响应时间")

#     def __init__(self, **kwargs):
#         # 过滤掉非ORM字段的参数
#         orm_fields = {k: v for k, v in kwargs.items() if hasattr(self.__class__, k)}
#         super().__init__(**orm_fields)
#         # 设置非ORM字段
#         self.qa_data = kwargs.get('qa_data', [])

#     def to_dict(self) -> dict:
#         """转换为字典，支持JSON序列化"""
#         def serialize_datetime(dt):
#             """序列化datetime对象"""
#             if dt is None:
#                 return None
#             return dt.strftime("%Y-%m-%d %H:%M:%S.%f") if hasattr(dt, 'strftime') else str(dt)

#         result = {
#             'session_id': self.session_id,
#             'agent_id': int(self.agent_id) if self.agent_id else None,
#             'qa_data': []
#         }
        
#         # 序列化qa_data
#         if hasattr(self, 'qa_data') and self.qa_data:
#             for qa in self.qa_data:
#                 qa_dict = {
#                     'question': qa.question,
#                     'answer': qa.answer,
#                     'qa_embedding': qa.qa_embedding,
#                     'feedback': qa.feedback,
#                     'by': qa.by,
#                     'request_time': serialize_datetime(qa.request_time),
#                     'response_time': serialize_datetime(qa.response_time),
#                     'references': []
#                 }
                
#                 # 序列化references
#                 if hasattr(qa, 'references') and qa.references:
#                     for ref in qa.references:
#                         ref_dict = {
#                             'chunk_type': ref.chunk_type,
#                             'chunk_file_path': ref.chunk_file_path,
#                             'file_ext': ref.file_ext,
#                             'page_num': ref.page_num,
#                             'content': ref.content,
#                             'download_link': ref.download_link,
#                             'preview_link': ref.preview_link,
#                             'similarity_score': float(ref.similarity_score) if ref.similarity_score is not None else None,
#                             'reranker_score': float(ref.reranker_score) if ref.reranker_score is not None else None
#                         }
#                         qa_dict['references'].append(ref_dict)
                
#                 result['qa_data'].append(qa_dict)
        
#         return result

#     def __repr__(self):
#         return f"KbotMdChatSession(qa_id={self.qa_id!r}, session_id={self.session_id!r}, agent_id={self.agent_id!r})"