from sqlalchemy import String, CLOB, Numeric, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from backend.core.database.vec_oracle import OracleVectorType
from backend.core.database.vec_pg import PgVectorType
from .base import Base

class KbotBizTxtEmbedding(Base):
    """文本向量嵌入基类"""
    
    embed_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment='向量记录唯一标识，主键')
    chunk_doc: Mapped[str] = mapped_column(CLOB, nullable=False, comment='文本块原始内容')
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, comment='JSON格式的文本块元数据')
    multi_vector: Mapped[int | None] = mapped_column(Numeric, comment='多向量标识(预留字段)')
    chunk_id: Mapped[str] = mapped_column(String(256), nullable=False, comment='关联的文本块ID')
    file_id: Mapped[int | None] = mapped_column(Numeric, comment='关联的文本文件ID')

class KbotBizTxtEmbeddingOracle(KbotBizTxtEmbedding):
    """Oracle数据库的文本向量嵌入模型"""
    embedding: Mapped[list] = mapped_column(OracleVectorType(1024), nullable=False, comment='文本向量(FLOAT32格式)')

class KbotBizTxtEmbeddingPG(KbotBizTxtEmbedding):
    """PostgreSQL数据库的文本向量嵌入模型"""
    embedding: Mapped[list] = mapped_column(PgVectorType(1024), nullable=False, comment='文本向量(FLOAT32格式)')

class KbotBizTxtEmbeddingMySQL(KbotBizTxtEmbedding):
    """MySQL数据库的文本向量嵌入模型"""
    embedding: Mapped[list] = mapped_column(JSON, nullable=False, comment='文本向量(FLOAT32格式，JSON编码存储)')