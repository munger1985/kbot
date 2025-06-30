from sqlalchemy import String, CLOB, Numeric, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from backend.core.database.vec_oracle import OracleVectorType
from backend.core.database.vec_pg import PgVectorType
from .base import Base

class KbotBizImgEmbedding(Base):
    """图片向量嵌入基类"""
    
    embed_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment='图片向量唯一标识，主键')
    img_txt: Mapped[str | None] = mapped_column(CLOB, comment='从图片中提取的文本内容')
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, comment='JSON格式的图片元数据')
    chunk_id: Mapped[str] = mapped_column(String(256), nullable=False, comment='关联的图片块ID')
    file_id: Mapped[int | None] = mapped_column(Numeric, comment='关联的图片文件ID')

class KbotBizImgEmbeddingOracle(KbotBizImgEmbedding):
    """Oracle数据库的图片向量嵌入模型"""
    embedding: Mapped[list] = mapped_column(OracleVectorType(1024), nullable=False, comment='图片向量(FLOAT32格式)')

class KbotBizImgEmbeddingPG(KbotBizImgEmbedding):
    """PostgreSQL数据库的图片向量嵌入模型"""
    embedding: Mapped[list] = mapped_column(PgVectorType(1024), nullable=False, comment='图片向量(FLOAT32格式)')

class KbotBizImgEmbeddingMySQL(KbotBizImgEmbedding):
    """MySQL数据库的图片向量嵌入模型"""
    embedding: Mapped[list] = mapped_column(JSON, nullable=False, comment='图片向量(FLOAT32格式，JSON编码存储)')