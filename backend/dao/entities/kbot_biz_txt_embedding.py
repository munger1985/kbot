from sqlalchemy import String, CLOB, Numeric, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.oracle import VECTOR
from .base import Base


class KbotBizTxtEmbedding(Base):
    """文本向量嵌入基类"""
    
    embed_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment='向量记录唯一标识，主键')
    chunk_doc: Mapped[str] = mapped_column(CLOB, nullable=False, comment='文本块原始内容')
    chunk_metadata: Mapped[dict] = mapped_column(String(4000), nullable=False, comment='JSON格式的文本块元数据')
    multi_vector: Mapped[int | None] = mapped_column(Numeric, comment='多向量标识(预留字段)')
    file_id: Mapped[int | None] = mapped_column(Numeric, comment='关联的文本文件ID')
    embedding: Mapped[list] = mapped_column(VECTOR, nullable=False, comment='文本向量(FLOAT64格式)')

    def __repr__(self):
        return f"KbotBizTxtEmbedding(embed_id={self.embed_id!r}, file_id={self.file_id!r})"