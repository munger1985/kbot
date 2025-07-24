from sqlalchemy import String, CLOB, Numeric
from sqlalchemy.dialects.oracle import VECTOR
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class KbotBizTxtEmbedding(Base):
    """文本向量嵌入表"""
    
    embed_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment='向量记录唯一标识，主键')
    file_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment='关联的文本文件ID')
    chunk_doc: Mapped[str] = mapped_column(CLOB, nullable=False, comment='文本块原始内容')
    chunk_metadata: Mapped[dict] = mapped_column(CLOB, nullable=False, comment='JSON格式的文本块元数据')
    multi_vector: Mapped[int | None] = mapped_column(Numeric(38, 0), comment='多向量标识(预留字段)')
    embedding: Mapped[list] = mapped_column(VECTOR, nullable=False, comment='文本向量(FLOAT64格式)')

    def __repr__(self):
        return f"KbotBizTxtEmbedding(embed_id={self.embed_id!r}, file_id={self.file_id!r})"