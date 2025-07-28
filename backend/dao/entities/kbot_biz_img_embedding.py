from sqlalchemy import String, CLOB, Numeric
from sqlalchemy.dialects.oracle import VECTOR
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class KbotBizImgEmbedding(Base):
    """图片向量嵌入表"""

    embed_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment='图片向量唯一标识，主键')
    kb_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment='关联的知识库ID')
    file_id: Mapped[int | None] = mapped_column(Numeric, comment='关联的图片文件ID')
    img_txt: Mapped[str | None] = mapped_column(CLOB, comment='从图片中提取的文本内容')
    chunk_metadata: Mapped[dict] = mapped_column(CLOB, nullable=False, comment='JSON格式的图片元数据')
    embedding: Mapped[list] = mapped_column(VECTOR, nullable=False, comment='图片向量(FLOAT64格式)')

    def __repr__(self):
        return f"KbotBizImgEmbedding(embed_id={self.embed_id!r}, kb_id={self.kb_id!r}, file_id={self.file_id!r})"