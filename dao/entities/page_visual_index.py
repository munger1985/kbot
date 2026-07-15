"""页面视觉索引实体 — 适配 Oracle 23ai"""

import uuid
from datetime import datetime
from sqlalchemy import String, Numeric, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, VectorField


class PageVisualIndexEntity(BaseEntity):
    __tablename__ = "page_visual_index"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="主键")
    file_id: Mapped[str] = mapped_column(String(256), nullable=False, comment="关联的文件 ID")
    kb_id: Mapped[str] = mapped_column(String(256), nullable=False, comment="关联的知识库 ID")
    page_no: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="页码编号")
    image_path: Mapped[str] = mapped_column(String(1000), nullable=False, comment="页面截图的存储路径")
    embedding: Mapped[list[float] | None] = mapped_column(VectorField(), comment="ColQwen2 mean-pooled 视觉 embedding（全精度 vector）")
    caption: Mapped[str | None] = mapped_column(Text, comment="VLM 生成的页面摘要描述")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), comment="记录创建时间")
