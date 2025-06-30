from enum import Enum
from sqlalchemy import String, Date, CLOB
from sqlalchemy.dialects.oracle import NUMBER, VARCHAR2
from sqlalchemy.sql import func
from sqlalchemy import Sequence
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from .base import Base

class PromptCategory(int, Enum):
    """Prompt category enumeration."""
    RAG = 1
    SUMMARY = 2

class KbotMdPrompt(Base):
    """Prompt entity for KBOT_MD_PROMPT table."""

    prompt_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="提示词唯一标识，主键"
    )
    app_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    domain_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="关联的业务域ID（可选）"
    )
    name: Mapped[str | None] = mapped_column(
        String(256),
        comment="提示词名称"
    )
    prompt_category: Mapped[str] = mapped_column(
        NUMBER(38, 0),
        nullable=False,
        comment="提示词类型：RAG-知识库, SUMMARY-摘要"
    )
    template: Mapped[str | None] = mapped_column(
        CLOB,
        comment="提示词模板内容（CLOB大文本）"
    )
    status: Mapped[str | None] = mapped_column(
        VARCHAR2(1),
        comment="提示词状态：Y-启用, N-禁用"
    )
    descs: Mapped[str | None] = mapped_column(
        String(512),
        comment="提示词详细描述"
    )
    created_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="创建用户"
    )
    created_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="创建时间，默认系统当前时间"
    )
    updated_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="最后修改用户"
    )
    updated_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="最后修改时间，默认系统当前时间"
    )