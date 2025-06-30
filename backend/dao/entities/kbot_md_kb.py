from enum import Enum
from sqlalchemy import String, Date
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.sql import func
from sqlalchemy import UniqueConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

class KbCategory(int, Enum):
    """Knowledge base category enumeration."""
    KBOT = 1
    IMAGE_SEARCH = 2
    GEN_REPORT = 3
    TRANSLATE = 4
    SUMMARY = 5

class KbStatus(int, Enum):
    """Knowledge base status enumeration."""
    NEW = 1
    ENABLED = 2
    DISABLED = 3
    ARCHIVED = 4

class KbotMdKb(Base):
    """Knowledge base entity for KBOT_MD_KB table."""

    __table_args__ = (
        UniqueConstraint('app_id', 'domain_id', 'kb_name'),
    )
    
    kb_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="知识库唯一标识，主键"
    )
    app_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID，与DOMAIN_ID和KB_NAME组成联合唯一约束"
    )
    domain_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        ForeignKey("KBOT_MD_DOMAIN.domain_id"),
        comment="关联的业务域ID，外键引用KBOT_KB_DOMAIN表"
    )
    kb_name: Mapped[str | None] = mapped_column(
        String(256),
        comment="知识库名称，在同一业务域下具有唯一性"
    )
    kb_category: Mapped[str | None] = mapped_column(
        NUMBER(38, 0),
        comment="知识库类型：KBot(文搜文/文搜图)、ImageSearch(图搜图)、GenReport、Translate、Summary"
    )
    descs: Mapped[str | None] = mapped_column(
        String(512),
        comment="知识库详细描述信息"
    )
    db_conn_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="关联的向量数据库连接配置ID"
    )
    embed_model_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="使用的嵌入模型ID"
    )
    kb_status: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="知识库状态：1-NEW(新建)、2-ENABLED(启用)、3-DISABLED(禁用)、4-ARCHIVED(归档)"
    )
    created_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="记录创建人"
    )
    created_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="记录创建时间，默认系统当前时间"
    )
    updated_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="最后修改人"
    )
    updated_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="最后修改时间，默认系统当前时间"
    )