from enum import Enum
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.sql import func
from sqlalchemy import UniqueConstraint, ForeignKey
from .base import Base

class KbCategory(str, Enum):
    """Knowledge base category enumeration."""
    KBOT = "KBot"
    IMAGE_SEARCH = "ImageSearch"
    GEN_REPORT = "GenReport"
    TRANSLATE = "Translate"
    SUMMARY = "Summary"

class KbStatus(int, Enum):
    """Knowledge base status enumeration."""
    NEW = 1
    ENABLED = 2
    DISABLED = 3
    ARCHIVED = 4

class KbotMdKb(Base):
    """Knowledge base entity for KBOT_MD_KB table."""
    
    __tablename__ = "KBOT_MD_KB"
    __table_args__ = (
        UniqueConstraint('app_id', 'domain_id', 'kb_name'),
    )
    
    kb_id = Column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="知识库唯一标识，主键"
    )
    app_id = Column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID，与DOMAIN_ID和KB_NAME组成联合唯一约束"
    )
    domain_id = Column(
        NUMBER(38, 0),
        ForeignKey("KBOT_MD_DOMAIN.domain_id"),
        comment="关联的业务域ID，外键引用KBOT_KB_DOMAIN表"
    )
    kb_name = Column(
        String(256),
        comment="知识库名称，在同一业务域下具有唯一性"
    )
    kb_category = Column(
        String(256),
        comment="知识库类型：KBot(文搜文/文搜图)、ImageSearch(图搜图)、GenReport、Translate、Summary"
    )
    descs = Column(
        String(512),
        comment="知识库详细描述信息"
    )
    db_conn_id = Column(
        NUMBER(38, 0),
        comment="关联的向量数据库连接配置ID"
    )
    embed_mode_id = Column(
        NUMBER(38, 0),
        comment="使用的嵌入模型ID"
    )
    kb_status = Column(
        NUMBER(38, 0),
        comment="知识库状态：1-NEW(新建)、2-ENABLED(启用)、3-DISABLED(禁用)、4-ARCHIVED(归档)"
    )
    created_by = Column(
        String(512),
        comment="记录创建人"
    )
    created_time = Column(
        Date, 
        server_default=func.current_date(),
        comment="记录创建时间，默认系统当前时间"
    )
    updated_by = Column(
        String(512),
        comment="最后修改人"
    )
    updated_time = Column(
        Date, 
        server_default=func.current_date(),
        comment="最后修改时间，默认系统当前时间"
    )