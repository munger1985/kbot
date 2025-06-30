from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.sql import func
from sqlalchemy import UniqueConstraint
from .base import Base

class KbotMdDomain(Base):
    """Business domain entity for KBOT_MD_DOMAIN table."""
    
    __tablename__ = "KBOT_MD_DOMAIN"
    __table_args__ = (
        UniqueConstraint('app_id', 'name'),
    )
    
    domain_id = Column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="业务域ID，主键"
    )
    app_id = Column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属APP ID，与NAME组成唯一索引"
    )
    name = Column(
        String(256),
        comment="业务域名称"
    )
    status = Column(
        NUMBER(1, 0),
        comment="提示词状态：1-启用，0-禁用"
    )
    descs = Column(
        String(512),
        comment="业务域描述"
    )
    created_by = Column(
        String(512),
        comment="创建用户"
    )
    created_time = Column(
        Date, 
        server_default=func.current_date(),
        comment="创建时间，默认当前日期"
    )
    updated_by = Column(
        String(512),
        comment="修改用户"
    )
    updated_time = Column(
        Date, 
        server_default=func.current_date(),
        comment="修改时间，默认当前日期"
    )