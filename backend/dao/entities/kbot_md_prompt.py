from enum import Enum
from sqlalchemy import Column, Integer, String, Date, CLOB
from sqlalchemy.dialects.oracle import NUMBER, VARCHAR2
from sqlalchemy.sql import func
from sqlalchemy import Sequence
from .base import Base

class KbotMdPrompt(Base):
    """Prompt entity for KBOT_MD_PROMPT table."""
    
    __tablename__ = "KBOT_MD_PROMPT"
    
    prompt_id = Column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="提示词唯一标识，主键"
    )
    app_id = Column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    domain_id = Column(
        NUMBER(38, 0),
        comment="关联的业务域ID（可选）"
    )
    name = Column(
        String(256),
        comment="提示词名称"
    )
    prompt_category = Column(
        NUMBER(2, 0), 
        nullable=False,
        comment="提示词类型：1-系统提示词；2-知识库提示词模版 3-Agent提示词"
    )
    template = Column(
        CLOB,
        comment="提示词模板内容（CLOB大文本）"
    )
    status = Column(
        VARCHAR2(1),
        comment="提示词状态：Y-启用, N-禁用"
    )
    descs = Column(
        String(512),
        comment="提示词详细描述"
    )
    created_by = Column(
        String(512),
        comment="创建用户"
    )
    created_time = Column(
        Date, 
        server_default=func.current_date(),
        comment="创建时间，默认系统当前时间"
    )
    updated_by = Column(
        String(512),
        comment="最后修改用户"
    )
    updated_time = Column(
        Date, 
        server_default=func.current_date(),
        comment="最后修改时间，默认系统当前时间"
    )