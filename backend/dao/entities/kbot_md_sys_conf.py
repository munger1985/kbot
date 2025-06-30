from enum import Enum
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.dialects.oracle import NUMBER, VARCHAR2
from sqlalchemy.sql import func
from .base import Base

class ParamType(int, Enum):
    """Parameter type enumeration."""
    SERVICE_URL = 1
    RAG_CONTROL = 2
    RERANKER = 3
    SEARCH = 4
    FEEDBACK = 5
    DATA_PROCESS = 6

class KbotMdSysConf(Base):
    """System configuration entity for KBOT_MD_SYS_CONF table."""
    
    __tablename__ = "KBOT_MD_SYS_CONF"
    
    conf_id = Column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="配置项唯一标识，主键"
    )
    app_id = Column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    param_type = Column(
        NUMBER(38, 0),
        comment="1-ServiceURL/2-RagControl/3-Reranker/4-Search/5-Feedback/6-DataProcess DataProcess=》图片转成文本模型，语音转成文本模型。"
    )
    param_name = Column(
        String(256),
        comment="参数名称（如ImageToTextModel-图片转文本模型, SpeechToTextModel-语音转文本模型）"
    )
    param_value = Column(
        String(256),
        comment="参数值"
    )
    status = Column(
        NUMBER(1, 0),
        comment="配置状态：1-启用, 0-禁用"
    )
    descs = Column(
        String(512),
        comment="配置项详细描述"
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