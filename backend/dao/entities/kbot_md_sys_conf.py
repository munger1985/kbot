from enum import Enum
from sqlalchemy import String, Date
from sqlalchemy.dialects.oracle import NUMBER, VARCHAR2
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
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
    
    conf_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="配置项唯一标识，主键"
    )
    app_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    param_type: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="1-ServiceURL/2-RagControl/3-Reranker/4-Search/5-Feedback/6-DataProcess DataProcess=》图片转成文本模型，语音转成文本模型。"
    )
    param_name: Mapped[str | None] = mapped_column(
        String(256),
        comment="参数名称（如ImageToTextModel-图片转文本模型, SpeechToTextModel-语音转文本模型）"
    )
    param_value: Mapped[str | None] = mapped_column(
        String(256),
        comment="参数值"
    )
    status: Mapped[str | None] = mapped_column(
        VARCHAR2(1),
        comment="配置状态：Y-启用, N-禁用"
    )
    descs: Mapped[str | None] = mapped_column(
        String(512),
        comment="配置项详细描述"
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