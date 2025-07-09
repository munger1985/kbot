from sqlalchemy import String, Date
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


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
        NUMBER(2, 0),
        comment="参数类型枚举"
    )
    param_name: Mapped[str | None] = mapped_column(
        String(256),
        comment="参数名称（如ImageToTextModel-图片转文本模型, SpeechToTextModel-语音转文本模型）"
    )
    param_value: Mapped[str | None] = mapped_column(
        String(256),
        comment="参数值"
    )
    status: Mapped[int | None] = mapped_column(
        NUMBER(1, 0),
        comment="配置状态：1-启用, 0-禁用"
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