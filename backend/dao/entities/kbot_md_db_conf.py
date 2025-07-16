from sqlalchemy import String, Date
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class KbotMdDbConf(Base):
    """Database configuration entity for KBOT_MD_DB_CONF table."""
    
    db_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="数据库配置唯一标识，主键"
    )
    app_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    db_display_name: Mapped[str | None] = mapped_column(
        String(256),
        comment="数据库显示名称（用户友好名称）"
    )
    
    db_type: Mapped[int | None] = mapped_column(
        NUMBER(2, 0), 
        comment="数据库类型枚举"
    )
    db_conn_str: Mapped[dict | None] = mapped_column(
        String(4000),
        comment="JSON格式的数据库连接字符串，包含主机、端口、认证等信息"
    )
    status: Mapped[int | None] = mapped_column(
        NUMBER(1, 0), 
        comment="配置状态：1-启用, 0-禁用"
    )
    descs: Mapped[str | None] = mapped_column(
        String(512),
        comment="连接配置详细描述"
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