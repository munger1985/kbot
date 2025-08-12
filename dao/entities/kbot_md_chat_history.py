from sqlalchemy import String, Date, Numeric, CLOB
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class KbotMdChatHistory(Base):
    """聊天历史记录表"""
    
    his_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="数据库配置唯一标识，主键")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="所属应用ID")
    session_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="所属应用ID")
    agent_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="所属应用ID")
    question: Mapped[str | None] = mapped_column( String(4000), comment="数据库显示名称（用户友好名称）")
    answer: Mapped[str | None] = mapped_column(CLOB, comment="数据库类型枚举")
    created_by: Mapped[str | None] = mapped_column(String(512), comment="创建用户")
    created_time: Mapped[Date] = mapped_column(Date, comment="创建时间，默认系统当前时间")
    updated_by: Mapped[str | None] = mapped_column(String(512), comment="最后修改用户")
    updated_time: Mapped[Date] = mapped_column(Date, comment="最后修改时间，默认系统当前时间")

    def __repr__(self):
        return f"KbotMdDbConf(session_id={self.session_id!r})"