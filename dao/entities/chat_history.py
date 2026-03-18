from sqlalchemy import String, Date, Numeric, CLOB, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity

class ChatHistoryEntity(BaseEntity):
    """Table for Model-driven Chat History records.

    This entity maps to the database table `kbot_md_chat_history` and stores
    full chat interaction records between users and AI agents, including questions,
    answers, and association information with apps/sessions/agents.
    """

    __tablename__ = "kbot_md_chat_history"
    
    his_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Unique identifier for chat history record, primary key")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated application ID")
    session_id: Mapped[str] = mapped_column(String(256), nullable=False, comment="Associated session ID")
    agent_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated AI agent ID")
    question: Mapped[str | None] = mapped_column(String(4000), comment="User's question content")
    answer: Mapped[str | None] = mapped_column(CLOB, comment="AI agent's answer content")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")
    