from sqlalchemy import String, Date, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity

class ChatSessionEntity(BaseEntity):
    """Table for Chat Session management (Model-driven).

    This entity maps to the database table `kbot_md_chat_session` and stores session
    information for conversations between users and AI agents, including session title,
    associated application/agent/user, and audit timestamps.
    """

    __tablename__ = "kbot_md_chat_session"

    session_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment="Unique session identifier, primary key")
    session_title: Mapped[str | None] = mapped_column(String(256), comment="Session title")
    app_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Associated application ID")
    agent_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Associated AI agent ID")
    appuser: Mapped[str | None] = mapped_column(String(256), comment="User identifier (login username for user questions, 'ai' for AI answers)")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Last updater")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Last update time (default: current time)")
