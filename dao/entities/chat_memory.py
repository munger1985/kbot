from sqlalchemy import SmallInteger, DateTime, func, CLOB, text, Numeric, CHAR, String
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from .base import BaseEntity, ArrayField, VectorField

class ChatMemoryEntity(BaseEntity):
    """Stores Chat Q&A interaction data between users and AI agents.
    This entity maps to `kbot_md_chat_memory` and records core Q&A data including questions, answers,
    embeddings, feedback, and timestamps, with foreign keys to chat sessions and AI agents.
    """
    __tablename__ = "kbot_md_chat_memory"
    
    memory_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Auto-increment primary key")
    session_id: Mapped[str] = mapped_column(CHAR(36), nullable=False, comment="Associated chat session UUID (foreign key to kbot_md_chat_session.session_id)")
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, comment="Record user id or user name")
    question: Mapped[str] = mapped_column(CLOB, comment="User question content (CLOB)")
    answer: Mapped[str | None] = mapped_column(CLOB, comment="AI answer content (CLOB)")
    question_vector: Mapped[list[float]]= mapped_column(VectorField(), comment="Question embedding vector")
    references: Mapped[list[dict]] = mapped_column(ArrayField(), comment="List of retrieved knowledge chunks")
    feedback: Mapped[int] = mapped_column(SmallInteger, default=0, comment="User feedback: 0 - Unrated, 1 - Positive, -1 - Negative")
    request_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="Request initiation timestamp (UTC)")
    response_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="Response completion timestamp (UTC)")