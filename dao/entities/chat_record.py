from sqlalchemy import SmallInteger, DateTime, func, Text, text, Numeric, CHAR
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from .base import BaseEntity, ArrayField, VectorField

class ChatRecordEntity(BaseEntity):
    """Stores Chat Q&A interaction data between users and AI agents.
    This entity maps to `kbot_md_chat_record` and records core Q&A data including questions, answers,
    embeddings, feedback, and timestamps, with foreign keys to chat sessions and AI agents.
    """
    __tablename__ = "kbot_md_chat_record"
    
    record_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Auto-increment primary key")
    session_id: Mapped[str] = mapped_column(CHAR(36), nullable=False, comment="Associated chat session UUID (foreign key to kbot_md_chat_session.session_id)")
    question: Mapped[str | None] = mapped_column(Text, comment="User question content (CLOB)")
    answer: Mapped[str | None] = mapped_column(Text, comment="AI answer content (CLOB)")
    question_vector: Mapped[dict | None] = mapped_column(VectorField(), comment="Question embedding vector")
    references: Mapped[dict | None] = mapped_column(ArrayField(), comment="List of retrieved knowledge chunks")
    feedback: Mapped[int] = mapped_column(SmallInteger, default=0, comment="User feedback: 0 - Unrated, 1 - Positive, -1 - Negative")
    request_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="Request initiation timestamp (UTC)")
    response_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="Response completion timestamp (UTC)")