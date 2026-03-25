
from datetime import datetime
from typing import Any
from sqlalchemy import String, Integer, CLOB, JSON, DateTime, ForeignKey, Numeric, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from .base import BaseEntity, VectorField


class UserProfileEntity(BaseEntity):
    """
    Stores long-term user characteristics and technical preferences across sessions.
    """
    __tablename__ = "kbot_md_user_profile"

    user_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment="Unique identifier for the user")
    global_preferences: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Persistent technical stack preferences")
    frequent_entities: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Frequently occurring entities (e.g., project names, IPs)")
    profile_summary: Mapped[str | None] = mapped_column(CLOB, comment="LLM-generated user behavior summary")
    last_update_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationship to sessions
    contexts: Mapped[list["ConversationContextEntity"]] = relationship("ConversationContextEntity", back_populates="user")

class ConversationContextEntity(BaseEntity):
    """
    Manages active session states and rolling conversation summaries.
    """
    __tablename__ = "kbot_md_conv_context"

    session_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment="Unique session UUID")
    user_id: Mapped[str] = mapped_column(ForeignKey("kbot_md_user_profile.user_id"), nullable=False)
    session_title: Mapped[str | None] = mapped_column(String(256), comment="Session title")
    app_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Associated application ID")
    agent_id: Mapped[str | None] = mapped_column(String(256), comment="Associated AI Agent identifier")
    session_state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Current active state machine parameters")
    context_summary: Mapped[str | None] = mapped_column(CLOB, comment="Short-to-medium term rolling summary")
    interaction_count: Mapped[int] = mapped_column(Integer, default=0, comment="Total turns in this session")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["UserProfileEntity"] = relationship("UserProfileEntity", back_populates="contexts")
    entries: Mapped[list["MemoryEntryEntity"]] = relationship("MemoryEntryEntity", back_populates="context")

class MemoryEntryEntity(BaseEntity):
    """
    Records atomic Q&A interactions with semantic vectors and state snapshots.
    """
    __tablename__ = "kbot_md_memory_entry"

    entry_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("kbot_md_conv_context.session_id"), nullable=False)
    
    # Core RAG Fields
    standalone_query: Mapped[str | None] = mapped_column(CLOB, comment="Context-enriched rewritten question")
    search_keywords: Mapped[str | None] = mapped_column(String(1000), comment="Includes extracted keywords and expanded synonyms in hybrid search")
    memory_vector: Mapped[list[float]] = mapped_column(VectorField(), comment="Oracle 23ai native vector for semantic search")
    
    # Metadata & Content
    turn_entities: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Entity snapshot for this specific turn")
    raw_question: Mapped[str] = mapped_column(CLOB, comment="Original user input")
    answer: Mapped[str | None] = mapped_column(CLOB, comment="AI generated response")
    retrieved_chunks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=True, comment="References used for this answer")
    intent_category: Mapped[str | None] = mapped_column(String(64), comment="Intent category for this turn")
    feedback: Mapped[int] = mapped_column(Numeric(1, 0), default=0, comment="User feedback for this turn, -1: bad, 0: neutral, 1: good")
    request_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    response_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    context: Mapped["ConversationContextEntity"] = relationship("ConversationContextEntity", back_populates="entries")