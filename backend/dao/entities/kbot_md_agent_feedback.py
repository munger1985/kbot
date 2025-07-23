from datetime import datetime
from typing import Optional
from sqlalchemy import String, Date
from sqlalchemy.dialects.oracle import CLOB, NUMBER
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class KbotMdAgentFeedback(Base):
    """Agent feedback entity model"""
    
    fb_id: Mapped[int] = mapped_column(
        "FB_ID", 
        NUMBER, 
        primary_key=True, 
        server_default="KBOT.ISEQ$$_72400.nextval"
    )
    app_id: Mapped[Optional[int]] = mapped_column("APP_ID", NUMBER)
    feedback_type: Mapped[Optional[int]] = mapped_column("FEEDBACK_TYPE", NUMBER)
    feedback_reason: Mapped[Optional[str]] = mapped_column("FEEDBACK_REASON", String(4000))
    question: Mapped[Optional[str]] = mapped_column("QUESTION", String(4000))
    question_vector: Mapped[Optional[str]] = mapped_column("QUESTION_VECTOR", String(4000))
    answer_disp: Mapped[Optional[str]] = mapped_column("ANSWER_DISP", CLOB)
    answer_json: Mapped[Optional[dict]] = mapped_column("ANSWER_JSON", String(4000))
    agent_id: Mapped[Optional[int]] = mapped_column("AGENT_ID", NUMBER)
    created_by: Mapped[Optional[str]] = mapped_column("CREATED_BY", String(256))
    created_time: Mapped[Optional[datetime]] = mapped_column(
        "CREATED_TIME", 
        Date, 
        server_default="CURRENT_DATE"
    )
    updated_by: Mapped[Optional[str]] = mapped_column("UPDATED_BY", String(256))
    updated_time: Mapped[Optional[datetime]] = mapped_column(
        "UPDATED_TIME", 
        Date, 
        server_default="CURRENT_DATE"
    )

    def __repr__(self):
        return f"KbotMdAgentFeedback(fb_id={self.fb_id}, app_id={self.app_id}, feedback_type={self.feedback_type})"