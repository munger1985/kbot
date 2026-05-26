from sqlalchemy import String, Date, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity


class AgentConfEntity(BaseEntity):
    """Configuration table for AI agent (Model-driven).

    This entity maps to the database table `kbot_md_agent_conf` and stores core
    configuration parameters for model-driven AI agents, including associations
    with apps, tools, search rules, and audit timestamps.
    """

    __tablename__ = "kbot_md_agent_conf"

    conf_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Primary key ID")
    app_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Associated APP_ID")
    agent_id: Mapped[int] = mapped_column(Numeric(38, 0), comment="AI agent ID")
    tool_id: Mapped[int] = mapped_column(Numeric(38, 0), comment="Knowledge base ID / FUNC_ID, etc.")
    @property
    def kb_id(self) -> int:
        return self.tool_id
    tool_type: Mapped[int] = mapped_column(Numeric(2, 0), comment="Enumeration type")
    tool_weight: Mapped[float | None] = mapped_column(Numeric(2, 2), comment="Knowledge base retrieval weight")
    reranker_flag: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Whether reranking is required")
    search_type: Mapped[int] = mapped_column(Numeric(2, 0), comment="Search type enumeration")
    search_topk: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Number of search TOP-K results")
    search_score_threshold: Mapped[float | None] = mapped_column(Numeric(38, 0), comment="Search similarity threshold")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")