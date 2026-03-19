from sqlalchemy import String, Date, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, OracleJSON


class AgentEntity(BaseEntity):
    """Table for Model-driven AI Agent configuration.

    This entity maps to the database table `kbot_md_agent` and stores core
    configuration information for model-driven AI agents, including basic info,
    LLM parameters, similarity settings, reranking rules, and audit timestamps.
    """

    __tablename__ = "kbot_md_agent"

    agent_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Primary key ID")
    app_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Associated app_id")
    domain_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Associated domain ID")
    agent_name: Mapped[str | None] = mapped_column(String(256), comment="AI agent name")
    agent_desc: Mapped[str | None] = mapped_column(String(512), comment="AI agent description")  # Fixed typo: agnet_desc → agent_desc
    welcome: Mapped[str | None] = mapped_column(String(512), comment="Welcome message")
    prompt_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Prompt ID")
    llm_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Large language model ID")
    llm_params: Mapped[dict | None] = mapped_column(OracleJSON, comment="LLM parameter configuration (JSON format)")
    feedback_similarity_flag: Mapped[int | None] = mapped_column(Numeric(1, 0), default=0, comment="Feedback similarity switch")
    synonym_similarity_flag: Mapped[int | None] = mapped_column(Numeric(1, 0), default=0, comment="Synonym similarity switch")
    reranker_model_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Reranker model ID")
    reranker_topk: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Number of reranked results to retrieve")
    reranker_score_threshold: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Reranker score threshold")
    agent_status: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="Status: Enumeration type")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")