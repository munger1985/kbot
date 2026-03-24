from sqlalchemy import String, Date, Numeric, func, JSON
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity

class AIModelEntity(BaseEntity):
    """Table for Model configuration (Model-driven).

    This entity maps to the database table `kbot_md_models` and stores core configuration
    for AI models, including identification, provider info, API settings, parameters,
    status control, and audit timestamps for model-driven AI applications.
    """

    __tablename__ = "kbot_md_models"
    
    model_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Unique model identifier, primary key")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated application ID")
    display_name: Mapped[str | None] = mapped_column(String(256), comment="Model display name (user-friendly name)")
    model_name: Mapped[str] = mapped_column(String(256), comment="Model technical name (e.g., gpt-4, text-embedding-ada-002)")
    category: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="Model category enumeration")
    provider: Mapped[str] = mapped_column(String(256), comment="Model provider (e.g., local, OpenAI, Azure, Anthropic)")
    api_endpoint: Mapped[str | None] = mapped_column(String(256), comment="API endpoint URL (e.g., https://api.openai.com/v1)")
    api_key: Mapped[str | None] = mapped_column(String(256), comment="API key (recommended to store encrypted)")
    status: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Model status: 1 - Enabled, 0 - Disabled")
    model_params: Mapped[dict | None] = mapped_column(JSON, comment="Default model parameters configuration in JSON format")
    descs: Mapped[str | None] = mapped_column(String(512), comment="Detailed model description")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")