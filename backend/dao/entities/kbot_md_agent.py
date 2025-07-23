from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, BigInteger, Date, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database.meta_oracle import Base


class KBotMdAgent(Base):
    """Agent metadata orm."""
    
    __tablename__ = "KBOT_MD_AGENT"
    
    AGENT_ID: Mapped[int] = mapped_column(BigInteger, primary_key=True, comment="主键ID")
    APP_ID: Mapped[Optional[int]] = mapped_column(BigInteger, comment="所属APP_ID")
    DOMAIN_ID: Mapped[Optional[int]] = mapped_column(BigInteger, comment="所属DOMAIN")
    AGENT_NAME: Mapped[Optional[str]] = mapped_column(String(256), comment="智能体名称")
    AGNET_DESC: Mapped[Optional[str]] = mapped_column(String(512), comment="智能体描述")
    WELCOME: Mapped[Optional[str]] = mapped_column(String(512), comment="开场白")
    PROMPT_ID: Mapped[Optional[int]] = mapped_column(BigInteger, comment="Prompt ID")
    LLM_ID: Mapped[Optional[int]] = mapped_column(BigInteger, comment="大语言模型ID")
    LLM_PARAMS: Mapped[Optional[dict]] = mapped_column(JSON, comment="LLM参数配置(JSON格式)")
    FEEDBACK_SIMILARITY_FLAG: Mapped[Optional[int]] = mapped_column(Integer, default=0, comment="反馈相似度开关")
    SYNONYM_SIMILARITY_FLAG: Mapped[Optional[int]] = mapped_column(Integer, default=0, comment="同义词相似度开关")
    RERANKER_MODEL_ID: Mapped[Optional[int]] = mapped_column(BigInteger, comment="重排分模型ID")
    RERANKER_TOPK: Mapped[Optional[int]] = mapped_column(BigInteger, comment="重排分取数个数")
    RERANKER_SCORE_THRESHOLD: Mapped[Optional[int]] = mapped_column(BigInteger, comment="重排分数阈值")
    AGENT_STATUS: Mapped[Optional[int]] = mapped_column(Integer, comment="状态: AgentStatus枚举")
    CREATED_BY: Mapped[Optional[str]] = mapped_column(String(256), comment="创建用户")
    CREATED_TIME: Mapped[Optional[datetime]] = mapped_column(Date, server_default="CURRENT_DATE", comment="创建时间")
    UPDATED_BY: Mapped[Optional[str]] = mapped_column(String(256), comment="修改用户")
    UPDATED_TIME: Mapped[Optional[datetime]] = mapped_column(Date, server_default="CURRENT_DATE", comment="修改时间")

    def __repr__(self):
        return f"KBotMdAgent(AGENT_ID={self.AGENT_ID!r}, APP_ID={self.APP_ID!r}, DOMAIN_ID={self.DOMAIN_ID!r}, AGENT_NAME={self.AGENT_NAME!r})"