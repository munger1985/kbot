from sqlalchemy import String, Date, Numeric, CLOB
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class KBotMdAgent(Base):
    """智能体表"""
    
    agent_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="主键id")
    app_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="所属app_id")
    domain_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="所属domain")
    agent_name: Mapped[str | None] = mapped_column(String(256), comment="智能体名称")
    agnet_desc: Mapped[str | None] = mapped_column(String(512), comment="智能体描述")
    welcome: Mapped[str | None] = mapped_column(String(512), comment="开场白")
    prompt_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="prompt id")
    llm_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="大语言模型id")
    llm_params: Mapped[dict | None] = mapped_column(CLOB, comment="llm参数配置(json格式)")
    feedback_similarity_flag: Mapped[int | None] = mapped_column(Numeric(1, 0), default=0, comment="反馈相似度开关")
    synonym_similarity_flag: Mapped[int | None] = mapped_column(Numeric(1, 0), default=0, comment="同义词相似度开关")
    reranker_model_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="重排分模型id")
    reranker_topk: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="重排分取数个数")
    reranker_score_threshold: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="重排分数阈值")
    agent_status: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="状态：枚举")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="创建用户")
    created_time: Mapped[Date] = mapped_column(Date, comment="创建时间")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="修改用户")
    updated_time: Mapped[Date] = mapped_column(Date, comment="修改时间")

    def __repr__(self):
        return f"KBotMdAgent(agent_id={self.agent_id!r},agent_name={self.agent_name!r})"