from sqlalchemy import String, Date, Numeric, CLOB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


class KBotMdAgentConf(Base):
    """智能体配置表"""

    conf_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="主键ID")
    app_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="所属APP_ID")
    agent_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="智能体ID")
    tool_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="知识库ID/FUNC_ID等")
    tool_type: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="枚举类型")
    tool_weight: Mapped[int | None] = mapped_column(Numeric(2, 2), comment="知识库检索权重")
    reranker_flag: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="是否需要重排")
    search_type: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="搜索类型枚举类型")
    search_topk: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="搜索TOPK个数")
    search_score_threshold: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="搜索相似度阈值")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="创建用户")
    created_time: Mapped[Date] = mapped_column(Date, comment="创建时间")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="修改用户")
    updated_time: Mapped[Date] = mapped_column(Date, comment="修改时间")


    def __repr__(self):
        return f"KBotMdAgentConf(conf_id={self.conf_id!r}, agent_id={self.agent_id!r}, tool_id={self.tool_id!r}, tool_type={self.tool_type!r})"