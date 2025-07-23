from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from core.database.meta_oracle import Base


class KBotMdAgentConf(Base):
    """KBOT_MD_AGENT_CONF表的ORM模型"""

    __tablename__ = "KBOT_MD_AGENT_CONF"

    conf_id: Mapped[int] = mapped_column(
        primary_key=True,
        comment="主键ID",
        server_default=text("KBOT.ISEQ$$_72398.nextval")
    )
    app_id: Mapped[Optional[int]] = mapped_column(
        comment="所属APP_ID"
    )
    agent_id: Mapped[Optional[int]] = mapped_column(
        comment="智能体ID"
    )
    tool_id: Mapped[Optional[int]] = mapped_column(
        comment="知识库ID/FUNC_ID等"
    )
    tool_type: Mapped[Optional[int]] = mapped_column(
        comment="枚举类型"
    )
    tool_weight: Mapped[Optional[int]] = mapped_column(
        comment="知识库检索权重"
    )
    reranker_flag: Mapped[Optional[int]] = mapped_column(
        comment="是否需要重排"
    )
    search_type: Mapped[Optional[int]] = mapped_column(
        comment="搜索类型枚举类型"
    )
    search_topk: Mapped[Optional[int]] = mapped_column(
        comment="搜索TOPK个数"
    )
    search_score_threshold: Mapped[Optional[int]] = mapped_column(
        comment="搜索相似度阈值"
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        comment="创建用户",
        server_default=text("USER")
    )
    created_time: Mapped[Optional[datetime]] = mapped_column(
        comment="创建时间",
        server_default=func.current_date()
    )
    updated_by: Mapped[Optional[str]] = mapped_column(
        comment="修改用户",
        server_default=text("USER")
    )
    updated_time: Mapped[Optional[datetime]] = mapped_column(
        comment="修改时间",
        server_default=func.current_date()
    )

    def __repr__(self):
        return f"KBotMdAgentConf(conf_id={self.conf_id!r}, agent_id={self.agent_id!r}, tool_id={self.tool_id!r}, tool_type={self.tool_type!r})"