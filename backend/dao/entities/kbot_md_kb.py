from sqlalchemy import String, Date
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.sql import func
from sqlalchemy import UniqueConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class KbotMdKb(Base):
    """Knowledge base entity for KBOT_MD_KB table."""

    __table_args__ = (
        UniqueConstraint('app_id', 'domain_id', 'kb_name'),
    )
    
    kb_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="知识库唯一标识，主键"
    )
    app_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID，与DOMAIN_ID和KB_NAME组成联合唯一约束"
    )
    domain_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        ForeignKey("KBOT_MD_DOMAIN.domain_id"),
        comment="关联的业务域ID，外键引用KBOT_KB_DOMAIN表"
    )
    kb_name: Mapped[str | None] = mapped_column(
        String(256),
        comment="知识库名称，在同一业务域下具有唯一性"
    )
    kb_category: Mapped[int | None] = mapped_column(
        NUMBER(2, 0),
        comment="知识库类型枚举类型"
    )
    descs: Mapped[str | None] = mapped_column(
        String(512),
        comment="知识库详细描述信息"
    )
    db_conn_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="关联的向量数据库连接配置ID"
    )
    txt_embed_model_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="文本嵌入模型ID"
    )
    img_embed_model_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="图片嵌入模型ID"
    )
    summary_model_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="摘要模型ID"
    )
    img2txt_model_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="图片转文本模型ID"
    )
    kb_status: Mapped[int | None] = mapped_column(
        NUMBER(1, 0),
        comment="知识库状态枚举类型"
    )
    security_level: Mapped[int | None] = mapped_column(
        NUMBER(1, 0),
        comment="文件安全等级枚举类型"
    )
    chunk_parser: Mapped[int | None] = mapped_column(
        String(4000),
        comment="数据Chunk参数"
    )
    enable_summary: Mapped[int | None] = mapped_column(
        NUMBER(1, 0),
        comment="1-启用,0-不启用"
    )
    is_img2txt: Mapped[int | None] = mapped_column(
        NUMBER(1, 0),
        comment="是否把IMAGE转成文本:1-是,0-否"
    )
    is_table_head_fill: Mapped[int | None] = mapped_column(
        NUMBER(1, 0),
        comment="Table表头是否拼装：1-是,0-否"
    )
    process_priority: Mapped[int | None] = mapped_column(
        NUMBER(1, 0),
        comment="处理优先级枚举类型"
    )
    created_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="记录创建人"
    )
    created_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="记录创建时间，默认系统当前时间"
    )
    updated_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="最后修改人"
    )
    updated_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="最后修改时间，默认系统当前时间"
    )