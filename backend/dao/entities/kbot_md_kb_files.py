from sqlalchemy import String, Date
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class KbotMdKbFiles(Base):
    """Knowledge base files entity for KBOT_MD_KB_FILES table."""
    
    file_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="文件唯一标识，主键"
    )
    app_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    kb_id: Mapped[int] = mapped_column(
        NUMBER(38, 0),
        nullable=False,
        comment="关联的知识库ID"
    )
    batch_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="关联的批次ID（可选）"
    )
    file_path: Mapped[str | None] = mapped_column(
        String(512),
        comment="文件完整存储路径，包含文件名"
    )
    file_name: Mapped[str | None] = mapped_column(
        String(256),
        comment="原始文件名（不含扩展名）"
    )
    file_ext: Mapped[str | None] = mapped_column(
        String(256),
        comment="文件扩展名"
    )
    status: Mapped[int | None] = mapped_column(
        NUMBER(2, 0),
        comment="文件状态枚举类型"
    )
    file_version: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="文件版本号，每次更新递增"
    )
    is_overwrite: Mapped[int] = mapped_column(
        NUMBER(1, 0),
        comment="是否覆盖：1-是,0-否"
    )
    security_level: Mapped[int | None] = mapped_column(
        NUMBER(1, 0),
        comment="文件安全等级枚举类型"
    )
    file_size: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="文件大小（字节）"
    )
    chunks_cnt: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="文件分块数量"
    )
    chunk_parser: Mapped[str | None] = mapped_column(
        String(4000),
        comment="数据chunk参数"
    )
    enable_summary: Mapped[int] = mapped_column(
        NUMBER(1, 0),
        comment="1-启用,0-不启用"
    )
    is_img2txt: Mapped[int] = mapped_column(
        NUMBER(1, 0),
        comment="是否把IMAGE转成文本:1-是,0-否"
    )
    is_table_head_fill: Mapped[int] = mapped_column(
        NUMBER(1, 0),
        comment="Table表头是否拼装：1-是,0-否"
    )
    biz_metadata: Mapped[str | None] = mapped_column(
        String(4000),
        comment="JSON格式的业务元数据(存储为字符串)"
    )
    process_priority: Mapped[int | None] = mapped_column(
        NUMBER(1, 0),
        comment="处理优先级枚举类型"
    )
    log_msg: Mapped[str | None] = mapped_column(
        String(1000),
        comment="处理日志信息"
    )
    created_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="文件上传用户"
    )
    created_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="上传时间，默认系统当前时间"
    )
    updated_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="最后修改用户"
    )
    updated_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="最后修改时间，默认系统当前时间"
    )
    approved_by: Mapped[str | None] = mapped_column(
        String(256),
        comment="审批用户"
    )
    approved_time: Mapped[Date | None] = mapped_column(
        Date,
        comment="审批时间"
    )
    approve_comments: Mapped[str | None] = mapped_column(
        String(1024),
        comment="审批意见"
    )