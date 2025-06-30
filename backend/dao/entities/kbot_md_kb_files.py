from enum import Enum
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.dialects.oracle import NUMBER, VARCHAR2
from sqlalchemy.sql import func
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from .base import Base

class FileStatus(int, Enum):
    """File status enumeration."""
    UPLOADED = 1
    APPROVED = 2
    REJECTED = 3
    PARSING = 4
    PARSED = 5
    REPARSE = 6
    PARSE_FAILED = 7
    ARCHIVED = 8

class ProcessPriority(int, Enum):
    """Process priority enumeration."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class KbotMdKbFiles(Base):
    """Knowledge base files entity for KBOT_MD_KB_FILES table."""
    
    __tablename__ = "KBOT_MD_KB_FILES"
    
    file_id = Column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="文件唯一标识，主键"
    )
    app_id = Column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    kb_id = Column(
        NUMBER(38, 0),
        ForeignKey("KBOT_MD_KB.kb_id"),
        nullable=False,
        comment="关联的知识库ID"
    )
    batch_id = Column(
        NUMBER(38, 0),
        comment="关联的批次ID（可选）"
    )
    file_path = Column(
        String(512),
        comment="文件完整存储路径，包含文件名"
    )
    file_name = Column(
        String(256),
        comment="原始文件名（不含扩展名）"
    )
    file_ext = Column(
        String(256),
        comment="文件扩展名"
    )
    status = Column(
        NUMBER(38, 0),
        comment="文件状态：1-已上传,2-已审批,3-已拒绝,4-解析中,5-解析完成,6-重新解析,7-解析失败,8-已归档"
    )
    file_version = Column(
        NUMBER(38, 0),
        comment="文件版本号，每次更新递增"
    )
    is_overwrite = Column(
        VARCHAR2(1),
        server_default="Y",
        comment="是否覆盖：Y-是,N-否"
    )
    security_level = Column(
        String(256),
        comment="文件安全等级：高、中、低"
    )
    file_size = Column(
        NUMBER(38, 0),
        comment="文件大小（字节）"
    )
    chunks_cnt = Column(
        NUMBER(38, 0),
        comment="文件分块数量"
    )
    process_params = Column(
        JSONB,
        comment="JSON格式的处理参数配置"
    )
    enable_summary = Column(
        VARCHAR2(1),
        server_default="N",
        comment="是否启用摘要：Y-启用,N-不启用"
    )
    biz_metadata = Column(
        JSONB,
        comment="JSON格式的业务元数据"
    )
    process_priority = Column(
        NUMBER(38, 0),
        comment="处理优先级：1-低,2-中,3-高"
    )
    log_msg = Column(
        String(1000),
        comment="处理日志信息"
    )
    created_by = Column(
        String(512),
        comment="文件上传用户"
    )
    created_time = Column(
        Date, 
        server_default=func.current_date(),
        comment="上传时间，默认系统当前时间"
    )
    updated_by = Column(
        String(512),
        comment="最后修改用户"
    )
    updated_time = Column(
        Date, 
        server_default=func.current_date(),
        comment="最后修改时间，默认系统当前时间"
    )
    approved_by = Column(
        String(256),
        comment="审批用户"
    )
    approved_time = Column(
        Date,
        comment="审批时间"
    )
    approve_comments = Column(
        String(1024),
        comment="审批意见"
    )