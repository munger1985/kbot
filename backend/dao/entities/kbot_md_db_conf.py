from enum import Enum
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.dialects.oracle import NUMBER, VARCHAR2
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from .base import Base

class DbType(str, Enum):
    """Database type enumeration."""
    ORACLEDB = "ORACLEDB"
    ADB = "ADB"
    HEATWAVE = "HEATWAVE"
    ELASTICSEARCH = "ELASTICSEARCH"
    MILVUS = "MILVUS"
    FAISS = "FAISS"
    PINECONE = "PINECONE"
    WEAVIATE = "WEAVIATE"

class DbStatus(str, Enum):
    """Database configuration status enumeration."""
    ENABLED = "Y"
    DISABLED = "N"

class KbotMdDbConf(Base):
    """Database configuration entity for KBOT_MD_DB_CONF table."""
    
    __tablename__ = "KBOT_MD_DB_CONF"
    
    db_id = Column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="数据库配置唯一标识，主键"
    )
    app_id = Column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    db_display_name = Column(
        String(256),
        comment="数据库显示名称（用户友好名称）"
    )
    db_type = Column(
        String(20),
        comment="数据库类型：ORACLEDB-Oracle数据库, ADB-Oracle自治数据库, HEATWAVE-MySQL HeatWave, ELASTICSEARCH, MILVUS, FAISS, PINECONE, WEAVIATE等"
    )
    db_conn_str = Column(
        JSONB,
        comment="JSON格式的数据库连接字符串，包含主机、端口、认证等信息"
    )
    status = Column(
        VARCHAR2(1),
        comment="配置状态：Y-启用, N-禁用"
    )
    descs = Column(
        String(512),
        comment="连接配置详细描述"
    )
    created_by = Column(
        String(512),
        comment="创建用户"
    )
    created_time = Column(
        Date, 
        server_default=func.current_date(),
        comment="创建时间，默认系统当前时间"
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