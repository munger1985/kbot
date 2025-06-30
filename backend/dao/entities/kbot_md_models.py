from enum import Enum
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.dialects.oracle import NUMBER, VARCHAR2
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from .base import Base

class KbotMdKbModels(Base):
    """Knowledge base models entity for KBOT_MD_MODELS table."""
    
    __tablename__ = "KBOT_MD_MODELS"
    
    model_id = Column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="模型唯一标识，主键"
    )
    app_id = Column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    display_name = Column(
        String(256),
        comment="模型显示名称（用户友好名称）"
    )
    model_name = Column(
        String(256),
        comment="模型技术名称（如gpt-4、text-embedding-ada-002等）"
    )
    category = Column(
        NUMBER(2, 0), 
        comment="模型类别：1-LLM-大语言模型, 2-EMBEDDING-嵌入模型, 3-RERANKER-重排序模型, 4-VLM-视觉语言模型"
    )
    provider = Column(
        String(256),
        comment="模型提供商（如OpenAI、Azure、Anthropic等）"
    )
    api_endpoint = Column(
        String(256),
        comment="API端点URL（如https://api.openai.com/v1）"
    )
    api_key = Column(
        String(256),
        comment="API密钥（建议加密存储）"
    )
    status = Column(
        VARCHAR2(1),
        comment="模型状态：Y-启用, N-禁用"
    )
    model_params = Column(
        JSONB,
        comment="JSON格式的模型默认参数配置，如{\"TEMPERATURE\":0,\"MAX_TOKENS\":1024}"
    )
    model_template = Column(
        JSONB,
        comment="JSON格式的模型模板配置"
    )
    descs = Column(
        String(512),
        comment="模型详细描述"
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