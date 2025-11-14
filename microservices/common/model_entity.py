from sqlalchemy import String, Date, Numeric, CLOB, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import declared_attr

class Base(DeclarativeBase):
    """Base class for all database models."""
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Convert CamelCase to SNAKE_CASE"""
        name = cls.__name__
        return (
            ''.join(['_' + c.lower() if c.isupper() else c for c in name])
            .lstrip('_')
            .upper()
        )
    
class KbotMdModels(Base):
    """模型表"""
    
    model_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="模型唯一标识，主键")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="所属应用ID")
    display_name: Mapped[str | None] = mapped_column(String(256), comment="模型显示名称（用户友好名称）")
    model_name: Mapped[str] = mapped_column(String(256), comment="模型技术名称（如gpt-4、text-embedding-ada-002等）")
    category: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="模型类别枚举")
    provider: Mapped[str] = mapped_column(String(256), comment="模型提供商（如local, OpenAI, Azure, Anthropic等）")
    api_endpoint: Mapped[str | None] = mapped_column(String(256), comment="API端点URL（如https://api.openai.com/v1）")
    api_key: Mapped[str | None] = mapped_column(String(256), comment="API密钥（建议加密存储）")
    status: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="模型状态：1-启用, 0-禁用")
    model_params: Mapped[dict | None] = mapped_column(CLOB, comment="JSON格式的模型默认参数配置")
    descs: Mapped[str | None] = mapped_column(String(512), comment="模型详细描述")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="创建用户")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="创建时间")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="修改用户")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="修改时间")
    
    def __repr__(self):
        return f"KbotMdModels(model_id={self.model_id!r}, provider={self.provider!r})"
    