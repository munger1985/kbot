from sqlalchemy import String, Date, CLOB, Numeric
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class KbotMdPrompt(Base):
    """Prompt entity for KBOT_MD_PROMPT table."""

    prompt_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="提示词唯一标识，主键")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="所属应用ID")
    domain_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="关联的业务域ID（可选）")
    name: Mapped[str | None] = mapped_column(String(256), comment="提示词名称")
    prompt_unique_name: Mapped[str] = mapped_column(String(256), comment="提示词在数据库中的唯一名字，用于程序中的模型调用")
    prompt_category: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="提示词枚举类型")
    template: Mapped[str | None] = mapped_column(CLOB, comment="提示词模板内容（CLOB大文本）")
    status: Mapped[int | None] = mapped_column(Numeric(1,0), comment="提示词状态：1-启用, 0-禁用")
    descs: Mapped[str | None] = mapped_column(String(512), comment="提示词详细描述")
    created_by: Mapped[str | None] = mapped_column(String(512), comment="创建用户")
    created_time: Mapped[Date] = mapped_column(Date, comment="创建时间，默认系统当前时间")
    updated_by: Mapped[str | None] = mapped_column(String(512), comment="最后修改用户")
    updated_time: Mapped[Date] = mapped_column(Date, comment="最后修改时间，默认系统当前时间")

    def __repr__(self):
        return f"KbotMdPrompt(prompt_id={self.prompt_id!r}, prompt_category={self.prompt_category!r}, prompt_unique_name={self.prompt_unique_name!r})"