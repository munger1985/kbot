from sqlalchemy import Numeric, String, Date, JSON, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity


class ParserConfEntity(BaseEntity):
    __tablename__ = "kbot_md_parser_conf"
    
    parser_conf_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, autoincrement=True, comment="自增ID，主键")
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), comment="Domain ID, referencing domain table")
    engine: Mapped[str] = mapped_column(String(10), comment="Parser engine：Enum：ParserEngine")
    parser_params: Mapped[dict] = mapped_column(JSON, comment="Parser parameters")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")

    def to_dict(self) -> dict:
        """将 ORM 对象转换为字典，用于API响应"""
        return {
            "parser_conf_id": self.parser_conf_id,
            "domain_id": self.domain_id,
            "engine": self.engine,
            "parser_params": self.parser_params,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }