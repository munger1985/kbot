import json
from datetime import timezone
from decimal import Decimal
from uuid import UUID

from sqlalchemy import DateTime, Dialect, LargeBinary, Text, TypeDecorator
from sqlalchemy.types import UserDefinedType
import array as array_module
from sqlalchemy.dialects.oracle import (
    RAW,
    TIMESTAMP as ORA_TIMESTAMP,
    VECTOR as ORA_VECTOR,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base


BaseEntity = declarative_base()


class UUIDv7Type(TypeDecorator):
    """Oracle RAW(16) / PostgreSQL UUID 的统一映射。"""

    impl = LargeBinary(16)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect):
        if dialect.name == "oracle":
            return dialect.type_descriptor(RAW(16))
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(LargeBinary(16))

    def process_bind_param(self, value, dialect: Dialect):
        if value is None:
            return None
        parsed = value if isinstance(value, UUID) else UUID(str(value))
        if dialect.name == "postgresql":
            return parsed
        return parsed.bytes

    def process_result_value(self, value, dialect: Dialect):
        if value is None or isinstance(value, UUID):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            return UUID(bytes=bytes(value))
        return UUID(str(value))


class UniversalTimestamp(TypeDecorator):
    """Oracle TIMESTAMP / PostgreSQL TIMESTAMP 的统一映射。"""

    impl = DateTime
    cache_ok = True

    def __init__(self, *, timezone: bool = True):
        super().__init__(timezone=timezone)
        self.timezone = timezone

    def load_dialect_impl(self, dialect: Dialect):
        if dialect.name == "oracle":
            return dialect.type_descriptor(
                ORA_TIMESTAMP(timezone=self.timezone)
            )
        return dialect.type_descriptor(DateTime(timezone=self.timezone))

    def process_bind_param(self, value, dialect: Dialect):
        if value is None or not self.timezone:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("带时区时间字段必须传入 aware datetime")
        normalized = value.astimezone(timezone.utc)
        if dialect.name == "oracle":
            # Oracle Thin 以 Session 时区解释 naive datetime。
            return normalized.replace(tzinfo=None)
        return normalized

    def process_result_value(self, value, dialect: Dialect):
        if value is None or not self.timezone:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            # Oracle Session 已由 DatabaseRuntime 固定为 +00:00。
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class UniversalVector(TypeDecorator):
    """
    支持动态维度的跨库向量适配器
    """
    impl = Text # 默认降级实现
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect):
        # if dialect.name == 'postgresql':
        #     if PG_VECTOR is not None:
        #         # PG 支持不带维度的向量定义：VECTOR
        #         return dialect.type_descriptor(PG_VECTOR(self.dim))
        #     return dialect.type_descriptor(Text())
        if dialect.name == 'oracle':
            if ORA_VECTOR is not None:
                # Oracle 23ai+ 也支持不指定维度的向量定义
                return dialect.type_descriptor(ORA_VECTOR())
            return dialect.type_descriptor(Text())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect: Dialect):
        if value is None: return None
        # 对于 Oracle VECTOR 类型，需要转换为数组格式
        if dialect.name == 'oracle' and isinstance(value, list):
            return array_module.array('f', value)
        # 如果是 list，自动探测维度（可选逻辑）
        return value

    def process_result_value(self, value, dialect: Dialect):
        if value is None: return None
        return list(value) if not isinstance(value, list) else value

def VectorField():
    """
    便捷工厂函数，定义指定维度的向量字段
    """
    return UniversalVector()


class OracleNativeJSON(UserDefinedType):
    """Oracle 26ai 原生 JSON 列映射。"""

    cache_ok = True

    def get_col_spec(self, **kwargs):
        return "JSON"

    def bind_processor(self, dialect):
        """将 Python 对象编码为 Oracle 可隐式转换的 JSON 文本。"""

        def process(value):
            if value is None:
                return None

            def default_encoder(item):
                if isinstance(item, Decimal):
                    return float(item)
                if isinstance(item, UUID):
                    return str(item)
                raise TypeError(
                    f"{item.__class__.__name__} 无法序列化为 JSON"
                )

            return json.dumps(
                value,
                default=default_encoder,
                ensure_ascii=False,
            )

        return process

    def result_processor(self, dialect, coltype):
        """兼容驱动返回原生对象或 JSON 文本两种形式。"""

        def normalize(item):
            if isinstance(item, Decimal):
                return (
                    int(item)
                    if item == item.to_integral_value()
                    else float(item)
                )
            if isinstance(item, dict):
                return {
                    key: normalize(nested)
                    for key, nested in item.items()
                }
            if isinstance(item, list):
                return [normalize(nested) for nested in item]
            return item

        def process(value):
            if value is None:
                return value
            if isinstance(value, str):
                value = json.loads(value)
            return normalize(value)

        return process
