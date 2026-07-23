import json
from datetime import timezone
from decimal import Decimal
from uuid import UUID

from sqlalchemy import DateTime, Dialect, LargeBinary, Text, TypeDecorator
import array as array_module
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.oracle import (
    RAW,
    TIMESTAMP as ORA_TIMESTAMP,
    VECTOR as ORA_VECTOR,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID


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


class OracleJSON(TypeDecorator):
    """
    自适应 Oracle JSON 处理器。
    完美调和 oracledb 驱动底层自动反序列化 OSON 与 SQLAlchemy 二次解析带来的冲突。
    """
    # 🎯 核心修正 1：避开原生 JSON 类型的深度拦截，由我们完全接管处理流程
    impl = Text  
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect):
        """根据当前数据库方言动态加载底层实现类型"""
        # 🎯 始终使用 Text 作为底层实现，由 process_bind_param/process_result_value 完全接管序列化/反序列化
        # 避免使用原生 JSON() 类型，因为在 oracledb async 模式下会触发
        # 'OracleDialectAsync_oracledb' object has no attribute '_json_deserializer' 错误
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect: Dialect):
        """序列化：将 Python 对象转换为 JSON 字符串或保持原样供驱动处理"""
        if value is None:
            return None
        
        # 兼容高精度 Decimal 并防止中文被转义为 \uXXXX
        def default_encoder(item):
            if isinstance(item, Decimal):
                return float(item)
            if isinstance(item, UUID):
                return str(item)
            raise TypeError(f"Object of type {item.__class__.__name__} is not JSON serializable")
            
        # 即使底层是原生 JSON 类型，直接传纯净的 JSON 字符串也是最安全的写入方式
        return json.dumps(value, default=default_encoder, ensure_ascii=False)

    def process_result_value(self, value, dialect: Dialect):
        """反序列化：如果驱动已经解构成 dict/list，则直接放行，杜绝 TypeError"""
        if value is None:
            return None
            
        # 🎯 核心防御：如果 oracledb 驱动在 thin 异步模式下已经自动反序列化成了字典/列表，直接放行
        if isinstance(value, (dict, list)):
            return value
            
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return value
                
        return value
