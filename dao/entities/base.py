import json
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TypeDecorator, JSON, Text, Dialect
from sqlalchemy import UnicodeText
from sqlalchemy.dialects.postgresql import JSONB
# from sqlalchemy.dialects.oracle import RAW as ORA_RAW
from sqlalchemy.dialects.oracle import VECTOR as ORA_VECTOR  # Oracle 23ai+
# from pgvector.sqlalchemy import Vector as PG_VECTOR
from sqlalchemy.ext.mutable import MutableList

BaseEntity = declarative_base()

class OracleJSON(TypeDecorator):
    """
    针对 Oracle 异步驱动定制的 JSON 类型
    解决 AttributeError: 'OracleDialectAsync_oracledb' object has no attribute '_json_deserializer'
    """
    impl = UnicodeText
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value)

class UniversalArray(TypeDecorator):
    """
    针对现代数据库优化的 JSON 数组/对象字段
    - PostgreSQL 18: 使用原生 JSONB (二进制存储，支持索引)
    - Oracle 26ai: 使用原生 JSON 类型 (OSON 格式，高性能)
    """
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            # PG 18 依然推荐 JSONB 进行高效索引
            return dialect.type_descriptor(JSONB())
        elif dialect.name == 'oracle':
            # Oracle 23ai/26ai 原生 JSON 类型
            return dialect.type_descriptor(JSON())
        else:
            # 兜底使用标准 JSON
            return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        return value

def ArrayField():
    """
    赋予 JSON 字段感知对象内部修改（如 .append()）的能力
    """
    return MutableList.as_mutable(UniversalArray())


class UniversalVector(TypeDecorator):
    """
    支持动态维度的跨库向量适配器
    """
    impl = Text # 默认降级实现
    cache_ok = True

    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim # dim 可以为 None，表示不限制维度的向量

    def load_dialect_impl(self, dialect: Dialect):
        # if dialect.name == 'postgresql':
        #     if PG_VECTOR is not None:
        #         # PG 支持不带维度的向量定义：VECTOR
        #         return dialect.type_descriptor(PG_VECTOR(self.dim))
        #     return dialect.type_descriptor(Text())
        if dialect.name == 'oracle':
            if ORA_VECTOR is not None:
                # Oracle 23ai+ 也支持不指定维度的向量定义
                return dialect.type_descriptor(ORA_VECTOR(self.dim) if self.dim else ORA_VECTOR())
            return dialect.type_descriptor(Text())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect: Dialect):
        if value is None: return None
        # 如果是 list，自动探测维度（可选逻辑）
        return value

    def process_result_value(self, value, dialect: Dialect):
        if value is None: return None
        return list(value) if not isinstance(value, list) else value

def VectorField(dim: int | None = None):
    """
    便捷工厂函数，定义指定维度的向量字段
    """
    return UniversalVector(dim=dim)