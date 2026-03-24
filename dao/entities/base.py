import json
from sqlalchemy import TypeDecorator, Text
import array as array_module
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TypeDecorator, Text, Dialect
from sqlalchemy.dialects.oracle import VECTOR as ORA_VECTOR  # Oracle 23ai+

BaseEntity = declarative_base()

        
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
        # 对于 Oracle VECTOR 类型，需要转换为数组格式
        if dialect.name == 'oracle' and isinstance(value, list):
            return array_module.array('f', value)
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