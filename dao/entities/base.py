import json
from sqlalchemy import TypeDecorator, Text
import array as array_module
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TypeDecorator, Text, Dialect
from sqlalchemy.dialects.oracle import VECTOR as ORA_VECTOR  # Oracle 23ai+
from loguru import logger

BaseEntity = declarative_base()



class OracleJSON(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        # 1. 处理 None 或空字典：直接存为 NULL
        # 很多时候 Oracle 报 JZN-00085 就是因为无法正确处理传入的 "{}" 字符串
        if value is None or (isinstance(value, dict) and not value):
            return None

        try:
            # 2. 如果已经是字符串，校验并去除首尾空格
            if isinstance(value, str):
                json.loads(value) # 验证合法性
                return value.strip()
            
            # 3. 序列化对象，确保不带任何多余的空白
            # 尝试使用 ensure_ascii=False，让 Oracle 处理原生的 UTF-8 字符
            return json.dumps(value, ensure_ascii=False).strip()
        except Exception:
            # 兜底：如果报错，返回 None 而不是空的 {}
            return None

    def process_result_value(self, value, dialect):
        if value is None:
            return {} # 读取时回填为空字典
        try:
            return json.loads(value)
        except:
            return value
        
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