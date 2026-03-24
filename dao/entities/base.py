import json
from sqlalchemy import TypeDecorator, Text
import array as array_module
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TypeDecorator, Text, Dialect
from sqlalchemy.dialects.oracle import VECTOR as ORA_VECTOR  # Oracle 23ai+
from loguru import logger

BaseEntity = declarative_base()



class OracleJSON(TypeDecorator):
    """
    专为 Oracle 异步驱动定制的 JSON 类型
    绕过驱动缺失的 _json_deserializer，并解决 ORA-40441 语法错误
    """
    # 如果 JSON 数据量大，建议用 CLOB；如果较小，用 Text
    impl = Text 
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """发送数据到数据库：对象 -> 字符串"""
        if value is None:
            return None
        
        # 核心修复：确保不进行双重序列化
        if isinstance(value, str):
            try:
                # 校验是否为合法JSON
                json.loads(value)
                return value.strip()
            except json.JSONDecodeError:
                # 如果是纯字符串而非JSON，包装成字符串对象
                return json.dumps(value, ensure_ascii=False).strip()
        
        # 序列化字典或列表
        # 使用 ensure_ascii=False 减少转义开销，但需确保数据库字符集支持 UTF8
        return json.dumps(value, ensure_ascii=False).strip()

    def process_result_value(self, value, dialect):
        """从数据库读取数据：字符串 -> 对象"""
        if value is None:
            return None
        
        # 核心修复：手动处理反序列化，不依赖 dialect._json_deserializer
        if isinstance(value, (dict, list)):
            return value
            
        try:
            # Oracle 21c/23ai 有时会返回 LOB 对象或特殊的包装类型
            # 将其转换为字符串后解析
            str_value = str(value)
            return json.loads(str_value)
        except (ValueError, TypeError, json.JSONDecodeError):
            # 兜底：如果解析失败，返回原始字符串
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