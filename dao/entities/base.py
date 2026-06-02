import json
from decimal import Decimal
from sqlalchemy import TypeDecorator, Text
import array as array_module
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TypeDecorator, Text, Dialect, JSON
from sqlalchemy.dialects.oracle import VECTOR as ORA_VECTOR  # Oracle 23ai+
from core.config.settings import get_embed_config


BaseEntity = declarative_base()

        
class UniversalVector(TypeDecorator):
    """
    支持动态维度的跨库向量适配器
    """
    impl = Text # 默认降级实现
    cache_ok = True

    def __init__(self):
        super().__init__()
        self.dims = get_embed_config().dimensions

    def load_dialect_impl(self, dialect: Dialect):
        # if dialect.name == 'postgresql':
        #     if PG_VECTOR is not None:
        #         # PG 支持不带维度的向量定义：VECTOR
        #         return dialect.type_descriptor(PG_VECTOR(self.dim))
        #     return dialect.type_descriptor(Text())
        if dialect.name == 'oracle':
            if ORA_VECTOR is not None:
                # Oracle 23ai+ 也支持不指定维度的向量定义
                return dialect.type_descriptor(ORA_VECTOR(self.dims) if self.dims else ORA_VECTOR())
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
        # 🎯 核心修正 2：如果是在 Oracle 下，通知 SQLAlchemy 底层使用的是原生 JSON/OSON 存储层
        if dialect.name == 'oracle':
            return dialect.type_descriptor(JSON())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect: Dialect):
        """序列化：将 Python 对象转换为 JSON 字符串或保持原样供驱动处理"""
        if value is None:
            return None
        
        # 兼容高精度 Decimal 并防止中文被转义为 \uXXXX
        def default_encoder(item):
            if isinstance(item, Decimal):
                return float(item)
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