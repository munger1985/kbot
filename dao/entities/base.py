import json
import array as array_module
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TypeDecorator, JSON, Text, Dialect
from sqlalchemy import UnicodeText
from sqlalchemy.dialects.postgresql import JSONB
# from sqlalchemy.dialects.oracle import RAW as ORA_RAW
from sqlalchemy.dialects.oracle import VECTOR as ORA_VECTOR  # Oracle 23ai+
# from pgvector.sqlalchemy import Vector as PG_VECTOR
from sqlalchemy.ext.mutable import MutableList
from loguru import logger

BaseEntity = declarative_base()

class OracleJSON(TypeDecorator):
    """
    针对 Oracle 异步驱动定制的 JSON 类型
    解决 AttributeError: 'OracleDialectAsync_oracledb' object has no attribute '_json_deserializer'
    """
    impl = UnicodeText
    cache_ok = True

    def process_bind_param(self, value, dialect):
        # Handle None - return None for nullable fields
        if value is None:
            logger.debug(f"OracleJSON process_bind_param: value is None, returning None")
            return None

        # 确保只接受有效的 JSON 可序列化类型
        if not isinstance(value, (dict, list, str, int, float, bool)):
            logger.warning(f"OracleJSON Invalid JSON type detected: {type(value)}, value: {value}")
            # 尝试转换为字典或空字典
            if isinstance(value, str):
                try:
                    json_str = json.dumps(json.loads(value), ensure_ascii=False)
                except json.JSONDecodeError:
                    json_str = json.dumps({"raw_string": value}, ensure_ascii=False)
                return json_str
            else:
                return json.dumps({}, ensure_ascii=False)

        # 确保 JSON 输出不包含非法字符
        try:
            # 使用 ensure_ascii=False 保持中文等非 ASCII 字符
            # 使用自定义 default 处理无法序列化的对象
            def json_serializer(obj):
                """自定义 JSON 序列化器,处理常见数据类型"""
                if obj is None:
                    return None
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                if isinstance(obj, Decimal):
                    return float(obj)
                if hasattr(obj, '__dict__'):
                    # 尝试序列化对象的字典属性
                    try:
                        return obj.__dict__
                    except Exception:
                        return str(obj)
                # 兜底:转换为字符串
                return str(obj)

            json_str = json.dumps(value, ensure_ascii=False, default=json_serializer)

            # 验证生成的 JSON 字符串是有效的
            json.loads(json_str)  # 验证 JSON 格式

            # Ensure we never return an empty string
            if not json_str or json_str == '':
                json_str = '{}'

            # Log first few JSON strings for debugging (limit to avoid spam)
            if not hasattr(self, '_log_count'):
                self._log_count = 0
            if self._log_count < 5:
                self._log_count += 1
                logger.debug(f"OracleJSON process_bind_param #{self._log_count}: type={type(value)}, json_str[:200]={repr(json_str[:200])}, json_str length={len(json_str)}")

            return json_str
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"JSON serialization failed: {e}, value type: {type(value)}, value: {str(value)[:500]}")
            # 返回空对象作为后备
            return json.dumps({}, ensure_ascii=False)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # 如果已经是字典/列表，直接返回，避免重复反序列化
        if isinstance(value, (dict, list)):
            return value
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
            # Oracle 23ai/26ai 使用自定义 OracleJSON 类型避免 _json_serializer 错误
            return dialect.type_descriptor(OracleJSON())
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