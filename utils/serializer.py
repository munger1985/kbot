# utils/serializer.py
import array
from decimal import Decimal
from datetime import datetime, date
from typing import Any

class SerializerUtils:
    """序列化工具类"""
    
    @staticmethod
    def safe_serialize(obj: Any) -> Any:
        """
        安全序列化方法，处理各种不可序列化的类型
        这是对serialize_value方法的别名，保持向后兼容
        
        Args:
            obj: 需要序列化的对象
            
        Returns:
            可序列化的值
        """
        return SerializerUtils.serialize_value(obj)
    
    @staticmethod
    def serialize_value(value: Any) -> Any:
        """
        递归序列化值，处理各种不可序列化的类型
        
        Args:
            value: 需要序列化的值
            
        Returns:
            可序列化的值
        """
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, array.array):
            return SerializerUtils.array_to_list(value)
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except:
                return str(value)
        elif hasattr(value, 'to_dict'):
            # 优先使用对象的to_dict方法
            return value.to_dict()
        elif hasattr(value, '__dict__'):
            return SerializerUtils.object_to_dict(value)
        elif isinstance(value, (list, tuple)):
            return [SerializerUtils.serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: SerializerUtils.serialize_value(v) for k, v in value.items()}
        else:
            try:
                return str(value)
            except:
                return None
    
    @staticmethod
    def array_to_list(arr: array.array) -> list:
        """
        将array.array转换为list
        
        Args:
            arr: array.array对象
            
        Returns:
            list对象
        """
        try:
            if hasattr(arr, 'tolist'):
                return arr.tolist()
            else:
                return list(arr)
        except Exception as e:
            print(f"转换array.array到list时出错: {e}")
            return []
    
    @staticmethod
    def object_to_dict(obj: Any) -> dict:
        """
        将对象转换为字典
        
        Args:
            obj: 任意对象
            
        Returns:
            字典
        """
        # 如果有to_dict方法，优先使用
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
            
        result = {}
        for attr_name in dir(obj):
            if not attr_name.startswith('_') and attr_name not in ['metadata', 'registry']:
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):  # 排除方法
                        result[attr_name] = SerializerUtils.serialize_value(attr_value)
                except Exception as e:
                    print(f"处理属性 {attr_name} 时出错: {e}")
                    result[attr_name] = None
        return result
    
    @staticmethod
    def model_to_dict(model_instance: Any) -> dict:
        """
        专门用于SQLAlchemy模型的序列化
        
        Args:
            model_instance: SQLAlchemy模型实例
            
        Returns:
            序列化后的字典
        """
        if model_instance is None:
            return {}
        
        result = {}
        for column in model_instance.__table__.columns:
            try:
                value = getattr(model_instance, column.name)
                result[column.name] = SerializerUtils.serialize_value(value)
            except Exception as e:
                print(f"序列化列 {column.name} 时出错: {e}")
                result[column.name] = None
        
        # 处理关系字段
        for rel_name in dir(model_instance):
            if not rel_name.startswith('_') and rel_name not in ['metadata', 'registry']:
                try:
                    rel_value = getattr(model_instance, rel_name)
                    if hasattr(rel_value, '__iter__') and not isinstance(rel_value, (str, dict)):
                        # 处理一对多关系
                        result[rel_name] = [SerializerUtils.model_to_dict(item) for item in rel_value]
                    elif hasattr(rel_value, '__table__'):
                        # 处理多对一关系
                        result[rel_name] = SerializerUtils.model_to_dict(rel_value)
                except Exception as e:
                    print(f"处理关系字段 {rel_name} 时出错: {e}")
        
        return result

# 创建便捷函数
def safe_serialize(obj: Any) -> Any:
    """
    便捷函数，直接使用SerializerUtils.safe_serialize
    
    Args:
        obj: 需要序列化的对象
        
    Returns:
        可序列化的值
    """
    return SerializerUtils.safe_serialize(obj)

def serialize_value(value: Any) -> Any:
    """
    便捷函数，直接使用SerializerUtils.serialize_value
    
    Args:
        value: 需要序列化的值
        
    Returns:
        可序列化的值
    """
    return SerializerUtils.serialize_value(value)