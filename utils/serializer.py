# utils/serializer.py
import array
from decimal import Decimal
from datetime import datetime, date
from typing import Any

class SerializerUtils:
    """Serialization utility class.
    
    Provides recursive serialization methods to convert non-serializable Python objects
    (e.g., datetime, Decimal, array.array, custom objects) into JSON-serializable formats
    like strings, lists, and dictionaries.
    """
    
    @staticmethod
    def safe_serialize(obj: Any) -> Any:
        """
        Safe serialization method that handles various non-serializable types.
        This is an alias for serialize_value to maintain backward compatibility.
        
        Args:
            obj: Object to be serialized
            
        Returns:
            JSON-serializable value
        """
        return SerializerUtils.serialize_value(obj)
    
    @staticmethod
    def serialize_value(value: Any) -> Any:
        """
        Recursively serialize values, handling various non-serializable types.
        
        Converts complex types to JSON-serializable formats while preserving data integrity:
        - datetime/date → ISO format string
        - Decimal → float
        - array.array → list
        - bytes → UTF-8 string (with error handling)
        - Objects → dict (via to_dict method or __dict__)
        - Collections → recursively serialized
        
        Args:
            value: Value to be serialized
            
        Returns:
            JSON-serializable value
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
            # Prefer object's native to_dict method if available
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
        Convert array.array object to standard Python list.
        
        Args:
            arr: array.array object to convert
            
        Returns:
            list: Converted list of values
        """
        try:
            if hasattr(arr, 'tolist'):
                return arr.tolist()
            else:
                return list(arr)
        except Exception as e:
            print(f"Error converting array.array to list: {e}")
            return []
    
    @staticmethod
    def object_to_dict(obj: Any) -> dict:
        """
        Convert arbitrary object to dictionary by extracting attributes.
        
        Excludes private attributes (starting with '_'), metadata, and registry attributes,
        and skips callable methods/properties.
        
        Args:
            obj: Any Python object to convert
            
        Returns:
            dict: Dictionary of object attributes with serialized values
        """
        # Use to_dict method if available (highest priority)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
            
        result = {}
        for attr_name in dir(obj):
            if not attr_name.startswith('_') and attr_name not in ['metadata', 'registry']:
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):  # Exclude methods/functions
                        result[attr_name] = SerializerUtils.serialize_value(attr_value)
                except Exception as e:
                    print(f"Error processing attribute {attr_name}: {e}")
                    result[attr_name] = None
        return result
    
    @staticmethod
    def model_to_dict(model_instance: Any) -> dict:
        """
        Specialized serialization for SQLAlchemy model instances.
        
        Serializes model columns and relationship fields:
        - Columns: Serialized using standard serialize_value method
        - One-to-many relationships: List of serialized model dicts
        - Many-to-one relationships: Serialized model dict
        
        Args:
            model_instance: SQLAlchemy model instance
            
        Returns:
            dict: Serialized model data including columns and relationships
        """
        if model_instance is None:
            return {}
        
        result = {}
        # Serialize table columns
        for column in model_instance.__table__.columns:
            try:
                value = getattr(model_instance, column.name)
                result[column.name] = SerializerUtils.serialize_value(value)
            except Exception as e:
                print(f"Error serializing column {column.name}: {e}")
                result[column.name] = None
        
        # Process relationship fields
        for rel_name in dir(model_instance):
            if not rel_name.startswith('_') and rel_name not in ['metadata', 'registry']:
                try:
                    rel_value = getattr(model_instance, rel_name)
                    if hasattr(rel_value, '__iter__') and not isinstance(rel_value, (str, dict)):
                        # Handle one-to-many relationships
                        result[rel_name] = [SerializerUtils.model_to_dict(item) for item in rel_value]
                    elif hasattr(rel_value, '__table__'):
                        # Handle many-to-one relationships
                        result[rel_name] = SerializerUtils.model_to_dict(rel_value)
                except Exception as e:
                    print(f"Error processing relationship field {rel_name}: {e}")
        
        return result

# Create convenience functions
def safe_serialize(obj: Any) -> Any:
    """
    Convenience function that directly calls SerializerUtils.safe_serialize.
    
    Args:
        obj: Object to be serialized
        
    Returns:
        JSON-serializable value
    """
    return SerializerUtils.safe_serialize(obj)

def serialize_value(value: Any) -> Any:
    """
    Convenience function that directly calls SerializerUtils.serialize_value.
    
    Args:
        value: Value to be serialized
        
    Returns:
        JSON-serializable value
    """
    return SerializerUtils.serialize_value(value)