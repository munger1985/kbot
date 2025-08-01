from decimal import Decimal
from json import JSONEncoder
    
class DecimalEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            # 对于整数部分保持整数类型
            if obj == obj.to_integral_value():
                return int(obj)
            return float(obj)
        return super().default(obj)