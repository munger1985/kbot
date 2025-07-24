from decimal import Decimal
from json import JSONEncoder

class DecimalEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)  # 或者 str(obj) 如果需要保留精度
        return super(DecimalEncoder, self).default(obj)