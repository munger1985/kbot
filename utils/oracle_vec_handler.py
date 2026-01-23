import array
import oracledb
import numpy as np
from typing import Any

class OracleVecHandler:
    def __init__(self, float_type: str = 'f'):
        """
        :param float_type: 'f' 代表 float32 (推荐), 'd' 代表 float64
        """
        self.float_type = float_type

    def convert(self, vec: Any, to_string: bool = False) -> array.array | str:
        """
        主转换方法。
        注意：Oracle 23ai 推荐通过 python-oracledb 绑定 array.array 对象，
        而不是使用字符串拼装，这样更安全且性能更好。
        """
        if vec is None:
            raise ValueError("向量不能为空")

        # 1. 统一转换为 list
        vector_list = self._to_list(vec)
        
        # 2. 验证
        if not vector_list:
            raise ValueError("向量不能为空")

        # 3. 转换为目标格式
        if to_string:
            # 适用于手动拼接 SQL 或某些特定驱动模式
            return '[' + ','.join(map(str, vector_list)) + ']'
        
        # 推荐做法：返回 array.array，oracledb 驱动会自动识别
        return array.array(self.float_type, vector_list)

    def _to_list(self, vec: Any) -> list[float]:
        if isinstance(vec, (list, tuple)):
            return list(vec)
        if isinstance(vec, np.ndarray):
            # 避免使用 astype(np.float64)，直接根据需求转换提高效率
            return vec.flatten().tolist()
        if isinstance(vec, array.array):
            return vec.tolist()
        if isinstance(vec, str):
            cleaned = vec.strip().strip('[]')
            return [float(x.strip()) for x in cleaned.split(',') if x.strip()]
        
        raise TypeError(f"不支持的向量输入类型: {type(vec).__name__}")

    @staticmethod
    def get_type_handler():
        """
        静态工厂方法，用于在连接时注册
        用法: conn.outputtypehandler = OracleVecHandler.get_type_handler()
        """
        def handler(cursor, name, default_type, size, precision, scale):
            if default_type == oracledb.DB_TYPE_VECTOR:
                # 默认返回 list，如果需要 numpy 也可以在这里改
                return cursor.var(oracledb.DB_TYPE_VECTOR, arraysize=cursor.arraysize)
        return handler