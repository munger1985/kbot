import array
import oracledb
import numpy as np
from typing import Any, Union, List

class OracleVecHandler:

    def convert(self, vec: Any, to_string: bool = True) -> Union[array.array, str]:
        """
        主转换方法
        
        参数:
            vec: 输入向量
            to_string: 是否转换为字符串格式(否则返回array.array)
            
        返回:
            转换后的向量表示
        """
        
        # 统一转换为列表
        vector_list = self._to_list(vec)
        
        # 验证向量
        self._validate_vector(vector_list)
        
        # 转换为目标格式
        if to_string:
            result = self._to_oracle_string(vector_list)

        else:
            result = self._to_array(vector_list)
            
        return result
    
    def _to_list(self, vec: Any) -> List[float]:
        """转换为Python列表"""
        if isinstance(vec, str):
            return self._parse_string(vec)
        elif isinstance(vec, np.ndarray):
            return vec.astype(np.float64).tolist()
        elif isinstance(vec, (list, tuple)):
            return list(vec)
        elif isinstance(vec, array.array):
            return vec.tolist()
        else:
            raise ValueError(f"不支持的向量类型: {type(vec).__name__}")
    
    def _parse_string(self, vec_str: str) -> List[float]:
        """解析字符串格式的向量"""
        try:
            cleaned = vec_str.strip().strip('[]')
            return [float(x.strip()) for x in cleaned.split(',') if x.strip()]
        except Exception as e:
            raise ValueError(f"无效的向量字符串: {str(e)}")
    
    def _validate_vector(self, vec: List[float]):
        """验证向量有效性"""
        if not vec:
            raise ValueError("向量不能为空")
            
        if not all(isinstance(x, (float, int)) for x in vec):
            raise ValueError("向量必须只包含数值")
    
    def _to_oracle_string(self, vec: List[float]) -> str:
        """转换为Oracle需要的字符串格式"""
        return '[' + ','.join(map(str, vec)) + ']'
    
    def _to_array(self, vec: List[float]) -> array.array:
        """转换为Python数组"""
        return array.array('d', vec)
    
    @staticmethod
    def vector_type_handler(cursor, name, default_type, size, precision, scale):
        if default_type is oracledb.DB_TYPE_VECTOR:
            return cursor.var(default_type, arraysize=size or cursor.arraysize, outconverter=list)