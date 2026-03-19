import array
import oracledb
import numpy as np
from typing import Any

class OracleVecHandler:
    """Handler for Oracle 23ai vector type conversions.
    
    Provides methods to convert various vector representations (list, numpy array, string)
    to Oracle-compatible formats (array.array or string), and includes type handler
    registration for proper vector data retrieval from Oracle databases.
    """
    def __init__(self, float_type: str = 'f'):
        """Initialize OracleVecHandler with target float precision.

        Args:
            float_type: 'f' for float32 (recommended by Oracle 23ai), 'd' for float64
        """
        self.float_type = float_type

    def convert(self, vec: Any, to_string: bool = False) -> array.array | str:
        """Primary vector conversion method for Oracle compatibility.

        Note: Oracle 23ai recommends binding array.array objects via python-oracledb
        instead of string concatenation for SQL statements, as it's more secure and performant.

        Args:
            vec: Input vector (supports list, tuple, numpy array, array.array, or string format)
            to_string: If True, returns vector as string (e.g., "[1.0,2.0,3.0]") for manual SQL concatenation;
                       If False (default), returns array.array (recommended for parameter binding)

        Returns:
            array.array | str: Oracle-compatible vector representation

        Raises:
            ValueError: If input vector is empty or None
        """
        if vec is None:
            raise ValueError("Vector cannot be empty")

        # 1. Normalize input to list format
        vector_list = self._to_list(vec)
        
        # 2. Validation
        if not vector_list:
            raise ValueError("Vector cannot be empty")

        # 3. Convert to target format
        if to_string:
            # For manual SQL concatenation or specific driver modes
            return '[' + ','.join(map(str, vector_list)) + ']'
        
        # Recommended approach: return array.array (automatically recognized by oracledb driver)
        return array.array(self.float_type, vector_list)

    def _to_list(self, vec: Any) -> list[float]:
        """Internal method to normalize various vector types to list of floats.

        Args:
            vec: Input vector in supported format (list, tuple, numpy array, array.array, string)

        Returns:
            list[float]: Flattened list of float values

        Raises:
            TypeError: If input type is not supported
        """
        if isinstance(vec, (list, tuple)):
            return list(vec)
        if isinstance(vec, np.ndarray):
            # Avoid astype(np.float64) for better performance - direct conversion
            return vec.flatten().tolist()
        if isinstance(vec, array.array):
            return vec.tolist()
        if isinstance(vec, str):
            cleaned = vec.strip().strip('[]')
            return [float(x.strip()) for x in cleaned.split(',') if x.strip()]
        
        raise TypeError(f"Unsupported vector input type: {type(vec).__name__}")

    @staticmethod
    def get_type_handler():
        """Static factory method to create Oracle type handler for vector retrieval.

        Registers a custom output type handler to properly process Oracle DB_TYPE_VECTOR
        columns when fetching data from the database.

        Usage: 
            conn.outputtypehandler = OracleVecHandler.get_type_handler()

        Returns:
            Callable: Oracle output type handler function
        """
        def handler(cursor, name, default_type, size, precision, scale):
            if default_type == oracledb.DB_TYPE_VECTOR:
                # Returns list by default (modify here to return numpy array if needed)
                return cursor.var(oracledb.DB_TYPE_VECTOR, arraysize=cursor.arraysize)
        return handler