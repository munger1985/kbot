from loguru import logger
from typing import Any
import math
import numpy as np
import re
from mcp_tools import MCPTool, ToolResult
from core.dictionary import MCPToolType


class CalculatorTool(MCPTool):
    """科学计算器工具"""
    
    def __init__(self):
        super().__init__(
            tool_type=MCPToolType.CALCULATOR,
            tool_name="calculator",
            description="执行数学计算和科学计算，支持三角函数、对数、指数等复杂运算"
        )
        # 定义安全函数和环境
        self.safe_globals = {
            'abs': abs, 'max': max, 'min': min, 'round': round,
            'pow': pow, 'sum': sum,
            # 数学常数
            'pi': math.pi, 'e': math.e, 'inf': math.inf,
            # 基本数学函数
            'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log, 
            'log10': math.log10, 'log2': math.log2,
            # 三角函数
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'atan2': math.atan2,
            # 双曲函数
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            # 角度转换
            'degrees': math.degrees, 'radians': math.radians,
            # 其他数学函数
            'ceil': math.ceil, 'floor': math.floor, 'factorial': math.factorial,
            'gcd': math.gcd, 'hypot': math.hypot,
            # numpy函数（如果可用）
            'mean': np.mean if hasattr(np, 'mean') else lambda x: sum(x)/len(x),
            'std': np.std if hasattr(np, 'std') else lambda x: math.sqrt(sum((xi - sum(x)/len(x))**2 for xi in x)/(len(x)-1)),
        }
    
    def _preprocess_expression(self, expression: str) -> str:
        """预处理表达式，将常见数学符号转换为Python可识别的形式"""
        # 替换常见的数学符号和函数
        replacements = {
            '×': '*', '÷': '/', '^': '**', 
            '√': 'sqrt', 'π': 'pi',
            'sin⁻¹': 'asin', 'cos⁻¹': 'acos', 'tan⁻¹': 'atan',
            '°': '*pi/180',  # 度转弧度
        }
        
        # 处理根号表示法：√(x) 或 √x
        expression = re.sub(r'√\s*(\d+|\([^)]+\))', r'sqrt(\1)', expression)
        
        # 处理平方、立方等
        expression = re.sub(r'(\d+)\²', r'\1**2', expression)
        expression = re.sub(r'(\d+)\³', r'\1**3', expression)
        
        # 应用替换
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        
        return expression
    
    def _is_safe_expression(self, expression: str) -> bool:
        """检查表达式是否安全"""
        # 允许的字符集（扩展以支持更多数学符号）
        allowed_chars = set('0123456789+-*/.()[]{}<>|&~^%!@#$?=:;,\\ \t\n\r')
        allowed_chars.update(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'))
        
        if not all(c in allowed_chars for c in expression):
            return False
        
        # 禁止危险的关键字和函数调用
        dangerous_patterns = [
            r'__', r'import', r'eval', r'exec', r'compile', r'open',
            r'file', r'os\.', r'sys\.', r'subprocess', r'commands',
            r'breakpoint', r'memoryview', r'bytearray', r'setattr',
            r'delattr', r'property', r'super', r'globals', r'locals',
            r'vars', r'dir', r'type', r'isinstance', r'issubclass'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression):
                return False
        
        return True
    
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            expression = parameters.get("expression", "").strip()
            
            if not expression:
                raise ValueError("表达式不能为空")
            
            # 预处理表达式
            processed_expression = self._preprocess_expression(expression)
            
            # 安全检查
            if not self._is_safe_expression(processed_expression):
                raise ValueError("表达式包含不安全字符或模式")
            
            # 执行计算
            logger.info(f"计算表达式: {expression} -> {processed_expression}")
            result = eval(processed_expression, {"__builtins__": {}}, self.safe_globals)
            
            # 格式化结果
            if isinstance(result, (int, float)):
                if abs(result) > 1e10 or (abs(result) < 1e-10 and result != 0):
                    formatted_result = f"{result:.6e}"
                else:
                    formatted_result = f"{result:.6f}".rstrip('0').rstrip('.')
            else:
                formatted_result = str(result)
            
            content = f"计算结果: {expression} = {formatted_result}"
            
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=[content],
                confidence=1.0,
                metadata={
                    "expression": expression,
                    "processed_expression": processed_expression,
                    "result": result,
                    "formatted_result": formatted_result
                }
            )
            
        except Exception as e:
            logger.error(f"计算失败: {e}")
            error_msg = f"计算失败: {str(e)}"
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=[error_msg],
                confidence=0.0,
                metadata={"error": str(e), "expression": parameters.get("expression", "")}
            )
    
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string", 
                    "description": "数学表达式，支持：\n"
                                 "- 基本运算: +, -, *, /, ** (幂运算)\n"
                                 "- 三角函数: sin, cos, tan, asin, acos, atan\n" 
                                 "- 对数函数: log, log10, log2\n"
                                 "- 其他函数: sqrt, exp, factorial, gcd\n"
                                 "- 常数: pi, e\n"
                                 "- 示例: 'sin(pi/4)', 'sqrt(2)', 'log(100)'"
                }
            },
            "required": ["expression"]
        }