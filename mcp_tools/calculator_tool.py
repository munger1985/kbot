from loguru import logger
from typing import Any
from mcp_tools import MCPTool, ToolResult
from core.dictionary import MCPToolType


class CalculatorTool(MCPTool):
    """计算器工具"""
    
    def __init__(self):
        super().__init__(
            tool_type=MCPToolType.CALCULATOR,
            tool_name="calculator",
            description="执行数学计算"
        )
    
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            expression = parameters.get("expression", "")
            
            # 安全地执行数学计算
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                raise ValueError("表达式包含不安全字符")
            
            result = eval(expression)
            
            content=f"计算结果: {expression} = {result}"
            
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=[content],
                confidence=1.0,
                metadata={"expression": expression, "result": result}
            )
        except Exception as e:
            logger.error(f"计算失败: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string", 
                    "description": "数学表达式，支持加减乘除和括号"
                }
            },
            "required": ["expression"]
        }
