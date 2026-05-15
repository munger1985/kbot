from pydantic import BaseModel, Field
from typing import Any, Type

class SkillParam(BaseModel):
    name: str
    param_type: str  # 允许开发者在 md 里写 "string", "int", "list", "bool" 等
    description: str
    required: bool = True
    default: Any = None

class SkillMetadata(BaseModel):
    name: str
    description: str
    usage_example: str
    category: str = "general"
    params: list[SkillParam] = Field(default_factory=list)
    implementation_class: Type[Any]

    def to_anthropic_tool(self) -> dict[str, Any]:
        """
        将系统内部元数据，完美转化为严格对齐 Anthropic / JSON Schema 规范的 Tool 协议
        """
        # 核心：建立 Python/常识类型 到 标准 JSON Schema 类型的强映射表
        type_mapping = {
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "list": "array",
            "array": "array",
            "dict": "object",
            "object": "object",
            "bool": "boolean",
            "boolean": "boolean",
            "string": "string",
            "str": "string",
            "date": "string"  # 日期在 JSON Schema 中通常作为 string 处理
        }

        properties = {}
        required_params = []
        
        for p in self.params:
            # 安全获取标准类型，如果识别不到，默认退化为 "string" 兜底
            json_type = type_mapping.get(p.param_type.lower(), "string")
            
            properties[p.name] = {
                "type": json_type, 
                "description": p.description
            }
            if p.required:
                required_params.append(p.name)
                
        return {
            "name": self.name,
            # 将功能描述和示例融合，最大化帮助 Claude 理解这个工具什么时候该调、怎么调
            "description": f"【功能】{self.description} | 【使用示例】{self.usage_example}",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required_params
            }
        }