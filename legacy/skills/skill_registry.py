from pydantic import BaseModel, Field
from typing import Any, Type
from skills.base import SkillDomain, SkillRunMode

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
    domain: SkillDomain = SkillDomain.BUSINESS
    run_mode: SkillRunMode = SkillRunMode.READ_ONLY
    params: list[SkillParam] = Field(default_factory=list)
    implementation_class: Type[Any]

    def to_anthropic_tool(self) -> dict[str, Any]:
        """
        将系统内部元数据，完美转化为严格对齐 Anthropic / JSON Schema 规范的 Tool 协议
        """
        type_mapping = {
            "int": "integer", "integer": "integer",
            "float": "number", "number": "number",
            "list": "array", "array": "array",
            "dict": "object", "object": "object",
            "bool": "boolean", "boolean": "boolean",
            "string": "string", "str": "string", "date": "string"
        }

        properties = {}
        required_params = []
        
        for p in self.params:
            json_type = type_mapping.get(p.param_type.lower(), "string")
            properties[p.name] = {
                "type": json_type, 
                "description": p.description
            }
            if p.required:
                required_params.append(p.name)

        # 🚨 核心修复：输出给大模型的工具名称，必须与 SkillManager 内部注册的 Kebab-case 键 100% 强对齐
        # 比如将 "AskGraphSkill" 转换为 "ask-graph-skill"
        import re
        # 下面这行正则可以将大写驼峰转换为小写中划线 (e.g., AskGraphSkill -> ask-graph-skill)
        kebab_name = re.sub(r'(?<!^)(?=[A-Z])', '-', self.name).lower().replace('_', '-')
        # 兜底：如果它本身就已经带了中划线，防止重复处理
        if '--' in kebab_name:
            kebab_name = self.name.lower().replace('_', '-')
                
        return {
            "name": kebab_name, # 🎯 扔给 Claude 的名字现在变成 'ask-graph-skill'
            "description": f"【功能】{self.description} | 【使用示例】{self.usage_example}",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required_params
            }
        }